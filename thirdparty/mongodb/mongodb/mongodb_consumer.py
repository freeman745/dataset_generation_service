from typing import Any, Dict, Optional
from pathlib import Path
import argparse
import os
import sys
from datetime import datetime
import time
import traceback
import redis
import logging
import yaml
import bson
from mongodb.constants import (
    GeneralKeys as GKeys,
    LogMethodNames as LMNames,
    WS_NAME,
    ITEM_CONFIG_PATH,
    MONGODB_LOGGER_QUEUE,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    ACTION,
    DATA,
    SAFIE_PATH,
    PR_SEP,
)
from mongodb.mongodbif_base import get_db
from mongodb.utils.caller_info import CallerInfo
from mongodb.utils.decorators import exception_handler
from mongodb.utils.logger_utils import get_oid_dir, get_timestamp
from mongodb.utils.misc import get_git_info


ROMSPICKINGSYSTEM_DIR = str(Path(__file__).resolve().parent.parent.parent.parent)
RPS_ENV_SETTINGS_DIR = os.path.join(
    ROMSPICKINGSYSTEM_DIR, "launcher", "config", "RPS_environment_settings"
)


class MongoDBConsumer:
    def __init__(
        self, cfg_obj: Optional[Dict] = None, logger=None, log_level: str = "INFO"
    ) -> None:
        self.cfg_obj = cfg_obj
        self._set_consumer()
        self._set_db(self.cfg_obj)
        self._set_logger(logger, log_level)
        self._configure()
        self._logger.info(f"{self.__class__.__name__} has started in {log_level} mode.")

    def _set_db(self, cfg_obj):
        if not cfg_obj:
            with open(ITEM_CONFIG_PATH) as f:
                cfg_obj: dict = yaml.safe_load(f)
        self._cfg_obj = cfg_obj
        self._db = get_db(cfg_obj=cfg_obj)

    def _set_logger(self, logger=None, log_level: str = "INFO"):
        if logger is None:
            logger = logging.getLogger(self.__class__.__name__)
            logger.setLevel(getattr(logging, log_level))
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            stdout_handler.setFormatter(stdout_formatter)
            logger.addHandler(stdout_handler)
        self._logger = logger

    def _set_consumer(self):
        self._consumer = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

    def _configure(self):
        with open(SAFIE_PATH) as f:
            try:
                self.safie_url_templates = yaml.safe_load(f).get(WS_NAME, [])
            except:
                self.safie_url_templates = []

    def _consume(self, action_name: str, data: Dict):
        while self._db._db is None:
            self._set_db(self.cfg_obj)
            time.sleep(0.5)
        try:
            if action_name == LMNames.new_entry:
                result = self.new_entry(**data)
            elif action_name == LMNames.insert:
                result = self.insert(**data)
            elif action_name == LMNames.append:
                result = self.append(**data)
            elif action_name == LMNames.backup:
                result = self.backup(**data)
            elif action_name == LMNames.remove:
                result = self.remove(**data)
            else:
                self._logger.error(
                    f"Action must be one of: "
                    f"{', '.join([elem for elem in dir(LMNames) if not elem.startswith('__')])}. "
                    f"<{action_name}> was passed"
                )
                return
            self._logger.debug(result)
        except:
            self._logger.error(f"Error during logging: {traceback.format_exc()}")

    def run(self):
        while True:
            message = self._consumer.blpop(MONGODB_LOGGER_QUEUE, timeout=5 * 60)
            if message:
                _, message = message
                try:
                    kwargs = bson.BSON(message).decode()
                except:
                    self._logger.error(
                        f"Error while desirializing {message}\n{traceback.format_exc()}"
                    )
                action_name, data = kwargs[ACTION], kwargs[DATA]
                self._logger.debug(f"Consume action {action_name}, with data: {data}")
                self._consume(action_name, data)
            else:
                self._logger.info(f"{self.__class__.__name__} is still running...")
                # time.sleep(1)

    def new_entry(
        self,
        collection: str,
        order_id: str,
        data_dir: str,
        timestamp: datetime,
    ):
        if self._db.exists(collection, {GKeys.order_id: order_id}):
            self._logger.debug(f"entry for {order_id} already exists (first check)")
            return False
        git_info = get_git_info(ROMSPICKINGSYSTEM_DIR)
        env_git_info = get_git_info(RPS_ENV_SETTINGS_DIR)
        metadata = {
            GKeys.timestamp: timestamp,
            GKeys.oid_dir: get_oid_dir(data_dir, collection, order_id),
            GKeys.ws_name: WS_NAME,
            GKeys.git_sha: git_info["sha"],
            GKeys.git_branch: git_info["branch"],
            GKeys.env_git_sha: env_git_info["sha"],
            GKeys.env_git_branch: env_git_info["branch"],
            GKeys.order_line: self._get_order_line(order_id),
            GKeys.safie_urls: self._get_safie_urls(timestamp),
        }
        insert_dict = {GKeys.order_id: order_id, GKeys.metadata: metadata}
        if not self._db.exists(collection, {GKeys.order_id: order_id}):
            rc = self._db.insert_one(
                collection_name=collection, insert_dict=insert_dict
            )
            if rc:
                self._logger.info(f"New entry: {order_id}")
            return rc
        self._logger.debug(f"entry for {order_id} already exists")
        return False

    def insert(
        self,
        collection: str,
        order_id: str,
        key: Optional[str],
        value: Any,
        caller_info_dict: Optional[Dict] = None,
        aux: Optional[Dict] = None,
        condition: Optional[Dict] = None,
        add_metadata: bool = False,
        add_method_in_key: bool = False,
        add_key_prefix: bool = True,
    ):
        if key is None:
            return
        caller_info = CallerInfo(**caller_info_dict)
        if not add_method_in_key:
            caller_info.method_name = None
        if add_key_prefix:
            key = self._get_prefix(caller_info.cls_name, caller_info.method_name) + key
        if add_metadata or aux:
            payload = {GKeys.data: value}
            if aux:
                payload[GKeys.aux] = aux
            if add_metadata:
                metadata = {
                    GKeys.timestamp: get_timestamp(),
                    GKeys.lineno: caller_info.lineno,
                    GKeys.file_name: caller_info.file_name,
                }
                payload.update(metadata)
        else:
            payload = value
        condition = self._add_order_id(order_id, condition)
        if GKeys.order_id in condition:
            res = self._db.insert(key, payload, collection, condition)
            return res
        return False

    def append(
        self,
        collection: str,
        order_id: str,
        key: Optional[str],
        value: Any,
        caller_info_dict: Optional[Dict] = None,
        aux: Optional[Dict] = None,
        condition: Optional[Dict] = None,
        add_metadata: bool = False,
        add_method_in_key: bool = False,
        add_key_prefix: bool = True,
    ):
        if key is None:
            return
        caller_info = CallerInfo(**caller_info_dict)
        if caller_info and not add_method_in_key:
            caller_info.method_name = None
        if add_key_prefix:
            key = self._get_prefix(caller_info.cls_name, caller_info.method_name) + key
        if add_metadata or aux:
            payload = {GKeys.data: value}
            if aux:
                payload[GKeys.aux] = aux
            if add_metadata:
                metadata = {
                    GKeys.timestamp: get_timestamp(),
                    GKeys.lineno: caller_info.lineno,
                    GKeys.file_name: caller_info.file_name,
                }
                payload.update(metadata)
        else:
            payload = value
        condition = self._add_order_id(order_id, condition)
        if GKeys.order_id in condition:
            self._db.append(key, payload, collection, condition)
            return True
        return False

    def backup(self, collection: str, order_id: str, data_dir: str):
        if order_id is None:
            return
        oid_dir = get_oid_dir(data_dir, collection, order_id)
        if not os.path.isdir(oid_dir):
            return
        condition = self._add_order_id(order_id)
        filepath = os.path.join(oid_dir, "item.json")
        self._db.export_one(
            collection_name=collection, condition=condition, filepath=filepath
        )

    def remove(self, collection: str, order_id: str):
        from mongodb import MongoDBReader

        if order_id is None:
            return
        condition = {GKeys.order_id: order_id}
        res = self._db.delete_one(collection, condition)
        return res

    @staticmethod
    def _get_order_line(order_id):
        parts = order_id.split("_")
        if parts:
            return parts[0]

    def _get_safie_urls(self, timestamp):
        safie_urls = []
        for url in self.safie_url_templates:
            timestamp = int(timestamp.timestamp()) * 1000
            url = f"{url}?timestamp={timestamp}"
            safie_urls.append(url)
        return safie_urls

    @staticmethod
    def _add_order_id(order_id: str, condition: Optional[Dict] = None):
        if condition is None:
            condition = dict()
        condition.setdefault(GKeys.order_id, order_id)
        return condition

    @staticmethod
    def _get_prefix(cls_name: str, method_name: str):
        prefix = ""
        if cls_name:
            prefix += cls_name
            prefix += PR_SEP
        if method_name:
            prefix += method_name
            prefix += PR_SEP
        return prefix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MongoDB Consumer.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    log_level = "DEBUG" if args.debug else "INFO"
    mongodb_consumer = MongoDBConsumer(log_level=log_level)
    mongodb_consumer.run()

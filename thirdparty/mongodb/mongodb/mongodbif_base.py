#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2022 Masaru Morita <masaru.morita@roms.inc>
#

"""! mongodb_base
@brief Accesser to mongodb database.
"""
from typing import Dict, List, Optional, Tuple, Any, Union
import os
import logging
import yaml
import traceback
from pymongo import MongoClient, timeout
from pymongo.collection import Collection
from .constants import DOCUMENTS_DIR, GeneralKeys

# from apap_pkgs.dbif.constants import const


class MongoDBIFBase:
    """! MongoDBIF Base."""

    def __init__(
        self,
        logger=None,
        host: Optional[str] = None,
        cloud_host: Optional[str] = None,
        port: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        db_name: Optional[str] = None,
        timeout_sec: Optional[str] = None,
    ):
        """!Initializer
        @param[in] logger logger object to be used.
        """
        self._mongodb_client = None
        self._db = None
        self._collection = None

        self._host = host
        self._cloud_host = cloud_host
        self._port = port
        self._username = username
        self._password = password
        self._db_name = db_name
        self._timeout_sec = timeout_sec
        self._set_logger(logger)

    def _set_logger(self, logger=None, log_level: str = "INFO"):
        if logger is None:
            logger = logging.getLogger(self.__class__.__name__)
            logger.setLevel(getattr(logging, log_level))
        self._logger = logger

    def _configure(
        self, config_file: Optional[str] = None, cfg_obj: Optional[Dict] = None
    ) -> bool:
        self._config_file = config_file
        self._is_configured = False
        if cfg_obj is None:
            try:
                with open(config_file) as f:
                    cfg_obj: dict = yaml.safe_load(f)
            except:
                return False
        if self._host is None:
            self._host = cfg_obj.get("host")
        if self._cloud_host is None:
            self._cloud_host = cfg_obj.get("cloud_host")
        if self._port is None:
            self._port = cfg_obj.get("port")
        if self._username is None:
            self._username = cfg_obj.get("username")
        if self._password is None:
            self._password = cfg_obj.get("password")
        if self._db_name is None:
            self._db_name = cfg_obj.get("db_name")
        if self._timeout_sec is None:
            self._timeout_sec = cfg_obj.get("timeout_sec", 3) or self._timeout_sec

        if self._host is not None:
            self._mongodb_client = MongoClient(
                self._host,
                self._port,
                username=self._username,
                password=self._password,
                maxPoolSize=20,
            )
        else:
            self._mongodb_client = MongoClient(self._cloud_host, maxPoolSize=20)

        with timeout(3):
            try:
                self._mongodb_client.server_info()
            except:
                return False

        self._db = self._mongodb_client[self._db_name]
        self._is_configured = True

        return True

    def get_status(self) -> bool:
        with timeout(3):
            try:
                self._mongodb_client.server_info()
            except:
                return False
        return True

    def insert_one(self, collection_name: str, insert_dict: Dict) -> bool:
        """!insert_one
        @brief Insert one document to mongodb
        @param[in] collection_name: Name of a target collection_name
        @param[in] insert_dict: Dictionary to insert to mongodb
        @retval const.DBIF_OK: Successfully inserted an item
        @retval const.DBIF_ERR_FAILED_TO_CONNECT: Failed to connect to mongodb
        """
        collection = getattr(self._db, collection_name)
        try:
            with timeout(self._timeout_sec):
                collection = getattr(self._db, collection_name)
                collection.insert_one(insert_dict)
        except Exception:
            self._logger.error(traceback.format_exc())
            return False

        # return const.DBIF_OK
        return True

    def insert(self, key, value, collection_name: str, condition: Dict) -> bool:
        try:
            with timeout(self._timeout_sec):
                collection = getattr(self._db, collection_name)
                if condition:
                    collection.update_one(condition, {"$set": {key: value}})
                else:
                    collection.insert_one({key: value})
        except Exception:
            log_msg = (
                f"{self.__class__.__name__}.insert(): key: {key}, value: {value}, "
                f"collection_name: {collection_name}, condition: {condition}"
                f"\nError:\n{traceback.format_exc()}"
            )
            self._logger.error(log_msg)
            return False
        return True

    def append(
        self, key: str, value: Any, collection_name: str, condition: Dict
    ) -> bool:
        try:
            with timeout(self._timeout_sec):
                collection: Collection = getattr(self._db, collection_name)
                collection.update_one(condition, {"$push": {key: value}}, upsert=True)
        except Exception:
            log_msg = (
                f"{self.__class__.__name__}.append(): key: {key}, value: {value}, "
                f"collection_name: {collection_name}, condition: {condition}"
                f"\nError:\n{traceback.format_exc()}"
            )
            self._logger.error(log_msg)
            return False
        return True

    def update(
        self,
        update_dict: Dict[str, Any],
        collection_name: str,
        condition: Dict,
    ) -> bool:
        try:
            with timeout(self._timeout_sec):
                collection: Collection = getattr(self._db, collection_name)
                collection.update_one(condition, update_dict)
        except Exception:
            log_msg = (
                f"{self.__class__.__name__}.update(): update_dict: {update_dict}, "
                f"collection_name: {collection_name}, condition: {condition}"
                f"\nError:\n{traceback.format_exc()}"
            )
            self._logger.error(log_msg)
            return False
        return True

    def delete_one(self, collection_name: str, condition: Dict) -> bool:
        """!delete_one
        @brief Delete one document in mongodb
        @param[in] collection_name: Name of a target collection_name
        @param[in] condition: Condition of a dictionary to search a document to delete
        @retval const.DBIF_OK: Successfully deleted an item
        @retval const.DBIF_ERR_FAILED_TO_CONNECT: Failed to connect to mongodb
        """
        collection = getattr(self._db, collection_name)
        try:
            with timeout(self._timeout_sec):
                collection = getattr(self._db, collection_name)
                collection.delete_one(condition)
        except Exception:
            log_msg = f"{self.__class__.__name__}.delete_one(): "
            f"collection_name: {collection_name}, condition: {condition}"
            f"\nError:\n{traceback.format_exc()}"
            self._logger.error(log_msg)
            return False
        return True

    def delete_many(self, collection_name: str, condition=None) -> bool:
        """!delete_many
        @brief Delete multiple documents in mongodb
        @param[in] collection_name: Name of a target collection_name
        @param[in] condition: Condition of a dictionary to search documents to delete
        @retval const.DBIF_OK: Successfully deleted items
        @retval const.DBIF_ERR_FAILED_TO_CONNECT: Failed to connect to mongodb
        """
        collection = getattr(self._db, collection_name)
        if condition is None:
            condition = dict()
        try:
            with timeout(self._timeout_sec):
                collection = getattr(self._db, collection_name)
                collection.delete_many(condition)
        except Exception:
            log_msg = f"{self.__class__.__name__}.delete_many(): "
            f"collection_name: {collection_name}, condition: {condition}"
            f"\nError:\n{traceback.format_exc()}"
            self._logger.error(log_msg)
            return False

        return True

    def unset_one(
        self, fields: Union[str, List[str]], collection_name: str, condition: Dict
    ) -> bool:
        if isinstance(fields, str):
            fields = [fields]
        if not isinstance(fields, dict):
            unset_fields = {field: "" for field in fields}

        try:
            with timeout(self._timeout_sec):
                collection: Collection = getattr(self._db, collection_name)
                collection.update_one(condition, {"$unset": unset_fields})
        except Exception:
            log_msg = f"{self.__class__.__name__}.unset_one(): fields: {fields}, "
            f"collection_name: {collection_name}, condition: {condition}"
            f"\nError:\n{traceback.format_exc()}"
            self._logger.error(log_msg)
            return False
        return True

    def unset_many(
        self, fields: Union[str, List[str]], collection_name: str, condition: Dict
    ) -> bool:
        if isinstance(fields, str):
            fields = [fields]
        if not isinstance(fields, dict):
            fields = {field: "" for field in fields}

        try:
            with timeout(self._timeout_sec):
                collection: Collection = getattr(self._db, collection_name)
                collection.update_many(condition, {"$unset": fields})
        except Exception:
            log_msg = f"{self.__class__.__name__}.unset_many(): fields: {fields}, "
            f"collection_name: {collection_name}, condition: {condition}"
            f"\nError:\n{traceback.format_exc()}"
            self._logger.error(log_msg)
            return False
        return True

    def find(
        self,
        collection_name: str,
        condition: Dict,
        sort: Tuple[str, int] = None,
        limit: Optional[int] = None,
        projection: Optional[Dict] = None,
        allow_disk_use: bool = False,
        offset: Optional[int] = None,
        **kwargs,
    ):
        """!find
        @brief Find multiple documents in mongodb
        @param[in] collection_name: Name of a target collection_name
        @param[in] condition: Condition of a dictionary to search documents to delete
        @retval const.DBIF_OK: Successfully found items
        @retval const.DBIF_ERR_FAILED_TO_CONNECT: Failed to connect to mongodb
        """
        try:
            with timeout(self._timeout_sec):
                collection: Collection = getattr(self._db, collection_name)
                result = collection.find(
                    condition,
                    projection=projection,
                    allow_disk_use=allow_disk_use,
                    **kwargs,
                ).batch_size(50)
                if offset:
                    result = result.skip(offset)
                if sort is not None:
                    result = result.sort(*sort)
                if limit is not None:
                    result = result.limit(max(0, limit))
        except Exception:
            log_msg = f"{self.__class__.__name__}.find(): "
            f"collection_name: {collection_name}, condition: {condition}"
            f"\nError:\n{traceback.format_exc()}"
            self._logger.error(log_msg)
            return False, None
        return True, result

    def exists(self, collection_name: str, condition: Dict) -> bool:
        return self._db[collection_name].count_documents(condition, limit=1) >= 1

    def find_one(
        self,
        collection_name: str,
        condition: Dict,
        projection: Optional[Dict] = None,
        **kwargs,
    ) -> Tuple[bool, Dict]:
        """!find
        @brief Find multiple documents in mongodb
        @param[in] collection_name: Name of a target collection_name
        @param[in] condition: Condition of a dictionary to search documents to delete
        @retval const.DBIF_OK: Successfully found items
        @retval const.DBIF_ERR_FAILED_TO_CONNECT: Failed to connect to mongodb
        """
        try:
            with timeout(self._timeout_sec):
                collection: Collection = getattr(self._db, collection_name)
                result = collection.find_one(condition, projection=projection, **kwargs)
        except Exception:
            log_msg = f"{self.__class__.__name__}.find_one(): "
            f"collection_name: {collection_name}, condition: {condition}"
            f"\nError:\n{traceback.format_exc()}"
            self._logger.error(log_msg)
            return False, dict()
        if result is None:
            return False, result
        return True, result

    def get_index_information(self, collection_name: str):
        try:
            with timeout(self._timeout_sec):
                collection: Collection = getattr(self._db, collection_name)
                return collection.index_information()
        except Exception:
            log_msg = f"{self.__class__.__name__}.get_index_information(): "
            f"collection_name: {collection_name}"
            f"\nError:\n{traceback.format_exc()}"
            self._logger.error(log_msg)
            return None

    def create_index(
        self,
        collection_name: str,
        index: str,
        name: Optional[str] = None,
        unique: bool = True,
    ) -> bool:
        name = name or index
        index_info = self.get_index_information(collection_name)
        if name in index_info:
            return True
        try:
            with timeout(30):
                collection: Collection = getattr(self._db, collection_name)
                collection.create_index(
                    index, name=name or index, unique=unique, background=True
                )
        except Exception:
            log_msg = (
                f"{self.__class__.__name__}.create_index(): "
                f"collection_name: {collection_name}, index: {index}, "
                f"name: {name}, unique: {unique}"
                f"\nError:\n{traceback.format_exc()}"
            )
            self._logger.error(log_msg)
            return False
        return True

    def export_one(
        self,
        collection_name: str,
        condition: Dict,
        filepath: Optional[str] = None,
    ) -> Tuple[bool, Dict]:
        from bson.json_util import dumps

        rec, result = self.find_one(collection_name, condition)
        if not rec:
            return rec, result
        if filepath is None:
            output_dir = os.path.join(DOCUMENTS_DIR, collection_name)
            filename = f"{result[GeneralKeys.order_id]}.json"
            filepath = os.path.join(output_dir, filename)
        else:
            output_dir = os.path.dirname(filepath)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(filepath, "w") as f:
            f.write(dumps(result, indent=4))
        return True, result

    def export_many(
        self,
        collection_name: str,
        condition: Dict,
    ) -> Tuple[bool, List[Dict]]:
        from bson.json_util import dumps

        rec, results = self.find(collection_name, condition)
        if not rec:
            return rec
        output_dir = os.path.join(DOCUMENTS_DIR, collection_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        results = list(results)
        for result in results:
            filename = f"{result[GeneralKeys.order_id]}.json"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w") as f:
                f.write(dumps(result, indent=4))
        return True, results

    def import_one(self, collection_name: str, filepath: str) -> bool:
        from bson.json_util import loads

        with open(filepath) as f:
            s = f.read()
        insert_dict = loads(s)
        rec = self.insert_one(collection_name, insert_dict)
        return rec

    def import_many(self, collection_name: str, filepaths: List[str]) -> List[bool]:
        recs = []
        for filepath in filepaths:
            rec = self.import_one(collection_name, filepath)
            recs.append(rec)
        return recs


def get_db(config_file: Optional[str] = None, cfg_obj: Optional[Dict] = None):
    db_base = MongoDBIFBase()
    db_base._configure(config_file=config_file, cfg_obj=cfg_obj)
    return db_base

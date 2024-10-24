#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2023 Ozgur Yilmaz <ozgur.yilmaz@roms.inc>
#

"""!Logs execution flow and errors (i.e. all important events) into mongodb
@brief Log execution flow
"""
from typing import Any, Dict, Optional, Union
from dataclasses import asdict
from datetime import datetime
from time import time
import os
import logging
import traceback
import inspect
from dataclasses import asdict
import numpy as np
import yaml
import redis
import bson

from .constants import (
    ITEM_COLLECTION,
    ITEM_CONFIG_PATH,
    DATE_ID,
    MONGODB_LOGGER_QUEUE,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    ACTION,
    DATA,
    GeneralKeys as GKeys,
    LogMethodNames as LMNames,
    EventTypes,
)
from .utils.decorators import exception_handler
from .utils.logger_utils import (
    get_timestamp,
    get_oid_dir,
    is_oid_valid,
    get_date_id,
)
from .utils import (
    save_array,
    np2normal,
    CallerInfo,
)


DEACTIVATE = os.getenv("DEACTIVATE_MONGODB_LOGGER")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


class MongoDBLogger:
    """! MongoDB Logger."""

    def __init__(
        self,
        logger=None,
        log_level: str = LOG_LEVEL,
        internal_log_level: str = "INFO",
        collection: Optional[str] = None,
        name=None,
    ):
        self.set_name(name)
        self._set_internal_logger(logger, internal_log_level)
        self._order_id = None
        self._oid_dir = None
        self.set_log_level(log_level)
        self.configure()
        self._set_publiser()
        self._collection = collection or ITEM_COLLECTION

    @exception_handler
    def configure(
        self,
        cfg_yaml: str = ITEM_CONFIG_PATH,
        data_dir_path: Optional[str] = None,
    ):
        if DEACTIVATE:
            return
        self._set_data_dir(data_dir_path, cfg_yaml)

    def set_name(self, name: Optional[str] = None):
        if name:
            self.name = name
        else:
            current_frame = inspect.currentframe()
            caller_frame = current_frame.f_back.f_back
            caller_class_name = caller_frame.f_locals.get("self", None)
            if caller_class_name:
                self.name = caller_class_name.__class__.__name__
            else:
                self.name = caller_frame.f_globals["__name__"]

    def set_log_level(self, log_level: Union[str, int]):
        if isinstance(log_level, str):
            log_level = EventTypes[log_level].value
        self.log_level = log_level

    def _set_publiser(self):
        self._publisher = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

    def _publish(self, action: str, data: Dict):
        message = {ACTION: action, DATA: data}
        try:
            message_str = bson.BSON.encode(message)
        except:
            log_msg = f"Could not encode {message=}\n{traceback.format_exc()}"
            self._logger.error(log_msg)
        else:
            self._publisher.rpush(MONGODB_LOGGER_QUEUE, message_str)

    @exception_handler
    def new_entry(self, order_id: str):
        if DEACTIVATE:
            return
        if order_id == DATE_ID:
            order_id = get_date_id()
        if not is_oid_valid(order_id):
            self._order_id = None
            return
        self._order_id = order_id
        timestamp = get_timestamp()
        data = {
            GKeys.collection: self._collection,
            GKeys.order_id: order_id,
            GKeys.data_dir: self.data_dir,
            GKeys.timestamp: timestamp,
        }
        self._publish(LMNames.new_entry, data)

    @exception_handler
    def insert(
        self,
        key: Optional[str],
        value: Any,
        caller_info: Optional[CallerInfo] = None,
        log_level: EventTypes = EventTypes.INFO,
        has_np: bool = False,
        aux: Optional[Dict] = None,
        condition: Optional[Dict] = None,
        add_metadata: bool = False,
        add_method_in_key: bool = False,
        add_key_prefix: bool = True,
    ):
        if DEACTIVATE or self._order_id is None or log_level.value > self.log_level:
            return
        if caller_info is None:
            caller_info = CallerInfo(cls_name=self.name)
        if has_np:
            value = np2normal(value)
        data = {
            GKeys.collection: self._collection,
            GKeys.order_id: self._order_id,
            GKeys.key: key,
            GKeys.value: value,
            "caller_info_dict": asdict(caller_info),
            "aux": aux,
            "condition": condition,
            "add_metadata": add_metadata,
            "add_method_in_key": add_method_in_key,     
            "add_key_prefix": add_key_prefix,
        }
        self._publish(LMNames.insert, data)

    @exception_handler
    def append(
        self,
        key: Optional[str],
        value: Any,
        caller_info: Optional[CallerInfo] = None,
        log_level: EventTypes = EventTypes.INFO,
        has_np: bool = False,
        aux: Optional[Dict] = None,
        condition: Optional[Dict] = None,
        add_metadata: bool = False, 
        add_method_in_key: bool = False,
        add_key_prefix: bool = True,
    ):
        if DEACTIVATE or self._order_id is None or log_level.value > self.log_level:
            return
        if caller_info is None:
            caller_info = CallerInfo(cls_name=self.name)
        if has_np:
            value = np2normal(value)
        data = {
            GKeys.collection: self._collection,
            GKeys.order_id: self._order_id,
            GKeys.key: key,
            GKeys.value: value,
            "caller_info_dict": asdict(caller_info),
            "aux": aux,
            "condition": condition,
            "add_metadata": add_metadata,
            "add_method_in_key": add_method_in_key,
            "add_key_prefix": add_key_prefix,
        }
        self._publish(LMNames.append, data)

    @exception_handler
    def append_event(
        self,
        brief: str,
        event_type: int,
        caller_info: Optional[CallerInfo] = None,
        aux: Optional[Dict] = None,
        **kwargs,
    ):
        if DEACTIVATE or self._order_id is None or event_type > self.log_level:
            return
        start = time()
        if caller_info is None:
            caller_info = CallerInfo(cls_name=self.name)
        key = GKeys.events
        value = {
            GKeys.brief: str(brief),
            GKeys.event_type: event_type,
            GKeys.timestamp: get_timestamp(),
            **asdict(caller_info),
        }
        if aux is not None:
            value[GKeys.aux] = aux
        self.append(
            key,
            value,
            caller_info=caller_info,
            add_metadata=False,
            add_key_prefix=False,
            **kwargs,
        )
        self._logger.debug(f"append_event time {time()-start}")

    @exception_handler
    def debug(self, *args, **kwargs):
        self._append_level_event(EventTypes.DEBUG.value, args, kwargs)

    @exception_handler
    def info(self, *args, **kwargs):
        self._append_level_event(EventTypes.INFO.value, args, kwargs)

    @exception_handler
    def decision(self, *args, **kwargs):
        self._append_level_event(EventTypes.DECISION.value, args, kwargs)

    @exception_handler
    def warn(self, *args, **kwargs):
        self._append_level_event(EventTypes.WARNING.value, args, kwargs)

    @exception_handler
    def error(self, *args, **kwargs):
        self._append_level_event(EventTypes.ERROR.value, args, kwargs)

    @exception_handler
    def insert_array(
        self,
        key: Optional[str],
        array: np.ndarray,
        caller_info: Optional[CallerInfo] = None,
        log_level: EventTypes = EventTypes.INFO,
        save_kwargs: Optional[Dict] = None,
        *args,
        **kwargs,
    ):
        if (
            DEACTIVATE
            or self._order_id is None
            or key is None
            or log_level.value > self.log_level
        ):
            return
        start = time()
        if caller_info is None:
            caller_info = CallerInfo(cls_name=self.name)
        path = self._get_file_path(self._order_id, key, caller_info)
        res, path = save_array(
            path, array, **(save_kwargs if save_kwargs is not None else dict())
        )
        if not res:
            return res
        res2 = self.insert(
            key,
            path,
            caller_info=caller_info,
            *args,
            **kwargs,
        )
        self._logger.debug(f"insert_array time {time()-start}")
        return res2

    @exception_handler
    def append_array(
        self,
        key: Optional[str],
        array: np.ndarray,
        caller_info: Optional[CallerInfo] = None,
        log_level: EventTypes = EventTypes.INFO,
        aux: Optional[Dict] = None,
        condition: Optional[Dict] = None,
        add_metadata: bool = False,
        save_kwargs: Optional[Dict] = None,
    ):
        if (
            DEACTIVATE
            or self._order_id is None
            or key is None
            or log_level.value > self.log_level
        ):
            return
        start = time()
        if caller_info is None:
            caller_info = CallerInfo(cls_name=self.name)
        suffix = os.sep + datetime.now().strftime("%H%M%S%f")[:-2]
        path = self._get_file_path(self._order_id, key, caller_info, suffix)
        res, path = save_array(
            path, array, **(save_kwargs if save_kwargs is not None else dict())
        )
        if not res:
            return res
        path += os.path.splitext(path)[1]
        if add_metadata or aux:
            res2 = self.append(
                key,
                os.path.dirname(path),
                aux=aux,
                condition=condition,
                caller_info=caller_info,
                add_metadata=add_metadata,
            )
        else:
            res2 = self.insert(
                key,
                os.path.dirname(path),
                aux=aux,
                condition=condition,
                caller_info=caller_info,
                add_metadata=add_metadata,
            )
        self._logger.debug(f"append_array time {time()-start}")
        return res2

    @exception_handler
    def backup(self):
        if DEACTIVATE or self._order_id is None:
            return
        data = {
            GKeys.collection: self._collection,
            GKeys.order_id: self._order_id,
            GKeys.data_dir: self.data_dir,
        }
        self._publish(LMNames.backup, data)

    def _append_level_event(self, level, args, kwargs):
        if args:
            args = list(args)
            args.insert(1, level)
            kwargs.pop("event_type", None)
        else:
            kwargs["event_type"] = level
        self.append_event(*args, **kwargs)

    def _get_data_dir(self, cfg_yaml):
        if cfg_yaml is None:
            return None
        with open(cfg_yaml) as f:
            try:
                return yaml.safe_load(f)["data_dir_path"]
            except yaml.YAMLError as e:
                self._logger.error(str(e))

    def _set_data_dir(
        self, data_dir_path: Optional[str] = None, cfg_yaml: Optional[str] = None
    ):
        if data_dir_path is None:
            data_dir_path = self._get_data_dir(cfg_yaml)
        self.data_dir = os.path.expanduser(data_dir_path)

    def _get_file_path(self, order_id, key: str, caller_info: CallerInfo, suffix=""):
        oid_dir = get_oid_dir(self.data_dir, self._collection, order_id)
        file_path = (
            os.path.join(
                oid_dir,
                caller_info.cls_name or "",
                caller_info.method_name or "",
                key,
            )
            + suffix
        )
        return file_path

    def _set_internal_logger(self, logger=None, internal_log_level: str = "INFO"):
        if DEACTIVATE:
            return
        if logger is None:
            logger = logging.getLogger(self.name)
            logger.setLevel(getattr(logging, internal_log_level))
        self._logger = logger

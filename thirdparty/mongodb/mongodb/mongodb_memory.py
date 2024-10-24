#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2023 Ozgur Yilmaz <ozgur.yilmaz@roms.inc>
#

"""!Logs important estimates such as size, pad config, vacuum areas for each sku into mongodb
@brief Log execution flow
"""
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import logging

from . import MongoDBIFBase
from .constants import MEMORY_CONFIG_PATH, SKU_COLLECTION, PR_SEP
from .constants import MemoryKeys as DBKeys
from .utils.decorators import exception_handler
from .utils import np2normal, get_git_info


class MongoDBMemory:
    """! MongoDB Memory."""

    def __init__(
        self,
        logger=None,
        db_name: Optional[str] = None,
        collection: Optional[str] = None,
    ):
        self._set_logger(logger)
        self._db_base = MongoDBIFBase(db_name=db_name)
        self.configure()
        self._collection = collection or SKU_COLLECTION
        self._db_base.create_index(self._collection, index=DBKeys.jan)
        self._git_info = get_git_info(
            str(Path(__file__).resolve().parent.parent.parent)
        )

    def _set_logger(self, logger=None, log_level: str = "INFO"):
        if logger is None:
            logger = logging.getLogger(self.__class__.__name__)
            logger.setLevel(getattr(logging, log_level))
        self._logger = logger

    def configure(self, cfg_yaml: str = MEMORY_CONFIG_PATH):
        ret = self._db_base._configure(cfg_yaml)
        return ret

    @exception_handler(False)
    def insert_entry(
        self,
        feature_name: str,
        jan: str,
        value: Any,
        order_id: str,
        successful: bool = True,
        has_np: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self._db_base.exists(self._collection, {DBKeys.jan: jan}):
            res = self._db_base.insert(
                DBKeys.jan, jan, self._collection, condition=None
            )
            if not res:
                return res
        condition = {DBKeys.jan: jan}
        _, document = self._db_base.find_one(self._collection, condition=condition)
        try:
            document[feature_name][DBKeys.entries][order_id]
            exists = True
        except:
            exists = False
        if exists:
            return self._append_entry(
                feature_name=feature_name,
                jan=jan,
                value=value,
                order_id=order_id,
                successful=successful,
                has_np=has_np,
                metadata=metadata,
            )
        return self._insert_entry(
            feature_name=feature_name,
            jan=jan,
            value=value,
            order_id=order_id,
            successful=successful,
            has_np=has_np,
            metadata=metadata,
        )

    @exception_handler(False)
    def _insert_entry(
        self,
        feature_name: str,
        jan: str,
        value: Any,
        order_id: str,
        successful: bool = True,
        has_np: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        payload = self._create_entry(value, successful, has_np, metadata)
        condition = {DBKeys.jan: jan}
        key = f"{feature_name}{PR_SEP}{DBKeys.entries}{PR_SEP}{order_id}"
        res = self._db_base.insert(
            key, [payload], collection_name=self._collection, condition=condition
        )
        return res

    @exception_handler(False)
    def _append_entry(
        self,
        feature_name: str,
        jan: str,
        value: Any,
        order_id: str,
        successful: bool = True,
        has_np: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        payload = self._create_entry(value, successful, has_np, metadata)
        condition = {DBKeys.jan: jan}
        key = f"{feature_name}{PR_SEP}{DBKeys.entries}{PR_SEP}{order_id}"
        res = self._db_base.append(
            key, payload, collection_name=self._collection, condition=condition
        )
        return res

    @exception_handler(False)
    def update_entry(
        self,
        feature_name: str,
        jan: str,
        update_dict: Dict[str, Any],
        order_id: str,
    ) -> bool:
        new_update_dict = dict()
        for k, v in update_dict.items():
            k = f"{feature_name}{PR_SEP}{DBKeys.entries}{PR_SEP}{order_id}{PR_SEP}$[]{PR_SEP}{k}"
            new_update_dict[k] = v
        update_dict = {"$set": new_update_dict}
        res = self._db_base.update(
            update_dict=update_dict,
            collection_name=self._collection,
            condition={"jan": jan},
        )
        return res

    @exception_handler(False)
    def update_elem(
        self,
        feature_name: str,
        jan: str,
        update_dict: Dict[str, Any],
        order_id: str,
        index: int,
    ) -> bool:
        if index < 0:
            condition = {DBKeys.jan: jan}
            res, document = self._db_base.find_one(
                collection_name=self._collection, condition=condition
            )
            if not res:
                return res
            try:
                elems = document[feature_name][DBKeys.entries][order_id]
            except KeyError:
                return False
            index = len(elems) + index
        new_update_dict = dict()
        for k, v in update_dict.items():
            k = f"{feature_name}{PR_SEP}{DBKeys.entries}{PR_SEP}{order_id}{PR_SEP}{index}{PR_SEP}{k}"
            new_update_dict[k] = v
        update_dict = {"$set": new_update_dict}
        res = self._db_base.update(
            update_dict=update_dict,
            collection_name=self._collection,
            condition={"jan": jan},
        )
        return res

    @exception_handler(False)
    def update_feature(
        self,
        feature_name: str,
        jan: str,
        update_dict: Dict[str, Any],
    ) -> bool:
        new_update_dict = dict()
        for k, v in update_dict.items():
            k = f"{feature_name}{PR_SEP}{k}"
            new_update_dict[k] = v
        update_dict = {"$set": new_update_dict}
        res = self._db_base.update(
            update_dict=update_dict,
            collection_name=self._collection,
            condition={"jan": jan},
        )
        return res

    @exception_handler(False)
    def pop_entries(self, feature_name: str, jan: str, num: int = 1) -> bool:
        condition = {DBKeys.jan: jan}
        feature = self.get_feature(feature_name, jan=jan) or dict()
        entries = feature.get(DBKeys.entries, dict())
        entries_list = [(k, v) for k, v in entries.items()]
        entries_list.sort(key=lambda x: x[1][0].get(DBKeys.timestamp))
        if not entries_list:
            return False
        final_rec = True
        for i in range(min(num, len(entries_list))):
            pop_oid = entries_list[i][0]
            field = f"{feature_name}{PR_SEP}{DBKeys.entries}{PR_SEP}{pop_oid}"
            rec = self._db_base.unset_one(
                field, collection_name=self._collection, condition=condition
            )
            if not rec:
                final_rec = rec
                break
        return final_rec

    @exception_handler(False)
    def remove_entry(self, feature_name: str, jan: str, order_id: str) -> bool:
        condition = {DBKeys.jan: jan}
        field = f"{feature_name}{PR_SEP}{DBKeys.entries}{PR_SEP}{order_id}"
        return self._db_base.unset_one(
            field, collection_name=self._collection, condition=condition
        )

    @exception_handler(False)
    def remove_elem(
        self, feature_name: str, jan: str, order_id: str, index: int
    ) -> bool:
        condition = {DBKeys.jan: jan}
        field = (
            f"{feature_name}{PR_SEP}{DBKeys.entries}{PR_SEP}{order_id}{PR_SEP}{index}"
        )
        return self._db_base.unset_one(
            field, collection_name=self._collection, condition=condition
        )

    @exception_handler
    def _create_entry(
        self,
        value: Any,
        successful: bool = True,
        has_np: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if has_np:
            value = np2normal(value)
        entry = {
            DBKeys.value: value,
            DBKeys.timestamp: self._get_timestamp(),
            DBKeys.git_sha: self._git_info["sha"],
            DBKeys.git_branch: self._git_info["branch"],
            DBKeys.successful: successful,
        }
        if metadata:
            entry.update(**metadata)
        return entry

    @exception_handler(False)
    def set_stats(self, stats: Dict, feature_name: str, jan: str) -> bool:
        condition = {DBKeys.jan: jan}
        key = f"{feature_name}{PR_SEP}{DBKeys.stats}"
        res = self._db_base.insert(
            key, stats, collection_name=self._collection, condition=condition
        )
        return res

    @exception_handler(False)
    def set_stable(self, value, feature_name: str, jan: str) -> bool:
        condition = {DBKeys.jan: jan}
        key = f"{feature_name}{PR_SEP}{DBKeys.stable}"
        res = self._db_base.insert(
            key, value, collection_name=self._collection, condition=condition
        )
        return res

    @exception_handler(False)
    def set_add_entry(self, add_entry: bool, feature_name: str, jan: str) -> bool:
        condition = {DBKeys.jan: jan}
        key = f"{feature_name}{PR_SEP}{DBKeys.add_entry}"
        res = self._db_base.insert(
            key, add_entry, collection_name=self._collection, condition=condition
        )
        return res

    @exception_handler
    def get_feature(self, feature_name: str, jan: str) -> Optional[Dict[str, Any]]:
        projection = self._get_projection(feature_name)
        condition = {DBKeys.jan: jan}
        rc, document = self._db_base.find_one(
            collection_name=self._collection,
            condition=condition,
            projection=projection,
        )
        if rc:
            data = document.get(feature_name)
            return data

    # @exception_handler
    # def insert_feature(
    #     self,
    #     feature_data: Dict,
    #     feature_name: str,
    #     jan: str,
    # ):
    #     condition = {DBKeys.jan: jan}
    #     res = self._db_base.insert(
    #         feature_name,
    #         feature_data,
    #         collection_name=self._collection,
    #         condition=condition,
    #     )
    #     return res

    @exception_handler
    def get_features(self, features: List[str], jan: str) -> Optional[Dict[str, Any]]:
        projection = self._get_projection(features)
        condition = {DBKeys.jan: jan}
        _, document = self._db_base.find_one(
            collection_name=self._collection,
            condition=condition,
            projection=projection,
        )
        return document

    def clean_feature(
        self,
        feature_name: str,
        jan: str,
    ) -> bool:
        condition = {DBKeys.jan: jan}
        res = self._db_base.unset_one(
            feature_name, collection_name=self._collection, condition=condition
        )
        return res

    def clean_jan(self, jan: str):
        condition = {DBKeys.jan: jan}
        res = self._db_base.delete_one(
            collection_name=self._collection, condition=condition
        )
        return res

    @staticmethod
    def _get_timestamp():
        return datetime.utcnow()

    @staticmethod
    def _get_projection(
        projection: Optional[Union[str, List[str], Dict[str, bool]]] = None
    ) -> Dict[str, bool]:
        if projection:
            if isinstance(projection, str):
                projection = [projection]
            if not isinstance(projection, dict):
                projection = {k: True for k in projection}
            projection["_id"] = False
        return projection

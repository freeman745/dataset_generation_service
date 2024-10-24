from typing import Dict, List, Optional, Sequence
import os
import shutil
import traceback
import redis
import bson
from collections import Counter

from . import MongoDBIFBase
from .mongodb_reader import MongoDBReader
from .constants import (
    ITEM_COLLECTION,
    ITEM_CONFIG_PATH,
    MONGODB_LOGGER_QUEUE,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    ACTION,
    DATA,
    GeneralKeys as GKeys,
    LogMethodNames as LMNames,
)
from .constants import GeneralKeys as GKeys


class MongoDBHandler:
    def __init__(self, collection: Optional[str] = None):
        self._db_base = MongoDBIFBase()
        self._collection = collection or ITEM_COLLECTION
        self.configure()
        self._set_publiser()
        self.is_configured = self._db_base._is_configured

    def configure(self, cfg_yaml: str = ITEM_CONFIG_PATH):
        ret = self._db_base._configure(cfg_yaml)
        self.is_configured = self._db_base._is_configured
        return ret

    def remove_entry(
        self, collection: Optional[str] = None, condition: Optional[Dict] = None
    ):
        if collection is None:
            collection = self._collection
        return self._db_base.delete_one(collection, condition)

    def remove_entries(
        self, collection: str = ITEM_COLLECTION, condition: Optional[Dict] = None
    ):
        return self._db_base.delete_many(collection, condition)

    def remove_oid(
        self,
        order_id: str,
        collection: Optional[str] = None,
        condition: Optional[Dict] = None,
        remove_dir: bool = True,
    ):
        if collection is None:
            collection = self._collection
        if condition is None:
            condition = dict()
        condition[GKeys.order_id] = order_id
        if remove_dir:
            item = MongoDBReader(collection).get_item(order_id)
            if item:
                data_oid_dir = item.data_oid_dir
                if data_oid_dir and os.path.isdir(data_oid_dir):
                    shutil.rmtree(data_oid_dir, ignore_errors=True)
        data = {GKeys.collection: collection, GKeys.order_id: order_id}
        self._publish(LMNames.remove, data)

    def remove_oids(
        self,
        order_ids: Sequence[str],
        collection: Optional[str] = None,
        condition: Optional[Dict] = None,
    ):
        if collection is None:
            collection = self._collection
        if condition is None:
            condition = dict()
        condition[GKeys.order_id] = {"$in": order_ids}
        return self.remove_entries(collection, condition)

    def remove_db_id(
        self,
        db_id: bson.ObjectId,
        collection: Optional[str] = None,
    ):
        if collection is None:
            collection = self._collection
        return self.remove_entries(collection, {GKeys.db_id: db_id})

    def remove_db_ids(
        self,
        db_ids: List[bson.ObjectId],
        collection: Optional[str] = None,
    ):
        if collection is None:
            collection = self._collection
        return self.remove_entries(collection, {GKeys.db_id: {"$in": db_ids}})

    def export_entry(
        self,
        collection: Optional[str] = None,
        condition: Optional[Dict] = None,
        filepath: Optional[str] = None,
    ):
        if collection is None:
            collection = self._collection
        return self._db_base.export_one(collection, condition, filepath)

    def export_entries(
        self,
        collection: Optional[str] = None,
        condition: Optional[Dict] = None,
    ):
        if collection is None:
            collection = self._collection
        return self._db_base.export_many(collection, condition)

    def export_oid(
        self,
        order_id: str,
        collection: Optional[str] = None,
        condition: Optional[Dict] = None,
    ):
        if collection is None:
            collection = self._collection
        if condition is None:
            condition = dict()
        condition[GKeys.order_id] = order_id
        return self.export_entry(collection, condition)

    def export_oids(
        self,
        order_ids: Sequence[str],
        collection: Optional[str] = None,
        condition: Optional[Dict] = None,
    ):
        if collection is None:
            collection = self._collection
        if condition is None:
            condition = dict()
        condition[GKeys.order_id] = {"$in": order_ids}
        return self.export_entries(collection, condition)

    def export_db_id(
        self,
        db_id: bson.ObjectId,
        collection: Optional[str] = None,
    ):
        if collection is None:
            collection = self._collection
        return self.export_entry(collection, {GKeys.db_id: db_id})

    def export_db_ids(
        self,
        db_ids: List[bson.ObjectId],
        collection: Optional[str] = None,
    ):
        if collection is None:
            collection = self._collection
        return self.export_entries(collection, {GKeys.db_id: {"$in": db_ids}})

    def import_entry(self, collection_name: str, filepath: str):
        self._db_base.import_one(collection_name, filepath)

    def import_entries(self, collection_name: str, filepaths: List[str]):
        self._db_base.import_many(collection_name, filepaths)

    def clean_db(
        self,
        collection: Optional[str] = None,
        del_oids=(None, "", "0"),
        remove_duplicate_oids: bool = True,
        condition: Optional[Dict] = None,
    ):
        if collection is None:
            collection = self._collection
        if condition is None:
            condition = dict()
        if remove_duplicate_oids:
            _, entries = self._db_base.find(
                collection,
                condition=condition,
                projection={GKeys.db_id: False, GKeys.order_id: True},
            )
            oids = [entry.get(GKeys.order_id) for entry in entries]
            counter = Counter(oids)
            del_oids = list(del_oids) + [k for k, v in counter.items() if v > 1]
            del_oids = list(set(del_oids))
        self.remove_oids(del_oids, collection=collection)

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

from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import yaml
from bson import ObjectId

from .utils.decorators import exception_handler

from . import MongoDBIFBase
from .mongodb_item import MongoDBItem
from .constants import (
    ITEM_COLLECTION,
    ITEM_CONFIG_PATH,
    EventTypes,
    GeneralKeys as GKeys,
    ClassNames as CNs,
    ActionManagerNodeKeys as AMKeys,
    PickDetectorNodeKeys as PcNKeys,
)


class MongoDBReader:
    """! MongoDB Reader."""

    def __init__(
        self,
        collection: Optional[str] = None,
        cfg_yaml: str = ITEM_CONFIG_PATH,
    ):
        self._db_base = MongoDBIFBase()
        self.configure(cfg_yaml)
        self.is_configured = self._db_base._is_configured
        self._collection = collection or ITEM_COLLECTION
        self._create_indexes(cfg_yaml)

    def configure(self, cfg_yaml: str = ITEM_CONFIG_PATH):
        ret = self._db_base._configure(cfg_yaml)
        self.is_configured = self._db_base._is_configured
        return ret

    @exception_handler
    def get_item_dict(
        self,
        collection: Optional[str] = None,
        elem_names: Union[None, str, Tuple[str], List[str], List[Tuple[str]]] = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        if collection is None:
            collection = self._collection
        condition = self._get_condition(**kwargs)
        projection = self._get_projection(elem_names)
        rec, item_dict = self._db_base.find_one(
            collection, condition, projection=projection
        )
        if rec:
            return item_dict

    @exception_handler
    def get_item_dicts(
        self,
        collection: Optional[str] = None,
        elem_names: Union[None, str, Tuple[str], List[str], List[Tuple[str]]] = None,
        newest_first: bool = True,
        max_num: Optional[int] = None,
        **kwargs,
    ) -> Optional[List[Dict[str, Any]]]:
        if collection is None:
            collection = self._collection
        sort = (
            (f"{GKeys.metadata}.{GKeys.timestamp}", -1)
            if newest_first
            else (f"{GKeys.metadata}.{GKeys.timestamp}", 1)
        )
        condition = self._get_condition(**kwargs)
        projection = self._get_projection(elem_names)
        rec, cursor = self._db_base.find(
            collection,
            condition,
            sort=sort,
            limit=max_num,
            projection=projection,
            allow_disk_use=False,
        )
        if not rec:
            return None
        entries = list(cursor)
        cursor.close()
        return entries

    def get_item(
        self,
        order_id: Optional[str] = None,
        **kwargs,
    ) -> Optional[MongoDBItem]:
        item_dict = self.get_item_dict(order_id=order_id, **kwargs)
        if item_dict:
            return MongoDBItem(item_dict)
        return None

    def get_items(self, **kwargs) -> Optional[List[MongoDBItem]]:
        item_dicts = self.get_item_dicts(**kwargs)
        if item_dicts is not None:
            return [MongoDBItem(item_dict) for item_dict in item_dicts]
        return None

    @exception_handler(list())
    def get_oids(
        self,
        **kwargs,
    ) -> List[str]:
        item_dicts = self.get_item_dicts(elem_names=GKeys.order_id, **kwargs)
        if item_dicts is None:
            return
        return [d[GKeys.order_id] for d in item_dicts]

    @exception_handler
    def get_index_information(self, collection: Optional[str] = None):
        if collection is None:
            collection = self._collection
        return self._db_base.get_index_information(collection)

    def _get_condition(
        self,
        order_id: Optional[str] = None,
        order_ids: Optional[List[str]] = None,
        db_id: Optional[ObjectId] = None,
        db_ids: Optional[List[ObjectId]] = None,
        date_start: Optional[Union[datetime, Tuple[int, ...]]] = None,
        date_stop: Optional[Union[datetime, Tuple[int, ...]]] = None,
        event_cls_names: Optional[List[str]] = None,
        event_types: Optional[List[str]] = None,
        event_briefs: Optional[List[str]] = None,
        jan_codes: Optional[List[str]] = None,
        has_multiple_picks: bool = False,
        has_not_multiple_picks: bool = False,
        is_successful: bool = False,
        is_not_successful: bool = False,
        condition: Optional[Dict] = None,
    ):
        if condition is None:
            condition = dict()

        # --- check order_id(s)
        if order_id is not None:
            condition[GKeys.order_id] = order_id
        elif order_ids is not None:
            condition[GKeys.order_id] = {"$in": order_ids}
        else:
            condition[GKeys.order_id] = {"$exists": True, "$nin": [None, ""]}

        # --- check db_id(s)
        if db_id is not None:
            condition[GKeys.db_id] = db_id
        elif db_ids is not None:
            condition[GKeys.db_id] = {"$in": db_ids}

        # --- dates
        if (
            date_start is not None or date_stop is not None
        ) and f"{GKeys.metadata}.{GKeys.timestamp}" not in condition:
            condition[f"{GKeys.metadata}.{GKeys.timestamp}"] = dict()

        if date_start is not None:
            if not isinstance(date_start, datetime):
                date_start = datetime(*date_start)
            condition[f"{GKeys.metadata}.{GKeys.timestamp}"]["$gte"] = date_start
        if date_stop is not None:
            if not isinstance(date_stop, datetime):
                date_stop = datetime(*date_stop)
            condition[f"{GKeys.metadata}.{GKeys.timestamp}"]["$lt"] = date_stop

        # --- events
        if (
            any([x for x in (event_cls_names, event_types, event_briefs)])
            and GKeys.events not in condition
        ):
            condition[GKeys.events] = {"$elemMatch": dict()}
        if event_cls_names:
            condition[GKeys.events]["$elemMatch"][GKeys.cls_name] = {
                "$in": event_cls_names
            }
        if event_types:
            condition[GKeys.events]["$elemMatch"][GKeys.event_type] = {
                "$in": [EventTypes[et].value for et in event_types]
            }
        if event_briefs:
            condition[GKeys.events]["$elemMatch"][GKeys.brief] = {"$in": event_briefs}

        # --- jan codes
        if jan_codes:
            condition[PcNKeys.jan] = {"$in": jan_codes}

        # --- multiple picks
        if has_multiple_picks:
            condition[f"{CNs.ActionDetectPick}.action_start.1"] = {"$exists": True}

        # --- no multiple picks
        if has_not_multiple_picks:
            condition[f"{CNs.ActionDetectPick}.action_start.1"] = {"$exists": False}

        # --- is_successful picks
        if is_successful:
            condition[f"{CNs.OrderManagerNode}"] = {"$exists": False}
            # condition[f"{CNs.ActionManagerNode}.msg"] = {
            #     "$elemMatch": {"action": "toplace"}
            # }
        # --- is_not_successful picks
        elif is_not_successful:
            condition[f"{CNs.OrderManagerNode}"] = {"$exists": True}
            # condition[f"{CNs.ActionManagerNode}.msg"] = {
            #     "$not": {"$elemMatch": {"action": "toplace"}}
            # }

        return condition

    @staticmethod
    def _get_projection(
        elem_names: Union[None, str, Tuple[str], List[str], List[Tuple[str]]] = None,
        projection: Optional[Dict] = None,
        include_id: bool = True,
    ) -> Dict[str, Any]:
        if projection is None:
            projection = dict()
        if not include_id:
            projection["_id"] = False
        if elem_names is None:
            elem_names = []
        if isinstance(elem_names, str):
            elem_names = [elem_names]
        for elem_name in elem_names:
            if isinstance(elem_name, tuple):
                elem_name = ".".join(elem_name)
            projection[elem_name] = True
        return projection

    @exception_handler(False)
    def _create_indexes(self, cfg_yaml, collection: Optional[str] = None):
        with open(cfg_yaml) as f:
            cfg_obj: dict = yaml.safe_load(f)
        indexes = cfg_obj.get("indexes", list())
        if collection is None:
            collection = self._collection
        rc = True
        for index_dict in indexes:
            index = index_dict["index"]
            unique = index_dict.get("unique", False)
            name = index_dict.get("name")
            rc &= self._db_base.create_index(
                collection, index, name=name, unique=unique
            )
        return rc

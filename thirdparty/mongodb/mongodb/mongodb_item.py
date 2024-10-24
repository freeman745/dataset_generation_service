from collections import defaultdict
import os
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from glob import glob
import datetime
from bson.json_util import loads

from .constants import (
    ClassGroups as CGs,
    GeneralKeys as GKeys,
    TargetDetectorKeys as TKeys,
    PickDetectorDepthKeys as PcKeys,
    SkuSizeCheckerKeys as SKeys,
    PlaceDetectorDepthKeys as PlKeys,
    PcdProjectorKeys as PdKeys,
    PickDetectorNodeKeys as PcNKeys,
    OrderManagerNodeKeys as OmKeys,
    ActionManagerNodeKeys as AMKeys,
    ErrorCategoriesNames as ECNames,
    ActionKeys as AcKeys,
    ClassNames as CNs,
    ActionNames,
    StepNames,
    action_mapping,
)
from .utils import (
    BBox,
    draw_bbox,
    draw_bboxes,
    draw_time_plot,
    crop_bbox,
    ti_normalize_8b,
    load_array,
    get_nested_item,
    has_index,
    get_dsize,
    ActionLog,
    get_agregated_duration,
    get_pcd_of_obj_from_pmap,
    memoize,
)
from .utils.decorators import (
    METHODS_LOGS_KEY,
    exception_handler,
)
from .utils.log_classes import MethodLog

Dsize = Optional[Union[float, int, Tuple[int, int]]]


class MongoDBItem:
    def __init__(
        self, item_dict: Union[Dict, str], oid_format: Optional[str] = None
    ) -> None:
        self._set_dict(item_dict)
        self.oid_format = oid_format

    def _set_dict(self, item_dict: Union[Dict, str]) -> None:
        if isinstance(item_dict, dict):
            self._dict = item_dict
        elif isinstance(item_dict, str):
            if not os.path.isfile(item_dict):
                raise ValueError(
                    f"if item_dict is str, it must be a path to a json file"
                )
            with open(item_dict) as f:
                s = f.read()
            self._dict = loads(s)
        else:
            raise TypeError(
                "item_dict must be either a dict or a str (path to json). "
                f"{type(item_dict)} was passed"
            )

    @property
    def order_id(self) -> str:
        return self._dict.get(GKeys.order_id, "")

    @property
    def db_id(self) -> str:
        return self._dict.get(GKeys.db_id, "")

    #############################################################################
    ## metadata ##

    @property
    def metadata(self) -> Dict:
        return self._dict.get(GKeys.metadata, dict())

    @property
    def timestamp(self) -> Optional[datetime.datetime]:
        return self.metadata.get(GKeys.timestamp)

    @property
    def timestamp_str(self) -> Optional[str]:
        if self.timestamp:
            return self.timestamp.strftime("%Y-%m-%d %H:%M:%S")

    @property
    def git_sha(self) -> Optional[str]:
        return self.metadata.get(GKeys.git_sha)

    @property
    def git_branch(self) -> Optional[str]:
        return self.metadata.get(GKeys.git_branch)

    @property
    def git_url(self) -> Optional[str]:
        if self.git_sha:
            return f"https://github.com/roms-jp/RomsPickingSystem/tree/{self.git_sha}"

    @property
    def or_data_oid_dir(self) -> Optional[str]:
        return self.metadata.get(GKeys.oid_dir)

    @property
    def safie_urls(self) -> List[str]:
        return self.metadata.get(GKeys.safie_urls, [])

    @property
    def data_oid_dir(self) -> Optional[str]:
        if self.oid_format is None:
            return self.or_data_oid_dir
        if "{}" not in self.oid_format:
            self.oid_format += "{}"
        return self.oid_format.format(self.or_data_oid_dir)

    @property
    def jan(self) -> Optional[str]:
        return self._get_nested((PcNKeys.jan,))

    @property
    def registered_size(self) -> Optional[Tuple[int, int, int]]:
        return self._get_nested(
            (CNs.PickDetectorDepth, PcKeys.detect_kwargs, 0, PcKeys.registered_size)
        )

    @property
    def registered_weight(self) -> Optional[float]:
        return self._get_nested(
            (CNs.PickDetectorDepth, PcKeys.detect_kwargs, 0, PcKeys.registered_weight)
        )

    @property
    def result_dict(self) -> Dict:
        return self._get_nested((CNs.OrderManagerNode), dict())

    @property
    def error_desc(self) -> Optional[str]:
        return self.result_dict.get(OmKeys.desc)

    @property
    def result_code(self) -> Optional[str]:
        return self.result_dict.get(OmKeys.code)

    @property
    def error_category(self) -> Optional[str]:
        sort_code = self.result_dict.get(OmKeys.sort_code)
        desc = self.result_dict.get(OmKeys.desc)
        # --- previous_failure
        if (
            desc in ["order_manager.FinishedErrorOccuredBeforeItemProceeded"]
            or sort_code == "0xfa2"
        ):
            return ECNames.previous_failure
        
        if desc in ["FinishedOrderComplete"] or self.result_code == "0000":
            return None
        # --- no_more_targets
        if desc in ["FinishedNoMoreTargets"]:
            return ECNames.no_more_targets

        # --- pick_failure
        if desc in [
            "FinishedNoMoreTargetsNotEmpty",
            "FinishedTooManyPickFailures",
        ]:
            return ECNames.pick_failure

        # --- collision
        if desc in ["FinishedBadIOConditions"]:
            return ECNames.collision

        # --- drop
        if sort_code == "0x6":
            return ECNames.drop

        # --- destination_full
        if desc in ["FinishedNoMoreDest", "FinishedCannotBePlaced"]:
            return ECNames.destination_full


        # --- system_failure
        if False:
            return ECNames.system_failure

        return ECNames.unknown

    @property
    def is_successful(self) -> Optional[bool]:
        return self.error_category is None

    @property
    def result_code(self) -> str:
        return self._get_nested((CNs.OrderManagerNode, OmKeys.result_code), "0000")

    @property
    def multi_picks(self) -> Optional[bool]:
        picks = self.get_actions(actions_included="detectpick")
        if len(picks) == 0:
            return None
        return len(picks) > 1

    @property
    def pick_attempts(self) -> int:
        return len(self.get_actions(actions_included="detectpick"))

    @property
    def result_publisher(self) -> Optional[int]:
        return self.result_dict.get("result_publisher")

    @property
    def str_dict(self) -> str:
        from bson.json_util import dumps

        return dumps(self._dict, indent=4)

    #############################################################################
    ## classes ##

    @property
    def PickDetectorDepth(self) -> Dict:
        return self._dict.get(CNs.PickDetectorDepth, dict())

    @property
    def PlaceDetectorDepth(self) -> Dict:
        return self._dict.get(CNs.PlaceDetectorDepth, dict())

    @property
    def SkuSizeChecker(self) -> Dict:
        return self._dict.get(CNs.SkuSizeChecker, dict())

    def get_cls_dict(
        self, cls_name: str, fields_included: Optional[List[str]] = None
    ) -> Dict:
        cls_dict = self._dict.get(cls_name, dict())
        if fields_included is not None:
            cls_dict = {field: cls_dict[field] for field in fields_included}
        return cls_dict

    #############################################################################
    ## methods ##

    def get_method_names(self, cls_name) -> List[str]:
        if cls_name not in self._dict:
            return []
        return list(self._dict[cls_name].keys())

    def get_method_dict(
        self,
        cls_name: str,
        method_name: str,
        fields_included: Optional[List[str]] = None,
    ) -> Dict:
        method_dict = self._get_nested((cls_name, method_name), dict())
        if fields_included is not None:
            method_dict = {
                key: method_dict[key] for key in fields_included if key in method_dict
            }
        return method_dict

    #############################################################################
    ## cls logs ##

    @exception_handler(list())
    def get_methods_logs(
        self,
        cls_names: Optional[Union[List[str], str]] = None,
        method_names: Optional[Union[List[str], str]] = None,
        duration_limit: Optional[float] = None,
    ) -> List[MethodLog]:
        methods_logs = [
            MethodLog(**log_dict)
            for log_dict in self._dict.get(METHODS_LOGS_KEY, list())
        ]
        methods_logs.sort(key=lambda x: x.start_time)
        if cls_names is not None:
            if isinstance(cls_names, str):
                cls_names = [cls_names]
            methods_logs = [log for log in methods_logs if log.cls_name in cls_names]
        if method_names is not None:
            if isinstance(method_names, str):
                method_names = [method_names]
            methods_logs = [
                log for log in methods_logs if log.method_name in method_names
            ]
        if duration_limit is not None:
            methods_logs = [
                log for log in methods_logs if log.duration >= (duration_limit / 1000)
            ]
        return methods_logs

    @exception_handler
    def get_pickd_logs(
        self, num: Optional[int] = -1, duration_limit: Optional[float] = None
    ) -> Union[List[List[MethodLog]], List[MethodLog]]:
        all_pickd_logs = self.get_methods_logs(CNs.PickDetectorDepth, duration_limit)
        pickd_logs_list, logs = [], []
        for log in all_pickd_logs:
            if log.method_name == "detect":
                if logs:
                    pickd_logs_list.append(logs)
                logs = [log]
            else:
                logs.append(log)
        if logs:
            pickd_logs_list.append(logs)
        if num is None:
            return pickd_logs_list
        return pickd_logs_list[num]

    @exception_handler
    def get_sizecheck_logs(
        self, duration_limit: Optional[float] = None
    ) -> List[MethodLog]:
        return self.get_methods_logs(CNs.SkuSizeChecker, duration_limit)

    @exception_handler
    def get_placed_logs_list(
        self, duration_limit: Optional[float] = None
    ) -> List[List[MethodLog]]:
        all_logs = self.get_methods_logs(CNs.PlaceDetectorDepth, duration_limit)
        logs_list = []
        logs = []
        for log in all_logs:
            if log.method_name == "detect":
                if len(logs):
                    logs_list.append(logs)
                logs = [log]
            else:
                logs.append(log)
        if len(logs):
            logs_list.append(logs)
        return logs_list

    @exception_handler
    def get_checkplace_logs(
        self, duration_limit: Optional[float] = None
    ) -> Optional[List[MethodLog]]:
        logs_list = self.get_placed_logs_list(duration_limit)
        if len(logs_list) >= 1:
            return logs_list[0]

    @exception_handler
    def get_placed_logs(
        self, duration_limit: Optional[float] = None
    ) -> Optional[List[MethodLog]]:
        logs_list = self.get_placed_logs_list(duration_limit)
        if len(logs_list) == 2:
            return logs_list[1]

    #############################################################################
    ## cls plots ##

    @exception_handler
    def get_cls_plot(
        self,
        cls_name: str,
        logs: List[MethodLog],
        total_duration: Optional[float] = None,
        return_plot: bool = False,
    ) -> Optional[np.ndarray]:
        names = [log.method_name for log in logs]
        durations = [log.duration for log in logs]
        start_timestamps = [log.start_time for log in logs]
        end_timestamps = [log.end_time for log in logs]
        return draw_time_plot(
            cls_name,
            names,
            durations,
            start_timestamps,
            end_timestamps,
            total_duration,
            return_plot=return_plot,
        )

    @exception_handler
    def get_pickd_plot(
        self,
        num: Optional[int] = -1,
        duration_limit: Optional[float] = None,
        return_plot: bool = False,
    ) -> Optional[np.ndarray]:
        logs_list = self.get_pickd_logs(None, duration_limit)
        total_durations = self.get_pickd_duration(None)
        if num is None:
            plots = [
                self.get_cls_plot(
                    CNs.PickDetectorDepth,
                    logs,
                    round(total_duration, 3),
                    return_plot=return_plot,
                )
                for logs, total_duration in zip(logs_list, total_durations)
            ]
            return plots
        return self.get_cls_plot(
            CNs.PickDetectorDepth,
            logs_list[num],
            round(total_durations[num], 3),
            return_plot=return_plot,
        )

    @exception_handler
    def get_sizecheck_plot(
        self, duration_limit: Optional[float] = None, return_plot: bool = False
    ) -> Optional[np.ndarray]:
        logs = self.get_sizecheck_logs(duration_limit)
        total_duration = self.get_sizecheck_duration()
        return self.get_cls_plot(
            CNs.SkuSizeChecker, logs, round(total_duration, 3), return_plot=return_plot
        )

    @exception_handler
    def get_checkplace_plot(
        self, duration_limit: Optional[float] = None, return_plot: bool = False
    ) -> Optional[np.ndarray]:
        logs = self.get_checkplace_logs(duration_limit)
        total_duration = self.get_checkplace_duration()
        return self.get_cls_plot(
            "CheckPlace", logs, round(total_duration, 3), return_plot=return_plot
        )

    @exception_handler
    def get_placed_plot(
        self, duration_limit: Optional[float] = None, return_plot: bool = False
    ) -> Optional[np.ndarray]:
        logs = self.get_placed_logs(duration_limit)
        total_duration = self.get_placed_duration()
        return self.get_cls_plot(
            CNs.PlaceDetectorDepth,
            logs,
            round(total_duration, 3) if total_duration else None,
            return_plot=return_plot,
        )

    #############################################################################
    ## events ##

    def get_events(
        self,
        cls_names: Optional[List[Tuple[str, str]]] = None,
        event_types: Optional[Union[int, List[int]]] = None,
        briefs: Optional[List[str]] = None,
        time_ordered: bool = False,
        time_reversed: bool = False,
        fields_included: Optional[List[str]] = None,
    ) -> List[Dict]:
        events = self._dict.get(GKeys.events, list())

        if isinstance(event_types, int):
            event_types = [event_types]
        if event_types is not None:
            events = [e for e in events if e[GKeys.event_type] in event_types]
        if cls_names is not None:
            events = [e for e in events if e[GKeys.cls_name] in cls_names]
        if briefs is not None:
            events = [e for e in events if e[GKeys.brief] in briefs]

        if time_ordered:
            events = sorted(
                events,
                key=lambda x: x[GKeys.timestamp],
                reverse=time_reversed,
            )

        if fields_included is not None:
            events = [{key: event[key] for key in fields_included} for event in events]
        return events

    @exception_handler
    def get_pickd_events(self, *args, **kwargs) -> List[Dict]:
        return self.get_events(CGs.Pickd, *args, **kwargs)

    @exception_handler
    def get_sizecheck_events(self, *args, **kwargs) -> List[Dict]:
        return self.get_events(CGs.SizeCheck, *args, **kwargs)

    @exception_handler
    def get_placed_events(self, *args, **kwargs) -> List[Dict]:
        return self.get_events(CGs.Placed, *args, **kwargs)

    #############################################################################
    ## durations ##

    @exception_handler
    def get_pickd_duration(self, num: Optional[int] = -1) -> Union[List[float], float]:
        logs_list = self.get_pickd_logs(None)
        durations = [
            [log.duration for log in logs if log.method_name == "detect"][0]
            for logs in logs_list
        ]
        if num is None:
            return durations
        return durations[num]

    @exception_handler
    def get_sizecheck_duration(self) -> Optional[float]:
        logs = self.get_sizecheck_logs()
        for log in logs:
            if log.method_name == "estimate":
                return log.duration

    @exception_handler
    def get_checkplace_duration(self) -> Optional[float]:
        logs = self.get_checkplace_logs()
        if not logs:
            return
        for log in logs:
            if log.method_name == "detect":
                return log.duration

    @exception_handler
    def get_placed_duration(self) -> Optional[float]:
        logs = self.get_placed_logs()
        if not logs:
            return
        for log in logs:
            if log.method_name == "detect":
                return log.duration

    #############################################################################
    ## Pick
    #############################################################################
    ## detect kwargs ##

    @property
    def pickd_detect_kwargs(self, num: int = -1) -> Optional[Dict]:
        detect_kwargs_list = self._get_nested(
            (CNs.PickDetectorDepth, PcKeys.detect_kwargs), list()
        )
        if detect_kwargs_list:
            return detect_kwargs_list[num]

    @property
    def pickd_input_paths(self) -> Dict[str, str]:
        input_paths = dict()

        paths = self.pickd_visible_paths
        for i, path in enumerate(paths, 1):
            if path is not None:
                ext = os.path.splitext(path)[-1]
                input_paths[f"pickd_visible_{i}{ext}"] = path
        paths = self.pickd_depth_paths
        for i, path in enumerate(paths, 1):
            if path is not None:
                ext = os.path.splitext(path)[-1]
                input_paths[f"pickd_depth_{i}{ext}"] = path
        return input_paths

    #############################################################################
    ## pickdepth visible ##

    @property
    def pickd_visible_paths(self) -> List[str]:
        return self._get_paths(CNs.PickDetectorDepth, PcKeys.visible)

    @exception_handler
    def get_pickd_visible(
        self,
        num: int = -1,
        dsize: Dsize = None,
    ) -> Optional[np.ndarray]:
        if has_index(self.pickd_visible_paths, num):
            return load_array(self.pickd_visible_paths[num], dsize=dsize)

    pickd_visible: np.ndarray = property(get_pickd_visible)

    @property
    def pickd_num(self) -> int:
        if self.action_manager_msgs:
            return len(
                [
                    msg
                    for msg in self.action_manager_msgs
                    if msg["action"] == "detectpick"
                ]
            )
        if self.get_methods_logs():
            return len(
                [
                    log
                    for log in self.get_methods_logs()
                    if log.cls_name == CNs.PickDetectorDepth
                    and log.method_name == "detect"
                ]
            )
        return None

    #############################################################################
    ## pickedepth depth ##

    @property
    def pickd_depth_paths(self) -> List[str]:
        return self._get_paths(CNs.PickDetectorDepth, PcKeys.depth)

    @exception_handler
    def get_pickd_depth(
        self,
        num: int = -1,
        dsize: Dsize = None,
        nan: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        if has_index(self.pickd_depth_paths, num):
            array = load_array(self.pickd_depth_paths[num], dsize=dsize)
        else:
            return
        if isinstance(nan, str):
            if nan.lower() == "min":
                array = np.nan_to_num(array, np.nanargmin(array))
            elif nan.lower() == "max":
                array = np.nan_to_num(array, np.nanargmax(array))
        elif isinstance(nan, (float, int)):
            array = np.nan_to_num(array, nan)
        return array

    pickd_depth = property(get_pickd_depth)

    ##########

    def get_pickd_trimmed_depth(
        self,
        num: int = -1,
        dsize: Dsize = None,
        nan: Optional[str] = None,
    ) -> np.ndarray:
        depth = self.get_pickd_depth(num=num, dsize=dsize, nan=nan)
        min_x, min_y, max_x, max_y = self.pickd_area_bb.coords
        trimmed_depth = depth[min_y:max_y, min_x:max_x]
        return trimmed_depth

    ##########

    @exception_handler
    def get_pickd_depth_normalized(
        self, num: int = -1, dsize: Dsize = None
    ) -> np.ndarray:
        array = self.get_pickd_depth(num, dsize=dsize)
        return ti_normalize_8b(array[..., -1], nan="min")

    pickd_depth_normalized: np.ndarray = property(get_pickd_depth_normalized)

    #############################################################################
    ## pickdepth visible trimmed ##

    @property
    def pickd_trimmed_visible_paths(self) -> List[str]:
        return self._get_paths(
            CNs.PickDetectorDepth,
            PcKeys.trimmed_visible,
        )

    @exception_handler
    def get_pickd_trimmed_visible(
        self, num: int = -1, dsize: Dsize = None
    ) -> Optional[np.ndarray]:
        array = self.get_pickd_visible(num=num)
        if array is not None:
            return crop_bbox(array, self.pickd_area_bb, dsize=dsize)

    pickd_trimmed_visible: np.ndarray = property(get_pickd_trimmed_visible)

    #############################################################################
    ## pickdepth depth trimmed ##

    @property
    def pickd_trimmed_depth_paths(self) -> List[str]:
        return self._get_paths(
            CNs.PickDetectorDepth,
            PcKeys.trimmed_depth,
        )

    # @exception_handler
    # def get_pickd_trimmed_depth(self, num: int = -1, dsize: Dsize = None) -> np.ndarray:
    #     return load_array(self.pickd_trimmed_depth_paths[num], dsize=dsize)

    pickd_trimmed_depth = property(get_pickd_trimmed_depth)

    #############################################################################
    ## pickdepth depth aligned ##

    @property
    def pickd_aligned_depth_paths(self) -> List[str]:
        return self._get_paths(
            CNs.PickDetectorDepth,
            "aligned_depth",
        )

    @exception_handler
    def get_pickd_aligned_depth(self, num: int = -1, dsize: Dsize = None) -> np.ndarray:
        return load_array(self.pickd_aligned_depth_paths[num], dsize=dsize)

    @exception_handler
    def get_pickd_aligned_trimmed_depth(
        self, num: int = -1, dsize: Dsize = None
    ) -> np.ndarray:
        depth = self.get_pickd_aligned_depth(num=num)
        min_x, min_y, max_x, max_y = self.pickd_area_bb.coords
        trimmed_depth = depth[min_y:max_y, min_x:max_x]
        return trimmed_depth

    pickd_aligned_depth = property(get_pickd_aligned_depth)

    #############################################################################
    ## pickd bboxes ##

    @property
    def pickd_area_bb(self) -> BBox:
        area = self._get_nested((CNs.PickDetectorDepth, PcKeys.area))
        area_id = self._get_nested((CNs.PickDetectorDepth, PcKeys.areaid))
        try:
            return BBox(area[area_id], from_wd=True)
        except:
            return None

    @exception_handler
    def draw_pickd_area_bb(
        self,
        num: int = -1,
        dsize: Dsize = None,
        *args,
        **kwargs,
    ) -> np.ndarray:
        visible = self.get_pickd_visible(num)
        area_corners = self._get_nested((CNs.PickDetectorDepth, PcKeys.area_corners))
        area_id = self._get_nested((CNs.PickDetectorDepth, PcKeys.areaid))
        pts = np.array([area_corners[area_id]], dtype=np.int32)
        cv2.polylines(visible, pts, isClosed=True, color=(0, 255, 255), thickness=8)

        return visible

    ##########

    @exception_handler
    def get_pickd_all_bb_od(self, num: int = -1) -> List[BBox]:
        bb_info_list = self._get_nested((CNs.PickDetectorDepth, PcKeys.bb_info), list())
        return [BBox(bb_info[2:], score=bb_info[1]) for bb_info in bb_info_list[num]]

    pickd_all_bb_od: List[BBox] = property(get_pickd_all_bb_od)

    @exception_handler
    def draw_pickd_all_bb_od(
        self,
        num: int = -1,
        dsize: Dsize = None,
        *args,
        **kwargs,
    ) -> Optional[np.ndarray]:
        visible = self.get_pickd_visible(num)
        if visible is None:
            return
        bb_list = self.get_pickd_all_bb_od(num)
        return draw_bboxes(visible, bb_list, dsize=dsize, *args, **kwargs)

    ##########

    @exception_handler
    def get_pickd_bb_valid(self, num: int = -1) -> Optional[List[BBox]]:
        bb_info_list = self._get_nested((CNs.PickDetectorDepth, PcKeys.bb_valid))
        if bb_info_list is None:
            return
        return [
            BBox(bb_info["bb"], score=bb_info["conf"], from_wd=True)
            for bb_info in bb_info_list[num]
        ]

    pickd_bb_valid: List[BBox] = property(get_pickd_bb_valid)

    @exception_handler
    def draw_pickd_bb_valid(
        self,
        num: int = -1,
        dsize: Dsize = None,
        *args,
        **kwargs,
    ) -> Optional[np.ndarray]:
        visible = self.get_pickd_trimmed_visible(num)
        if visible is None:
            return
        bb_list = self.get_pickd_bb_valid(num)
        if bb_list is None:
            return
        return draw_bboxes(visible, bb_list, dsize=dsize, *args, **kwargs)

    ##########

    @exception_handler
    def get_pickd_bb_ordered(self, num: int = -1) -> List[BBox]:
        bb_info_list = self._get_nested((CNs.PickDetectorDepth, PcKeys.bb_ordered))
        return [
            BBox(
                bb_info["bb_od"],
                score=bb_info["conf_od"],
                from_wd=True,
            )
            for bb_info in bb_info_list[num]
        ]

    pickd_bb_ordered: List[BBox] = property(get_pickd_bb_ordered)

    @exception_handler
    def draw_pickd_bb_ordered(
        self,
        num: int = -1,
        dsize: Dsize = None,
        *args,
        **kwargs,
    ) -> np.ndarray:
        visible = self.get_pickd_trimmed_visible(num)
        bb_list = self.get_pickd_bb_ordered(num)
        labels = [f"{i+1}: {bb.score:.3f}, {bb.label}" for i, bb in enumerate(bb_list)]
        return draw_bboxes(
            visible, bb_list, labels=labels, dsize=dsize, *args, **kwargs
        )

    ##########

    # @exception_handler
    # def get_pickd_a_obj_od(self, num: int = -1) -> BBox:
    #     bb_info = self._get_nested((CNs.PickDetectorDepth, PcKeys.a_obj))
    #     return BBox(bb_info[num]["bb_od"], score=bb_info[num]["conf_od"], from_wd=True)

    # pickd_a_obj_od: BBox = property(get_pickd_a_obj_od)

    # @exception_handler
    # def draw_pickd_a_obj_od(
    #     self,
    #     vis_num: int = -1,
    #     bb_num: int = -1,
    #     dsize: Dsize = None,
    #     *args,
    #     **kwargs,
    # ) -> np.ndarray:
    #     visible = self.get_pickd_trimmed_visible(vis_num)
    #     bb_info = self.get_pickd_a_obj_od(bb_num)
    #     return draw_bbox(visible, bb_info, dsize=dsize, *args, **kwargs)

    ##########

    # @exception_handler
    # def get_pickd_a_obj_td(self, num: int = -1) -> BBox:
    #     bb_info = self._get_nested((CNs.PickDetectorDepth, PcKeys.a_obj))
    #     if num is None:
    #         return [BBox(bb["bb_td"], from_wd=True) for bb in bb_info]
    #     return BBox(bb_info[num]["bb_td"], from_wd=True)

    # pickd_a_obj_td: BBox = property(get_pickd_a_obj_td)

    # @exception_handler
    # def draw_pickd_a_obj_td(
    #     self,
    #     vis_num: int = -1,
    #     bb_num: int = -1,
    #     dsize: Dsize = None,
    #     *args,
    #     **kwargs,
    # ) -> np.ndarray:
    #     visible = self.get_pickd_trimmed_visible(vis_num)
    #     bb_info = self.get_pickd_a_obj_td(bb_num)
    #     return draw_bbox(visible, bb_info, dsize=dsize, *args, **kwargs)

    #############################################################################
    ## pickedepth segms ##

    @property
    def pickd_segms_paths(self) -> List[str]:
        return self._get_paths(CNs.PickDetectorDepth, PcKeys.segms)

    @exception_handler
    def get_pickd_segms(
        self,
        num: int = -1,
        dsize: Dsize = None,
    ) -> Optional[np.ndarray]:
        if has_index(self.pickd_segms_paths, num):
            segms = load_array(self.pickd_segms_paths[num], dsize=dsize)
            return segms

    pickd_segms = property(get_pickd_segms)

    @exception_handler
    def get_pickd_segms_comb(
        self,
        num: int = -1,
        dsize: Dsize = None,
    ) -> Optional[np.ndarray]:
        segms = self.get_pickd_segms(num=num)
        if len(segms) == 0:
            return None

        comb = np.zeros_like(self.get_pickd_visible()[..., 0])
        for segm, bbox in zip(segms, self.get_pickd_all_bb_od()):
            x_min, y_min, x_max, y_max = bbox.coords
            x_max = x_min + segm.shape[1]
            y_max = y_min + segm.shape[0]
            comb[y_min:y_max, x_min:x_max] = np.logical_or(
                segm, comb[y_min:y_max, x_min:x_max]
            )

        if dsize is not None:
            dsize = get_dsize(comb, dsize)
            comb = cv2.resize(comb, dsize=dsize)
        return comb

    @exception_handler
    def draw_pickd_segms(
        self,
        num: int = -1,
        dsize: Dsize = None,
    ) -> Optional[np.ndarray]:
        background = self.get_pickd_visible(num=num, dsize=dsize)
        if background is None:
            return
        comb = self.get_pickd_segms_comb(num=num, dsize=dsize)
        if comb is None:
            return
        foreground = (np.stack([comb] * 3, -1) * 255).astype(np.uint8)
        return self._draw_overlay(background, foreground)

    #############################################################################

    #############################################################################
    ## pickedepth segms ##

    @property
    def pickd_segms_valid_paths(self) -> List[str]:
        return self._get_paths(CNs.PickDetectorDepth, PcKeys.segms_valid)

    @exception_handler(list())
    def get_pickd_segms_valid(
        self,
        num: int = -1,
        dsize: Dsize = None,
    ) -> List[np.ndarray]:
        if has_index(self.pickd_segms_paths, num):
            arr_path = self.pickd_segms_valid_paths[num]
            segms = load_array(arr_path, dsize=dsize)
            return segms
        return list()

    pickd_segms_valid = property(get_pickd_segms_valid)

    @exception_handler
    def get_pickd_segms_valid_comb(
        self,
        num: int = -1,
        dsize: Dsize = None,
    ) -> np.ndarray:
        segms = self.get_pickd_segms_valid(num=num)
        if len(segms) == 0:
            return None

        comb = np.zeros_like(self.get_pickd_visible()[..., 0])
        for segm, bbox in zip(segms, self.get_pickd_all_bb_od()):
            x_min, y_min, x_max, y_max = bbox.coords
            x_max = x_min + segm.shape[1]
            y_max = y_min + segm.shape[0]
            comb[y_min:y_max, x_min:x_max] = np.logical_or(
                segm, comb[y_min:y_max, x_min:x_max]
            )

        if dsize is not None:
            dsize = get_dsize(comb, dsize)
            comb = cv2.resize(comb, dsize=dsize)
        return comb

    @exception_handler
    def draw_pickd_segms_valid(
        self,
        num: int = -1,
        dsize: Dsize = None,
    ) -> np.ndarray:
        background = self.get_pickd_visible(num=num, dsize=dsize)
        foreground = (
            np.stack([self.get_pickd_segms_valid_comb(num=num, dsize=dsize)] * 3, -1)
            * 255
        ).astype(np.uint8)
        return self._draw_overlay(background, foreground)

    #############################################################################
    ## pickedepth PCDs ##

    @exception_handler
    def get_tote_area_pcd(self, nan: Optional[str] = None, remove_noise: bool = True):
        trimmed_depth = self.get_pickd_trimmed_depth(nan=nan)
        return self.depth2pcd(trimmed_depth, remove_noise=remove_noise)

    @exception_handler(list())
    def get_pickd_ol_pcds(self, num: int = -1) -> List:
        pmap = self.get_pickd_depth()
        area = self.PickDetectorDepth[PcKeys.area]
        area_num = self.PickDetectorDepth[PcKeys.areaid]

        segms = self.get_pickd_segms_valid(num)
        masks = [1 - segm for segm in segms]
        objs = self.PickDetectorDepth.get(PcKeys.bb_valid)
        if not objs:
            return []
        objs = objs[num]

        ol_pcds = [
            get_pcd_of_obj_from_pmap(obj, pmap, area, area_num, mask=mask)
            for obj, mask in zip(objs, masks)
        ]
        return ol_pcds

    #############################################################################
    ## pickedepth result image ##

    @property
    def pickd_result_image_paths(self) -> List[str]:
        return self._get_paths(CNs.PickDetectorDepth, PcKeys.result_image)

    @exception_handler
    def get_pickd_result_image(
        self,
        num: int = -1,
        dsize: Dsize = None,
    ) -> Optional[np.ndarray]:
        if has_index(self.pickd_result_image_paths, num):
            image = load_array(self.pickd_result_image_paths[num], dsize=dsize)
            return image

    #############################################################################
    ## pickedepth ai_image image ##

    @property
    def pickd_ai_image_paths(self) -> List[str]:
        return self._get_paths(CNs.PickDetectorDepth, PcKeys.ai_image)

    @exception_handler
    def get_pickd_ai_image(
        self,
        num: int = -1,
        dsize: Dsize = None,
    ) -> Optional[np.ndarray]:
        if has_index(self.pickd_ai_image_paths, num):
            image = load_array(self.pickd_ai_image_paths[num], dsize=dsize)
            return image

    #############################################################################
    ## pickedepth cnn_image image ##

    @property
    def pickd_cnn_image_paths(self) -> List[str]:
        return self._get_paths(CNs.PickDetectorDepth, PcKeys.cnn_image)

    @exception_handler
    def get_pickd_cnn_image(
        self,
        num: int = -1,
        dsize: Dsize = None,
    ) -> Optional[np.ndarray]:
        if has_index(self.pickd_cnn_image_paths, num):
            image = load_array(self.pickd_cnn_image_paths[num], dsize=dsize)
            return image

    #############################################################################
    ## pickedepth vac_areas ##

    @property
    def pickd_vac_areas_paths(self) -> List[str]:
        return self._get_paths(CNs.PickDetectorDepth, PcKeys.vac_areas, 0, GKeys.data)

    @property
    def pickd_vac_areas_boxes(self) -> List[List[int]]:
        vac_areas = self._get_nested([CNs.PickDetectorDepth, PcKeys.vac_areas])
        return [vac_area[GKeys.aux]["box"] for vac_area in vac_areas]

    @exception_handler
    def get_pickd_vac_areas(
        self,
        num: int = -1,
        dsize: Dsize = None,
    ):
        if has_index(self.pickd_vac_areas_paths, num) and has_index(
            self.pickd_vac_areas_boxes, num
        ):
            vac_areas = load_array(self.pickd_vac_areas_paths[num], dsize=dsize)
            vac_box = self.pickd_vac_areas_boxes[num]
            return vac_areas, vac_box

    @property
    def pickd_vac_areas_all(self, dsize: Dsize = None):
        return [
            self.get_pickd_vac_areas(num, dsize=dsize)
            for num in range(len(self.pickd_vac_areas_paths))
        ]

    pickd_vac_areas = property(get_pickd_vac_areas)

    @exception_handler
    def get_pickd_vac_areas_comb(
        self, num: int = -1, dsize: Dsize = None
    ) -> np.ndarray:
        comb = np.zeros_like(self.get_pickd_trimmed_visible(num=num)[..., 0])

        for vac_areas, box in self.pickd_vac_areas_all:
            x_min, y_min, x_max, y_max = box
            for vac_area in vac_areas:
                comb[y_min:y_max, x_min:x_max] = np.logical_or(
                    vac_area, comb[y_min:y_max, x_min:x_max]
                )

        if dsize is not None:
            dsize = get_dsize(comb, dsize)
            comb = cv2.resize(comb, dsize=dsize)
        return comb

    @exception_handler
    def draw_pickd_vac_areas(
        self,
        num: int = -1,
        dsize: Dsize = None,
    ) -> np.ndarray:
        background = self.get_pickd_trimmed_visible(num=num, dsize=dsize)
        foreground = (
            np.stack([self.get_pickd_vac_areas_comb(num=num, dsize=dsize)] * 3, -1)
            * 255
        )
        return self._draw_overlay(background, foreground)

    #############################################################################
    ## pickedepth all_pickd_images ##

    @exception_handler
    def get_all_pickd_images(self, num: int = -1, get_all: bool = False):
        images = {
            "area_bb": self.draw_pickd_area_bb(num=num),
            "all_bb_od": self.draw_pickd_all_bb_od(num=num),
            "segms": self.draw_pickd_segms(num=num),
            "bb_valid": self.draw_pickd_bb_valid(num=num),
            "cnn_image": self.get_pickd_cnn_image(num=num),
            "ai_image": self.get_pickd_ai_image(num=num),
            "result_image": self.get_pickd_result_image(num=num),
        }
        if get_all:
            images.update(
                {
                    "visible": self.get_pickd_visible(num=num),
                    "trimmed_visible": self.get_pickd_trimmed_visible(num=num),
                    "depth_normalized": self.get_pickd_depth_normalized(num=num),
                    "trimmed_depth": self.get_pickd_trimmed_depth(num=num),
                    "bb_ordered": self.draw_pickd_bb_ordered(num=num),
                    "vac_areas": self.draw_pickd_vac_areas(num=num),
                    "pickd_plot": self.get_pickd_plot(num=num, return_plot=True),
                }
            )
        return images

    #############################################################################
    ## Size Check
    #############################################################################
    ## sizecheck visible ##

    @property
    def sizecheck_input_paths(self) -> Dict[str, str]:
        input_paths = dict()

        path = self.sizecheck_visible_path
        if path:
            ext = os.path.splitext(path)[-1]
            input_paths[f"sizecheck_visible{ext}"] = path
        path = self.sizecheck_depth_path
        if path:
            ext = os.path.splitext(path)[-1]
            input_paths[f"sizecheck_depth{ext}"] = path
        path = self.sizecheck_side_depth_path
        if path:
            ext = os.path.splitext(path)[-1]
            input_paths[f"sizecheck_side_depth{ext}"] = path
        return input_paths

    @property
    def sizecheck_visible_path(self) -> str:
        path = self._get_nested((CNs.SkuSizeChecker, SKeys.visible))
        return self._fix_path(path)

    @exception_handler
    def get_sizecheck_visible(self, dsize: Dsize = None) -> np.ndarray:
        return load_array(self.sizecheck_visible_path, dsize=dsize)

    sizecheck_visible: np.ndarray = property(get_sizecheck_visible)

    #############################################################################
    ## sizecheck depth ##

    @property
    def sizecheck_depth_path(self) -> str:
        path = self._get_nested((CNs.SkuSizeChecker, SKeys.depth))
        return self._fix_path(path)

    @exception_handler
    def get_sizecheck_depth(self, dsize: Dsize = None) -> np.ndarray:
        return load_array(self.sizecheck_depth_path, dsize=dsize)

    sizecheck_depth: np.ndarray = property(get_sizecheck_depth)

    @exception_handler
    def get_sizecheck_depth_normalized(self, dsize: Dsize = None) -> np.ndarray:
        array = self.get_sizecheck_depth(dsize=dsize)
        return ti_normalize_8b(array[..., -1], nan="min")

    sizecheck_depth_normalized: np.ndarray = property(get_sizecheck_depth_normalized)

    #############################################################################
    ## sizecheck side_depth ##

    @property
    def sizecheck_side_depth_path(self) -> str:
        path = self._get_nested((CNs.SkuSizeChecker, SKeys.side_points))
        return self._fix_path(path)

    @exception_handler
    def get_sizecheck_side_depth(self, dsize: Dsize = None) -> np.ndarray:
        return load_array(self.sizecheck_side_depth_path, dsize=dsize)

    sizecheck_side_depth: np.ndarray = property(get_sizecheck_side_depth)

    @exception_handler
    def get_sizecheck_side_depth_normalized(self, dsize: Dsize = None) -> np.ndarray:
        array = self.get_sizecheck_side_depth(dsize=dsize)
        return ti_normalize_8b(array, nan="min")

    sizecheck_depth_normalized: np.ndarray = property(
        get_sizecheck_side_depth_normalized
    )

    #############################################################################
    ## sizecheck visible trimmed ##

    @property
    def sizecheck_trimmed_visible_path(self) -> str:
        path = self._get_nested((CNs.SkuSizeChecker, SKeys.trimmed_visible))
        return self._fix_path(path)

    @exception_handler
    def get_sizecheck_trimmed_visible(self, dsize: Dsize = None) -> np.ndarray:
        return load_array(self.sizecheck_trimmed_visible_path, dsize=dsize)

    sizecheck_trimmed_visible: np.ndarray = property(get_sizecheck_trimmed_visible)

    #############################################################################
    ## sizecheck depth trimmed ##

    @property
    def sizecheck_trimmed_depth_path(self) -> str:
        path = self._get_nested((CNs.SkuSizeChecker, SKeys.trimmed_depth))
        return self._fix_path(path)

    @exception_handler
    def get_sizecheck_trimmed_depth(self, dsize: Dsize = None) -> np.ndarray:
        path = load_array(self.sizecheck_trimmed_depth_path, dsize=dsize)
        return self._fix_path(path)

    sizecheck_trimmed_depth: np.ndarray = property(get_sizecheck_trimmed_depth)

    @exception_handler
    def get_sizecheck_trimmed_depth_normalized(self, dsize: Dsize = None) -> np.ndarray:
        array = self.get_sizecheck_trimmed_depth(dsize=dsize)
        return cv2.convertScaleAbs(array)

    sizecheck_trimmed_depth_normalized: np.ndarray = property(
        get_sizecheck_trimmed_depth_normalized
    )

    #############################################################################
    ## sizecheck depth wo noise ##

    @property
    def sizecheck_depth_wo_noise_path(self) -> str:
        path = self._get_nested((CNs.SkuSizeChecker, SKeys.depth_wo_noise))
        return self._fix_path(path)

    @exception_handler
    def get_sizecheck_depth_wo_noise(self, dsize: Dsize = None) -> np.ndarray:
        return load_array(self.sizecheck_depth_wo_noise_path, dsize=dsize)

    sizecheck_depth_wo_noise: np.ndarray = property(get_sizecheck_depth_wo_noise)

    @exception_handler
    def get_sizecheck_depth_wo_noise_normalized(
        self, dsize: Dsize = None
    ) -> np.ndarray:
        array = self.get_sizecheck_depth_wo_noise(dsize=dsize)
        return cv2.convertScaleAbs(array)

    sizecheck_depth_wo_noise_normalized: np.ndarray = property(
        get_sizecheck_depth_wo_noise_normalized
    )

    #############################################################################
    ## sizecheck seg_output ##

    @property
    def sizecheck_seg_output_path(self) -> str:
        path = self._get_nested((CNs.SkuSizeChecker, SKeys.seg_output))
        return self._fix_path(path)

    @exception_handler
    def get_sizecheck_seg_output(self, dsize: Dsize = None) -> np.ndarray:
        return load_array(self.sizecheck_seg_output_path, dsize=dsize)

    sizecheck_seg_output: np.ndarray = property(get_sizecheck_seg_output)

    #############################################################################
    ## sizecheck bin_seg ##

    @exception_handler
    def get_sizecheck_bin_seg(self, dsize: Dsize = None) -> Optional[np.ndarray]:
        roi_left_top = self._get_nested((CNs.SkuSizeChecker, SKeys.roi_left_top))
        if roi_left_top is None:
            return
        roi_right_bottom = self._get_nested(
            (CNs.SkuSizeChecker, SKeys.roi_right_bottom)
        )
        if roi_right_bottom is None:
            return
        bin_seg = self.sizecheck_seg_output[
            roi_left_top[1] : roi_right_bottom[1], roi_left_top[0] : roi_right_bottom[0]
        ]

        if dsize is not None:
            dsize = get_dsize(bin_seg, dsize)
            bin_seg = cv2.resize(bin_seg, dsize=dsize)
        return bin_seg

    sizecheck_bin_seg: Optional[np.ndarray] = property(get_sizecheck_bin_seg)

    #############################################################################
    ## sizecheck bin_depth ##

    @property
    def sizecheck_bin_depth_path(self) -> str:
        path = self._get_nested((CNs.SkuSizeChecker, SKeys.bin_depth))
        return self._fix_path(path)

    @exception_handler
    def get_sizecheck_bin_depth(self, dsize: Dsize = None) -> np.ndarray:
        return load_array(self.sizecheck_bin_depth_path, dsize=dsize)

    sizecheck_bin_depth: np.ndarray = property(get_sizecheck_bin_depth)

    #############################################################################
    ## sizecheck bin_side_depth ##

    @property
    def sizecheck_bin_side_depth_path(self) -> str:
        path = self._get_nested((CNs.SkuSizeChecker, SKeys.binary_side_depth))
        return self._fix_path(path)

    @exception_handler
    def get_sizecheck_bin_side_depth(self, dsize: Dsize = None) -> np.ndarray:
        return load_array(self.sizecheck_bin_side_depth_path, dsize=dsize)

    sizecheck_bin_side_depth: np.ndarray = property(get_sizecheck_bin_side_depth)

    #############################################################################
    ## sizecheck bin_all ##

    @property
    def sizecheck_bin_all_path(self) -> str:
        path = self._get_nested((CNs.SkuSizeChecker, SKeys.bin_all))
        return self._fix_path(path)

    @exception_handler
    def get_sizecheck_bin_all(self, dsize: Dsize = None) -> np.ndarray:
        return load_array(self.sizecheck_bin_all_path, dsize=dsize)

    sizecheck_bin_all: np.ndarray = property(get_sizecheck_bin_all)

    #############################################################################
    ## sizecheck proj_img ##

    @property
    def sizecheck_proj_img_path(self) -> str:
        path = self._get_nested((CNs.SkuSizeChecker, SKeys.proj_img))
        return self._fix_path(path)

    @exception_handler
    def get_sizecheck_proj_img(self, dsize: Dsize = None) -> np.ndarray:
        return load_array(self.sizecheck_proj_img_path, dsize=dsize)

    sizecheck_proj_img: np.ndarray = property(get_sizecheck_proj_img)

    #############################################################################
    ## sizecheck out_info ##

    @property
    def sizecheck_out_info_path(self) -> str:
        path = self._get_nested((CNs.SkuSizeChecker, SKeys.out_info))
        return self._fix_path(path)

    @exception_handler
    def get_sizecheck_out_info(self, dsize: Dsize = None) -> np.ndarray:
        return load_array(self.sizecheck_out_info_path, dsize=dsize)

    sizecheck_out_info: np.ndarray = property(get_sizecheck_out_info)

    #############################################################################
    ## sizecheck all_sizecheck_images ##

    @exception_handler
    def get_all_sizecheck_images(self, dsize: Dsize = None, get_all: bool = False):
        images = {
            "visible": self.get_sizecheck_visible(),
            "bin_seg": self.get_sizecheck_bin_seg(),
            "bin_depth": self.get_sizecheck_bin_depth(),
            "bin_all": self.get_sizecheck_bin_all(),
            "bin_side_depth": self.get_sizecheck_bin_side_depth(),
            "out_info": self.get_sizecheck_out_info(),
        }
        if get_all:
            images.update(
                {
                    "depth_normalized": self.get_sizecheck_depth_normalized(),
                    "side_depth_normalized": self.get_sizecheck_side_depth_normalized(),
                    "trimmed_visible": self.get_sizecheck_trimmed_visible(),
                    "trimmed_depth_normalized": self.get_sizecheck_trimmed_depth_normalized(),
                    "depth_wo_noise_normalized": self.get_sizecheck_depth_wo_noise_normalized(),
                    "seg_output": self.get_sizecheck_seg_output(),
                    "proj_img": self.get_sizecheck_proj_img(),
                }
            )
        return images

    #############################################################################
    ## Placement
    #############################################################################
    ## kwargs ##

    @property
    def placed_check_kwargs(self) -> Dict:
        return self._get_nested((CNs.PlaceDetectorDepth, PlKeys.check_kwargs))

    @property
    def placed_detect_kwargs(self) -> Dict:
        return self._get_nested((CNs.PlaceDetectorDepth, PlKeys.detect_kwargs))

    @property
    def placed_input_paths(self) -> Dict[str, str]:
        input_paths = dict()

        path = self.placed_visible_path
        if path:
            ext = os.path.splitext(path)[-1]
            input_paths[f"placed_visible{ext}"] = path
        path = self.placed_depth_path
        if path:
            ext = os.path.splitext(path)[-1]
            input_paths[f"placed_depth{ext}"] = path
        return input_paths

    ## placedepth visible ##

    @property
    def placed_visible_path(self) -> List[str]:
        path = self._get_nested((CNs.PlaceDetectorDepth, PlKeys.visible))
        return self._fix_path(path)

    def get_placed_visible(
        self,
        dsize: Dsize = None,
    ) -> np.ndarray:
        try:
            return load_array(self.placed_visible_path, dsize=dsize)
        except:
            return None

    placed_visible = property(get_placed_visible)

    #############################################################################
    ## placedepth depth ##

    @property
    def placed_depth_path(self) -> List[str]:
        path = self._get_nested((CNs.PlaceDetectorDepth, PlKeys.depth))
        return self._fix_path(path)

    @exception_handler
    def get_placed_depth(
        self,
        dsize: Dsize = None,
    ) -> np.ndarray:
        return load_array(self.placed_depth_path, dsize=dsize)

    placed_depth = property(get_placed_depth)

    ##########

    @exception_handler
    def get_placed_depth_normalized(self, dsize: Dsize = None) -> np.ndarray:
        array = self.get_placed_depth(dsize=dsize)
        return ti_normalize_8b(array[..., -1], nan="min")

    placed_depth_normalized: np.ndarray = property(get_placed_depth_normalized)

    #############################################################################
    ## placedepth visible trimmed ##

    @property
    def placed_trimmed_visible_path(self) -> List[str]:
        path = self._get_nested((CNs.PlaceDetectorDepth, PlKeys.trimmed_visible))
        return self._fix_path(path)

    @exception_handler
    def get_placed_trimmed_visible(self, dsize: Dsize = None) -> np.ndarray:
        return crop_bbox(self.placed_visible, self.placed_area_bb, dsize=dsize)

    placed_trimmed_visible: np.ndarray = property(get_placed_trimmed_visible)

    #############################################################################
    ## placedepth depth trimmed ##

    @property
    def placed_trimmed_depth_path(self) -> List[str]:
        path = self._get_nested((CNs.PlaceDetectorDepth, PlKeys.trimmed_depth))
        return self._fix_path(path)

    @exception_handler
    def get_placed_trimmed_depth(self, dsize: Dsize = None) -> np.ndarray:
        return load_array(self.placed_trimmed_depth_path, dsize=dsize)

    placed_trimmed_depth: np.ndarray = property(get_placed_trimmed_depth)

    @property
    def placed_trimmed_depth_cleared_path(self) -> List[str]:
        path = self._get_nested((CNs.PlaceDetectorDepth, "trimmed_depth_cleared"))
        return self._fix_path(path)

    @exception_handler
    def get_placed_trimmed_depth_cleared(self, dsize: Dsize = None) -> np.ndarray:
        return load_array(self.placed_trimmed_depth_cleared_path, dsize=dsize)

    placed_trimmed_depth_cleared: np.ndarray = property(
        get_placed_trimmed_depth_cleared
    )

    #############################################################################
    ## placedepth debug_image ##

    @property
    def placed_debug_image_path(self) -> str:
        path = self._get_nested((CNs.PlaceDetectorDepth, PlKeys.debug_image))
        return self._fix_path(path)

    @exception_handler
    def get_placed_debug_image(self, dsize: Dsize = None) -> np.ndarray:
        return load_array(self.placed_debug_image_path, dsize=dsize)

    placed_debug_image: np.ndarray = property(get_placed_debug_image)

    #############################################################################
    ## placed mappings ##

    @property
    def placed_max_mapping_path(self) -> Optional[str]:
        path = self._get_nested((CNs.PlaceDetectorDepth, "max_mapping"))
        return self._fix_path(path)

    @exception_handler
    def get_placed_max_mapping(self, dsize: Dsize = None) -> Optional[np.ndarray]:
        return load_array(self.placed_max_mapping_path, dsize=dsize)

    placed_max_mapping: np.ndarray = property(get_placed_max_mapping)

    @property
    def placed_lower_max_mapping_path(self) -> Optional[str]:
        path = self._get_nested((CNs.PlaceDetectorDepth, "lower_max_mapping"))
        return self._fix_path(path)

    @exception_handler
    def get_placed_lower_max_mapping(self, dsize: Dsize = None) -> Optional[np.ndarray]:
        return load_array(self.placed_lower_max_mapping_path, dsize=dsize)

    placed_lower_max_mapping: np.ndarray = property(get_placed_lower_max_mapping)

    @property
    def placed_lower_mapping_path(self) -> Optional[str]:
        path = self._get_nested((CNs.PlaceDetectorDepth, "lower_mapping"))
        return self._fix_path(path)

    @exception_handler
    def get_placed_lower_mapping(self, dsize: Dsize = None) -> Optional[np.ndarray]:
        return load_array(self.placed_lower_mapping_path, dsize=dsize)

    placed_lower_mapping: np.ndarray = property(get_placed_lower_mapping)

    @property
    def placed_avg_mapping_path(self) -> Optional[str]:
        path = self._get_nested((CNs.PlaceDetectorDepth, "avg_mapping"))
        return self._fix_path(path)

    @exception_handler
    def get_placed_avg_mapping(self, dsize: Dsize = None) -> Optional[np.ndarray]:
        return load_array(self.placed_avg_mapping_path, dsize=dsize)

    placed_avg_mapping: np.ndarray = property(get_placed_avg_mapping)

    @exception_handler
    def get_placed_depth_sku_image(self, dsize: Dsize = None) -> Optional[np.ndarray]:
        sku_img = self.get_placed_trimmed_depth_cleared()
        trimmed_rc = self._get_nested((CNs.PlaceDetectorDepth, "trimmed_rc"))
        dims = self._get_nested((CNs.PlaceDetectorDepth, "dims"))
        if any([elem is None for elem in (sku_img, trimmed_rc, dims)]):
            return
        r_min = trimmed_rc[0] - dims[0] // 2
        c_min = trimmed_rc[1] - dims[1] // 2
        sku_img = sku_img.copy()
        sku_img[
            r_min : r_min + dims[0],
            c_min : c_min + dims[1],
        ] = self.get_placed_trimmed_depth_cleared().max()
        return sku_img

    placed_depth_sku_image: np.ndarray = property(get_placed_depth_sku_image)

    @exception_handler
    def get_placed_visible_sku_image(self, dsize: Dsize = None) -> Optional[np.ndarray]:
        sku_img = self.get_placed_trimmed_visible()
        trimmed_rc = self._get_nested((CNs.PlaceDetectorDepth, "trimmed_rc"))
        dims = self._get_nested((CNs.PlaceDetectorDepth, "dims"))
        if any([elem is None for elem in (sku_img, trimmed_rc, dims)]):
            return
        r_min = trimmed_rc[0] - dims[0] // 2
        c_min = trimmed_rc[1] - dims[1] // 2
        sku_img = sku_img.copy()

        sku_img[
            r_min : r_min + dims[0],
            c_min : c_min + dims[1],
        ] = [255, 255, 255]
        return sku_img

    placed_visible_sku_image: np.ndarray = property(get_placed_visible_sku_image)

    #############################################################################
    ## placed bboxes ##

    @property
    def placed_area_bb(self):
        area = self._get_nested((CNs.PlaceDetectorDepth, PlKeys.area))
        area_id = self._get_nested((CNs.PlaceDetectorDepth, PlKeys.areaid))
        return BBox(area[area_id], from_wd=True)

    @exception_handler
    def draw_placed_area_bb(
        self,
        dsize: Dsize = None,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return draw_bbox(
            self.placed_visible, self.placed_area_bb, dsize=dsize, *args, **kwargs
        )

    ##########

    @exception_handler
    def get_placed_all_bb_od(self) -> List[BBox]:
        bb_info_list = self._get_nested(
            (CNs.PlaceDetectorDepth, PlKeys.bb_info), list()
        )
        return [BBox(bb_info[2:], score=bb_info[1]) for bb_info in bb_info_list]

    placed_all_bb_od: List[BBox] = property(get_placed_all_bb_od)

    @exception_handler
    def draw_placed_all_bb_od(
        self,
        dsize: Dsize = None,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return draw_bboxes(
            self.placed_visible, self.placed_all_bb_od, dsize=dsize, *args, **kwargs
        )

    ##########

    @exception_handler
    def get_placed_bb_valid(self, num: int = -1) -> List[BBox]:
        bb_list = self.placed_all_bb_od
        bb_valid_inds = self._get_nested(
            (CNs.PlaceDetectorDepth, PlKeys.bb_valid_inds), list()
        )
        return [bb_list[i] for i in bb_valid_inds]

    placed_bb_valid: List[BBox] = property(get_placed_bb_valid)

    @exception_handler
    def draw_placed_bb_valid(
        self,
        dsize: Dsize = None,
        *args,
        **kwargs,
    ) -> np.ndarray:
        visible = self.get_placed_visible()
        bb_list = self.get_placed_bb_valid()
        return draw_bboxes(visible, bb_list, dsize=dsize, *args, **kwargs)

    #############################################################################
    ## placed all_placed_images ##

    @exception_handler
    def get_all_placed_images(self, dsize: Dsize = None, get_all: bool = False):
        images = {
            # **{
            #     "sku_image_visible": ti_normalize_8b(self.get_placed_visible_sku_image(dsize=dsize))
            #     if self.get_placed_visible_sku_image(dsize=dsize) is not None
            #     else {}
            # },
            "debug_image": self.get_placed_debug_image(dsize=dsize),
            "area_bb": self.draw_placed_area_bb(dsize=dsize),
            "all_bb_od": self.draw_placed_all_bb_od(dsize=dsize),
            "bb_valid": self.draw_placed_bb_valid(dsize=dsize),
            "max_mapping": self.get_placed_max_mapping(dsize=dsize),
            **{
                "lower_max_mapping": (
                    self.get_placed_lower_max_mapping(dsize=dsize).astype(np.uint8)
                    * 255
                    if self.get_placed_lower_max_mapping(dsize=dsize) is not None
                    else {}
                )
            },
            **{
                "lower_mapping": (
                    self.get_placed_lower_mapping(dsize=dsize).astype(np.uint8) * 255
                    if self.get_placed_lower_mapping(dsize=dsize) is not None
                    else {}
                )
            },
            "avg_mapping": self.get_placed_avg_mapping(dsize=dsize),
        }
        if get_all:
            images.update(
                {
                    "visible": self.get_placed_visible(dsize=dsize),
                    "trimmed_visible": self.get_placed_trimmed_visible(dsize=dsize),
                    "depth_normalized": self.get_placed_depth_normalized(dsize=dsize),
                    "trimmed_depth": self.get_placed_trimmed_depth(dsize=dsize),
                    "checkplace_plot": self.get_checkplace_plot(return_plot=True),
                    "placed_plot": self.get_placed_plot(return_plot=True),
                }
            )
        return images

    #############################################################################
    ## important images ##

    @exception_handler
    def get_important_images_paths_methods(self, num: int = -1):
        try:
            cnn_image = self.pickd_cnn_image_paths[num]
        except:
            cnn_image = None
        try:
            ai_image = self.pickd_ai_image_paths[num]
        except:
            ai_image = None
        try:
            pick_image = self.pickd_result_image_paths[num]
        except:
            pick_image = None
        try:
            sizecheck_vis_image = self.sizecheck_trimmed_visible_path
        except:
            sizecheck_vis_image = None
        try:
            sizecheck_out_image = self.sizecheck_out_info_path
        except:
            sizecheck_out_image = None
        try:
            place_image = self.placed_debug_image_path
        except:
            place_image = None
        paths = {
            # --- pickd
            "cnn_image": (
                cnn_image,
                "get_pickd_cnn_image",
            ),
            "ai_image": (
                ai_image,
                "get_pickd_ai_image",
            ),
            "pick_image": (
                pick_image,
                "get_pickd_result_image",
            ),
            # --- sizecheck
            "sizecheck_vis_image": (
                sizecheck_vis_image,
                "get_sizecheck_trimmed_visible",
            ),
            "sizecheck_out_image": (
                sizecheck_out_image,
                "get_sizecheck_out_info",
            ),
            # --- placed
            "place_image": (
                place_image,
                "get_placed_debug_image",
            ),
        }
        return paths

    @exception_handler
    def get_important_images(self, num: int = -1):
        images = {
            # --- pickd
            "cnn_image": self.get_pickd_cnn_image(num=num),
            "ai_image": self.get_pickd_ai_image(num=num),
            "pick_image": self.get_pickd_result_image(num=num),
            # --- sizecheck
            "sizecheck_vis_image": self.get_sizecheck_trimmed_visible(),
            "sizecheck_out_image": self.get_sizecheck_out_info(),
            # --- placed
            "place_image": self.get_placed_debug_image(),
        }
        return images

    #############################################################################
    ## Actions
    #############################################################################
    ## ActionManagerNode ##

    @property
    def action_manager_msgs(self) -> List[Dict]:
        return self._get_nested((CNs.ActionManagerNode, AMKeys.msg), list())

    @memoize
    @exception_handler(list())
    def get_actions(
        self,
        steps_included: Optional[List[str]] = None,
        actions_included: Optional[List[str]] = None,
        only_successful: bool = False,
        only_failed: bool = False,
    ) -> List[ActionLog]:
        try:
            actions: List[ActionLog] = [
                ActionLog(**action_dict)
                for action_dict in self._dict[AcKeys.action_results]
            ]
        except:
            return []
        if steps_included is not None:
            actions = [a for a in actions if a.step in steps_included]
        if actions_included is not None:
            actions = [a for a in actions if a.name in actions_included]
        if only_successful:
            actions = [a for a in actions if a.is_successful]
        if only_failed:
            actions = [a for a in actions if not a.is_successful]
        actions.sort(key=lambda x: x.start_time)
        return actions

    def get_fail_type(self):
        if self.is_successful:
            return None
        steps_actions = [
            (StepNames.imagingPick, ActionNames.capture),
            (StepNames.imagingPlace, ActionNames.capture),
            (StepNames.movingToPick, ActionNames.sequence),
            (StepNames.detectingPick, ActionNames.detectpick),
            (StepNames.checkingPlace, ActionNames.checkplace),
            (StepNames.picking, ActionNames.topick),
            (StepNames.picking, ActionNames.grasp),
            (StepNames.movingFromPick, ActionNames.frompick),
            (StepNames.movingToSizeCheck, ActionNames.sequence),
            (StepNames.imagingSize, ActionNames.capture),
            (StepNames.sizeChecking, ActionNames.checksku),
            (StepNames.detectingPlace, ActionNames.detectplace),
            (StepNames.movingToPlace, ActionNames.sequence),
            (StepNames.detectingPlace, ActionNames.planplace),
            (StepNames.placing, ActionNames.toplace),
            (StepNames.placing, ActionNames.release),
        ]
        for step, action in steps_actions:
            if not self.get_actions(step, action, only_successful=True):
                return f"{step}.{action} Fail"

    @memoize
    @property
    def start_time(self) -> Optional[datetime.datetime]:
        """
        start time: time when the first action of the item starts
        """
        actions = self.get_actions()
        start_times = [action.start_time for action in actions]
        if len(start_times) == 0:
            return None
        return min(start_times)

    @memoize
    @property
    def end_time(self) -> Optional[datetime.datetime]:
        """
        end time: time when the last action of the item ends
        """
        actions = self.get_actions()
        end_times = [action.end_time for action in actions]
        if len(end_times) == 0:
            return None
        return max(end_times)

    @property
    def total_duration(self) -> Optional[float]:
        start_time = self.start_time
        end_time = self.end_time
        if start_time and end_time:
            return (end_time - start_time).total_seconds()
        return None

    @memoize
    @property
    def aggregated_duration(self) -> Optional[float]:
        intervals = [
            (action.start_time, action.end_time) for action in self.get_actions()
        ]
        return get_agregated_duration(intervals)

    @exception_handler(list())
    def get_critical_path_actions(self) -> List[ActionLog]:
        actions = self.get_actions()
        entries = [[action, action.start_time, action.end_time] for action in actions]
        if not entries:
            return []
        entries.sort(key=lambda x: x[1])
        critical_path_actions = [entries[0]]
        for entry in entries[1:]:
            if entry[2] > critical_path_actions[-1][2]:
                critical_path_actions[-1][2] = min(
                    critical_path_actions[-1][2], entry[1]
                )
                # entry[1] = max(entry[1], critical_path_actions[-1][2])
                critical_path_actions.append(entry)
        return critical_path_actions

    @exception_handler
    def get_critical_path_categories_durations(self) -> Optional[Dict[str, float]]:
        critical_path_actions = self.get_critical_path_actions()
        cp_categories_durations = defaultdict(lambda: 0)
        for action, start_time, end_time in critical_path_actions:
            category = action_mapping.get(action.name)
            if category is None:
                return None
            duration = end_time - start_time
            cp_categories_durations[category] += duration.total_seconds()
        return dict(cp_categories_durations)

    @exception_handler
    def get_actions_plot(
        self, return_plot: bool = False, add_times: bool = False, full_name=False
    ) -> Optional[np.ndarray]:
        actions = self.get_actions()
        if full_name:
            names = [action.full_name for action in actions]
        else:
            names = [action.name for action in actions]
        durations = [action.duration for action in actions]
        start_timestamps = [action.start_time for action in actions]
        end_timestamps = [action.end_time for action in actions]
        color_seeds = [action.step for action in actions]
        return draw_time_plot(
            "Actions",
            names,
            durations,
            start_timestamps,
            end_timestamps,
            self.total_duration,
            return_plot=return_plot,
            add_times=add_times,
            color_seeds=color_seeds,
        )

    #############################################################################
    ## pads ##
    #############################################################################

    @property
    def td_pcd_paths(self) -> List[str]:
        return self._get_paths(CNs.PcdProjector, PdKeys.td_pcd)

    def get_td_pcd_inliers_paths(self, size: int) -> List[str]:
        return self._get_paths(CNs.PcdProjector, f"{PdKeys.td_pcd_inliers}-{size}")

    @exception_handler
    def get_td_pcd(
        self,
        num: int = -1,
    ) -> Optional[np.ndarray]:
        import open3d as o3d

        if has_index(self.td_pcd_paths, num):
            pcd_array = load_array(self.td_pcd_paths[num])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_array)
            return pcd

    @exception_handler
    def get_td_pcd_inliers(
        self,
        size: int,
        num: int = -1,
    ) -> Optional[np.ndarray]:
        pcd = self.get_td_pcd(num=num)
        inliers = load_array(self.get_td_pcd_inliers_paths(size=size)[num])
        inlier_cloud = pcd.select_by_index(inliers)
        return inlier_cloud

    @exception_handler
    def get_plane_model(
        self,
        size: int,
        num: int = -1,
    ) -> Optional[List[float]]:
        plane_model = self._get_nested(
            (CNs.PcdProjector, f"{PdKeys.td_plane_model}-{size}", num)
        )
        return plane_model

    @exception_handler
    def get_plane_model_pcd(
        self,
        size: int,
        num: int = -1,
        x_range: Tuple[int, int] = (-50, 50),
        y_range: Tuple[int, int] = (-50, 50),
        pcd=None,
        grid_size: int = 50,
    ) -> Optional[List[float]]:
        import open3d as o3d

        plane_model = self.get_plane_model(size=size, num=num)
        a, b, c, d = plane_model

        if pcd is not None:
            points = np.asarray(pcd.points)
            x_min, y_min, _ = points.min(axis=0)
            x_max, y_max, _ = points.max(axis=0)

            x_range = (x_min, x_max)
            y_range = (y_min, y_max)

        x, y = np.meshgrid(
            np.linspace(x_range[0], x_range[1], grid_size),
            np.linspace(y_range[0], y_range[1], grid_size),
        )
        z = -(a * x + b * y + d) / c
        plane_points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

        plane_pcd = o3d.geometry.PointCloud()
        plane_pcd.points = o3d.utility.Vector3dVector(plane_points)

        return plane_pcd

    #############################################################################
    ## misc ##
    #############################################################################

    def __repr__(self) -> str:
        description = f"{self.order_id}"
        description += f", {self.timestamp_str}, {self.total_duration} secs"
        description += ", Success" if self.is_successful else ", Fail"

        return description

    @staticmethod
    def depth2pcd(depth: np.ndarray, remove_noise: bool = False):
        import open3d as o3d

        reshaped_data = depth.reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(reshaped_data)
        if remove_noise:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        return pcd

    def _get_nested(self, index, default=None, default_for_nonexistent=True):
        return get_nested_item(
            self._dict,
            index=index,
            default_for_nonexistent=default_for_nonexistent,
            default=default,
        )

    def _get_paths(self, *inds) -> List[str]:
        paths = self._get_nested(inds, [])
        if isinstance(paths, str):
            paths = self._fix_path(paths)
            paths = sorted(glob(os.path.join(paths, "*")))
        else:
            paths = self._fix_paths(paths)
        return paths

    def _fix_paths(self, paths: List[str]) -> List[str]:
        return [self._fix_path(path) for path in paths]

    def _fix_path(self, path: str) -> str:
        if self.oid_format is not None:
            return path.replace(self.or_data_oid_dir, self.data_oid_dir)
        return path

    @staticmethod
    def _fix_home(path):
        sep = os.path.sep
        home = sep + "home" + sep
        if path.startswith(home):
            after_path = sep.join(path.replace(home, "").split(sep)[1:])
            new_home = os.path.expanduser("~")
            return os.path.join(new_home, after_path)
        return path

    @staticmethod
    def _draw_overlay(
        background, foreground, alpha=0.5, beta=0.5, gamma=0.0, dtype=np.uint8
    ):
        overlay = cv2.addWeighted(background, alpha, foreground, beta, gamma)
        if dtype is not None:
            overlay = overlay.astype(dtype)
        return overlay

    @staticmethod
    def _clip_values(
        array: np.ndarray,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        std_num: Optional[int] = None,
    ):
        if std_num is not None:
            mean_value, std_value = np.mean(array), np.std(array)
            print(mean_value, std_value)
            min_value = mean_value - (std_num * std_value)
            max_value = mean_value + (std_num * std_value)
            print(min_value, max_value)
        return np.clip(array, min_value, max_value)

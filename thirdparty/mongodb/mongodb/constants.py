import os
import yaml
from enum import Enum

ITEM_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "db_cfgs/rps_logs.yaml")
MEMORY_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "db_cfgs/rps_memory.yaml")
SAFIE_PATH = os.path.join(os.path.dirname(__file__), "safie_cfg.yaml")
ITEM_COLLECTION = os.getenv("ITEM_COLLECTION") or "Item"
SKU_COLLECTION = os.getenv("SKU_COLLECTION") or "sku_info"
DOCUMENTS_DIR = "/data/documents/"
OID_MAPS_DIR = "/data/db_data/oid_maps"
PR_SEP = "."
JAN = "jan"
DATE_ID = "date_id"

MONGODB_LOGGER_QUEUE = "mongodb_logger_queue"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
ACTION = "action"
DATA = "data"

if os.getenv("WS_NAME"):
    WS_NAME = os.getenv("WS_NAME")
else:
    try:
        with open(
            os.path.expanduser(
                "~/Project/RPS/RomsPickingSystem/launcher/config/RPS_environment_settings/main.yaml"
            )
        ) as f:
            WS_NAME = list(yaml.safe_load(f).keys())[0]
    except:
        WS_NAME = ""


class OidMapsKeys:
    access_count = "access_count"
    order_ids = "order_ids"


class EventTypes(Enum):
    ERROR = 1
    WARNING = 2
    DECISION = 3
    INFO = 4
    DEBUG = 5


class ClassNames:
    PickDetectorDepth = "PickDetectorDepth"
    PlaceDetectorDepth = "PlaceDetectorDepth"
    SkuSizeChecker = "SkuSizeChecker"
    SizeChecker3D = "SizeChecker3D"
    PickDetectorNode = "PickDetectorNode"
    SkuSizeCheckerNode = "SkuSizeCheckerNode"
    SizeChecker3DNode = "SizeChecker3DNode"
    PlaceDetectorNode = "PlaceDetectorNode"
    OrderManagerNode = "OrderManagerNode"
    ActionDetectPick = "ActionDetectPick"
    ActionManagerNode = "ActionManagerNode"
    PcdProjector = "PcdProjector"
    PointMapProjector = "PointMapProjector"
    PcdProjectorKeys = "PcdProjectorKeys"
    ManipulatorNode = "ManipulatorNode"
    GraspCheckNode = "GraspCheckNode"
    # ## Actions
    # ActionCapture = "ActionCapture"
    # ActionDetectPick = "ActionDetectPick"
    # ActionSequence = "ActionSequence"
    # ActionCheckPlace = "ActionCheckPlace"
    # ActionPlanPath = "ActionPlanPath"
    # ActionTracePath = "ActionTracePath"
    # ActionGrasp = "ActionGrasp"
    # ActionRelease = "ActionRelease"
    # ActionWait = "ActionWait"
    # ActionCheckSKU = "ActionCheckSKU"
    # ActionDetectPlace = "ActionDetectPlace"


class ActionNames:
    capture = "capture"
    sequence = "sequence"
    release = "release"
    grasp = "grasp"

    detectpick = "detectpick"
    detectplace = "detectplace"

    planpick = "planpick"
    planplace = "planplace"

    checkplace = "checkplace"
    checksku = "checksku"

    topick = "topick"
    frompick = "frompick"
    toplace = "toplace"
    fromplace = "fromplace"


class StepNames:
    imagingPick = "imagingPick"
    detectingPick = "detectingPick"
    movingToPick = "movingToPick"
    imagingPlace = "imagingPlace"
    checkingPlace = "checkingPlace"
    picking = "picking"
    movingFromPick = "movingFromPick"
    movingToSizeCheck = "movingToSizeCheck"
    imagingSize = "imagingSize"
    movingToPlace = "movingToPlace"
    sizeChecking = "sizeChecking"
    detectingPlace = "detectingPlace"
    placing = "placing"


class ClassGroups:
    Pickd = ["PickDetectorNode", "PickDetectorDepth"]
    SizeCheck = ["SkuSizeCheckerNode", "SkuSizeChecker"]
    Placed = ["PlaceDetectorNode", "PlaceDetectorDepth"]


class NoKeyHandler(type):
    def __getattr__(self, name):
        if hasattr(self.__class__, name):
            return name
        return None


class GeneralKeys(metaclass=NoKeyHandler):
    ws_name = "ws_name"
    timestamp = "timestamp"
    lineno = "lineno"
    file_name = "file_name"
    data = "data"
    metadata = "metadata"
    order_id = "order_id"
    db_id = "_id"
    data_dir = "data_dir"
    oid_dir = "oid_dir"
    git_sha = "git_sha"
    git_branch = "git_branch"
    env_git_sha = "env_git_sha"
    env_git_branch = "env_git_branch"
    order_line = "order_line"

    collection = "collection"
    key = "key"
    value = "value"

    events = "events"
    brief = "brief"
    event_type = "event_type"
    aux = "aux"

    cls_name = "cls_name"
    method_name = "method_name"

    caller_info = "caller_info"
    safie_urls = "safie_urls"

    jan: JAN


class LogMethodNames(metaclass=NoKeyHandler):
    new_entry = "new_entry"
    insert = "insert"
    append = "append"
    backup = "backup"
    remove = "remove"


EventsFields = [
    GeneralKeys.brief,
    GeneralKeys.event_type,
    GeneralKeys.timestamp,
    GeneralKeys.lineno,
    GeneralKeys.cls_name,
    GeneralKeys.method_name,
    "fname",
]


class TargetDetectorKeys(metaclass=NoKeyHandler):
    visible = "visible"
    depth = "depth"
    kwargs = "kwargs"
    visible_extra = "visible_extra"
    depth_extra = "depth_extra"
    
    areaid = "areaid"
    area = "area"
    area_corners = "area_corners"

    registered_size = "registered_size"
    registered_weight = "registered_weight"
    pad_config = "pad_config"
    retry_another = "retry_another"
    group = "group"
    tote_height = "tote_height"
    grasp_target = "grasp_target"
    crop_ratio_short = "crop_ratio_short"
    crop_ratio_long = "crop_ratio_long"
    force_middle_picking = "force_middle_picking"

    return_items = "return_items"
    config_file = "config_file"
    detect_kwargs = "detect_kwargs"

    tote_mask = "tote_mask"
    tote_points = "tote_points"
    tote_combined_mask = "tote_combined_mask"
    tote_registered_mask = "tote_registered_mask"
    tote_detected_mask = "tote_detected_mask"
    tote_translation_mm = "tote_translation_mm"
    tote_rot_deg = "tote_rot_deg"
    jan = JAN

    areas_2D = "areas_2D"
    areas_3D = "areas_3D"

    rc_t = "rc_t"
    rc_R = "rc_R"
    dv_t = "dv_t"
    dv_R = "dv_R"


class PickDetectorDepthKeys(TargetDetectorKeys):
    eulers = "eulers"
    aligned_depth = "aligned_depth"
    trimmed_depth = "trimmed_depth"
    bb_info = "bb_info"
    bb_info_sam = "bb_info_sam"
    bb_valid = "bb_valid"
    segms_valid = "segms_valid"
    bb_ordered = "bb_ordered"
    a_obj = "a_obj"
    segms = "segms"
    segms = "segms_sam"
    vac_areas = "vac_areas"
    proj_img_mm_crop = "proj_img_mm_crop"
    cnn_image = "cnn_image"
    ai_image = "ai_image"
    result_image = "result_image"
    force_vertical_approach = "force_vertical_approach"
    image_compare_mask = "image_compare_mask"

class PlaceDetectorDepthKeys(TargetDetectorKeys):
    check_kwargs = "check_kwargs"
    trimmed_depth = "trimmed_depth"
    bb_info = "bb_info"
    bb_valid_inds = "bb_valid_inds"
    mean_height = "mean_height"
    debug_image = "debug_image"

    size_mm_used = "size_mm_used"
    sku_size_px_used = "sku_size_px_used"

    place_map = "place_map"
    place_point = "place_point"

    tcp_on_sku_t = "tcp_on_sku_t"
    tcp_on_sku_R = "tcp_on_sku_R"


class SkuSizeCheckerKeys(metaclass=NoKeyHandler):
    visible = "visible"
    depth = "depth"
    trimmed_depth = "trimmed_depth"
    trimmed_visible = "trimmed_visible"
    areaid = "areaid"

    depth_wo_noise = "depth_wo_noise"
    return_items = "return_items"
    registered_size = "registered_size"

    roi_left_top = "roi_left_top"
    roi_right_bottom = "roi_right_bottom"

    out_info = "out_info"
    seg_output = "seg_output"

    bin_seg = "bin_seg"
    bin_depth = "bin_depth"
    bin_all = "bin_all"
    binary_side_depth = "binary_side_depth"
    side_points = "side_points"

    config_file = "config_file"
    proj_img = "proj_img"

    rc_t = "rc_t"
    rc_R = "rc_R"
    dv_t = "dv_t"
    dv_R = "dv_R"
    checksku_t = "checksku_t"
    checksku_R = "checksku_R"


class SizeChecker3DKeys(metaclass=NoKeyHandler):
    return_items = "return_items"


class PointMapProjectorKeys(metaclass=NoKeyHandler):
    td_pcd = "td_pcd"
    td_pcd_inliers = "td_pcd_inliers"
    td_plane_model = "td_plane_model"
class PcdProjectorKeys(metaclass=NoKeyHandler):
    td_pcd = "td_pcd"
    td_pcd_inliers = "td_pcd_inliers"
    td_plane_model = "td_plane_model"

class PointMapProjectorKeys(metaclass=NoKeyHandler):
    td_pcd = "td_pcd"
    td_pcd_inliers = "td_pcd_inliers"
    td_plane_model = "td_plane_model"

class ActionKeys(metaclass=NoKeyHandler):
    action_start = "action_start"
    action_results = "action_results"


class PickDetectorNodeKeys(metaclass=NoKeyHandler):
    msg_dict = "msg_dict"
    jan = JAN
    res = "res"
    rc = "rc"
    result_dict = "result_dict"


class SkuSizeCheckerNodeKeys(metaclass=NoKeyHandler):
    action_id = "id"
    msg_dict = "msg_dict"
    result_dict = "result_dict"


class SkuSizeChecker3DNodeKeys(metaclass=NoKeyHandler):
    action_id = "id"
    msg_dict = "msg_dict"


class PlaceDetectorNodeKeys(metaclass=NoKeyHandler):
    msg_dict = "msg_dict"
    res = "res"
    rc = "rc"
    result_dict = "result_dict"


class OrderManagerNodeKeys(metaclass=NoKeyHandler):
    publisher = "publisher"
    result_code = "result_code"
    sort_code = "sort_code"
    desc = "desc"
    barcode_pick = "barcode_pick"
    barcode_place = "barcode_place"
    jan = JAN


class ActionManagerNodeKeys(metaclass=NoKeyHandler):
    msg = "msg"


class OrderStepKeys(metaclass=NoKeyHandler):
    step_logs = "step_logs"


class MemoryKeys(metaclass=NoKeyHandler):
    jan = "jan"
    data = "data"

    entries = "entries"
    value = "value"
    timestamp = "timestamp"
    git_sha = "git_sha"
    git_branch = "git_branch"
    successful = "successful"
    picking_status = "picking_status"

    stats = "stats"
    avg = "avg"
    stdev = "stdev"
    cv = "cv"
    counter = "counter"

    default = "default"
    stable = "stable"


class StatMode(metaclass=NoKeyHandler):
    values = "values"
    mean = "mean"
    median = "median"
    stdev = "stdev"
    coef_var = "coef_var"
    count = "count"


class StatDurations(metaclass=NoKeyHandler):
    total = "total"
    aggregated = "aggregated"
    end2end = "end2end"


class ErrorCategoriesNames(metaclass=NoKeyHandler):
    no_more_targets = "no_more_targets"
    pick_failure = "pick_failure"
    collision = "collision"
    drop = "drop"
    destination_full = "destination_full"
    system_failure = "system_failure"
    previous_failure = "previous_failure"
    unknown = "unknown"


class ActionCategories(metaclass=NoKeyHandler):
    cameras = "cameras"
    algorithm = "algorithm"
    robotics = "robotics"
    other = "other"


action_mapping = {
    # --- cameras
    "capture": ActionCategories.cameras,
    # --- algorithm
    "checkplace": ActionCategories.algorithm,
    "detectpick": ActionCategories.algorithm,
    "planpick": ActionCategories.algorithm,
    "checksku": ActionCategories.algorithm,
    "detectplace": ActionCategories.algorithm,
    "planplace": ActionCategories.algorithm,
    # --- robotics
    "sequence": ActionCategories.robotics,
    "topick": ActionCategories.robotics,
    "grasp": ActionCategories.robotics,
    "frompick": ActionCategories.robotics,
    "toplace": ActionCategories.robotics,
    "release": ActionCategories.robotics,
    "fromplace": ActionCategories.robotics,
    "move_robot": ActionCategories.robotics,
    "weight": ActionCategories.robotics,
}

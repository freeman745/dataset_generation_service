import os
from datetime import datetime
from ..constants import WS_NAME


def get_timestamp():
    return datetime.now()


def get_oid_dir(data_dir: str, collection: str, order_id: str):
    datestamp = datetime.now().strftime("%Y%m%d")
    return os.path.join(
        data_dir,
        WS_NAME,
        collection,
        datestamp,
        str(order_id),
    )


def is_oid_valid(order_id):
    if not isinstance(order_id, str):
        return False
    if len(order_id) <= 1:
        return False
    if order_id.lower() == "none":
        return False
    return True


def get_date_id():
    now = datetime.now()
    time_oid = f"dateID{now.strftime('%Y%m')}_{now.strftime('%d')}"
    return time_oid

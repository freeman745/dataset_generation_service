import os
import sys
from mongodb.constants import ITEM_CONFIG_PATH
from mongodb import MongoDBIFBase

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError(
            f"Run: python3 {sys.argv[0]} <collection_name>, <filepath> [<db_cfg>]"
        )
    collection_name, filepath = sys.argv[1], sys.argv[2]
    if len(sys.argv) == 4:
        cfg_yaml = sys.argv[3]
    else:
        cfg_yaml = ITEM_CONFIG_PATH
    if not isinstance(collection_name, str):
        raise TypeError(
            f"<collection_name> must be a str. {collection_name} was passed"
        )
    if not (
        isinstance(filepath, str)
        and os.path.isfile(filepath)
        and filepath.endswith(".json")
    ):
        raise ValueError(f"<filepath> must be a valid .json file path")
    mongodb_base = MongoDBIFBase()
    mongodb_base._configure(cfg_yaml)
    # if not mongodb_base.get_status():
    #     raise RuntimeError("Connection with the DB could not be established.")
    rec = mongodb_base.import_one(collection_name, filepath)
    if rec:
        print(f"Successfully imported Item to {collection_name}")
    else:
        print(f"Could not import Item from {filepath}")

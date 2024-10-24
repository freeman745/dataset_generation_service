# MongoDB Package Documentation

## Short description
The `mongodb` package is used for logging and accessing messages, events and files in a
structured way. The messages are logged using the `MongoDBLogger` (`mongodb/mongodb_logger.py`) objects. The messages are stored either as files or as entries to a MongoDB database.

Each pick-place circle of one item is logged as a separate document in the database.
- To access the item information one can use the `MongoDBItem` (`mongodb/mongodb_item.py`) object.
- To access the statistics of many picks (based on various filters) one can use the `MongoDBStats` (`mongodb/mongodb_stats.py`) object.

## Logging mechanism
In order to create a new entry (pick & place of an item) the `new_entry()` method of a `MongoDBLogger` object must be called with the `order_id` argument. All the entries after that call will be added to the document of this item, until the `new_entry()` method is called again.

### Types of logs
There are 3 types of logs that are supported by the `MongoDBLogger` class:
- `entries`: pairs of `key`, `value` that can be any python object able to be encoded by `bson` (e.g. numpy arrays should first be transformed to lists).
- `events`: a string depicting a specific event at the algorithm, with a specific log level.
- `arrays`: a numpy array, or list of numpy arrays. These arrays will be stored as files instead of entries in the database.

### Modes of logging
The `entry` and `array` types can be logged in one of two modes:
- `insert`: create a new entry in the DB/file. If already exists, replace it.
- `append`: append the entry to the list of entries for this key, or create a new file under the same directory.

## Message consumption
When the messages are created using the `MongoDBLogger` methods, they are broadcasted via a REDIS client to a specific topic. The `MongoDBConsumer` object created by a systemctl service is listening to the topic, receives the messages, and communicates with the MongoDB instance to store them to the database. Both REDIS and MongoDB run as separate containers based on the current architecture.

## Setup
To run the logger we need:
The MongoDB container
- The `mongo` service (see `docker-compose.yaml` of RPS).
- The `redis` service (see `docker-compose.yaml` of RPS).
- The `mongodb_consumer` service (see `/etc/systemd/system/mongodb_consumer.service`) from `./mongodb/scripts/mongodb_consumer.service`.
- The directory to which the numpy array files will be stored (`data_dir_path` from `rps_logs.yaml`) with read and write permissions to all users.
  - `data_dir_path`= `/data/db_data/` 

## RPS Memory
Under `mongodb/mongodb_memory.py` there is the `MongoDBMemory` class used for logging, reading, changing, and deleting elements from the memory database.

There is the `rps_memory` database in MongoDB with the `sku_info` collection. For each jan code (SKU) there is one document. Each document has one entry per feature (e.g. size). For each entry, there is a list with the logged values from each pick.

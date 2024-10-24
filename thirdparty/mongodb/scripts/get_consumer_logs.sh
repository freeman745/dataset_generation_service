#!/bin/bash

MONGODB_DIR="$(dirname "$(dirname "$(realpath "$0")")")"
tail -f $MONGODB_DIR/mongodb_consumer.log
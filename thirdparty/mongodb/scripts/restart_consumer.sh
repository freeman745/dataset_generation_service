#!/bin/bash

# Kill the existing session (if it exists)
tmux kill-session -t mongodb_consumer 2>/dev/null

# Check if --debug is passed as an argument
DEBUG_ARG=""
if [[ "$1" == "--debug" ]]; then
    DEBUG_ARG=" --debug"
fi

MONGODB_DIR="$(dirname "$(dirname "$(realpath "$0")")")"
SCRIPT="$MONGODB_DIR/mongodb/mongodb_consumer.py\"$DEBUG_ARG\""

# Start the session again
unset TMUX && tmux new-session -d -s mongodb_consumer \
"while true; do \
    python3 -u \"$SCRIPT\" \
    > $MONGODB_DIR/mongodb_consumer.log 2>&1; \
    sleep 5; \
done"

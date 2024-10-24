#!/bin/bash

# Kill the existing session (if it exists)
tmux kill-session -t mongodb_consumer 2>/dev/null
#!/bin/sh

# Get the hostname from the environment variable
host=$(eval $HOSTNAME_COMMAND)

# Start the celery worker
celery -A celeryconf worker --loglevel=info --concurrency 1 --hostname=worker@%h --hostname="${host}" -E

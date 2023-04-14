#!/bin/bash

set -o errexit
set -o nounset

cd /app

worker_ready() {
    celery -A celeryconf inspect ping
}

until worker_ready; do
  >&2 echo 'Celery workers not available'
  sleep 1
done
>&2 echo 'Celery workers is available'

echo "Starting flower"
celery -A celeryconf  \
    --broker="${REDIS_URL}/0" \
    flower

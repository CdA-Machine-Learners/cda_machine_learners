from celery import Celery
from os import environ

REDIS_URL = environ.get('REDIS_URL', 'redis://redis:6379')
celery_app = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)
celery_app.autodiscover_tasks(['tasks'], force=True)

@celery_app.task
def divide(x, y):
    import time
    time.sleep(5)
    return x / y

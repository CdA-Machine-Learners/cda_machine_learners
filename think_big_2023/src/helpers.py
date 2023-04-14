import asyncio
import sys
from bot import bot
from celery.exceptions import TaskRevokedError

def export(func):
    """
    Use a snippit to avoid retyping function/class names.
    Automatically adds the function to the module's __all__ list.
    This allows the function to be imported with `from module import *`.
    """
    mod = sys.modules[func.__module__]
    if hasattr(mod, '__all__'):
        name = func.__name__
        all_ = mod.__all__
        if name not in all_:
            all_.append(name)
    else:
        mod.__all__ = [func.__name__]


def add_async_command(func):
    """Decorator to add a synchronous command to the Discord bot and export it to the module."""
    export(func)
    decorated_func = bot.hybrid_command()(func)
    return decorated_func


class TaskFailedError(Exception):
    """Custom exception class for failed tasks."""
    pass


async def await_task(task, interval=3, max_wait_time=900):
    """Automatically handle simple deferred tasks."""
    i = 0
    while not task.ready():
        await asyncio.sleep(interval)
        i += interval
        if i > max_wait_time:
            task.revoke()
            print('Task took too long')
            raise TaskRevokedError("Task was revoked due to exceeding max_wait_time")

    result = task.get()
    if task.status == 'FAILURE':
        raise TaskFailedError("Task failed")

    return result


async def process_deferred_task(ctx, task, interval=3):
    """Automatically handle simple deferred tasks."""
    await ctx.defer()
    try:
        result = await await_task(task, interval)
    except TaskRevokedError:
        await ctx.send('Task timed out')
        return
    except TaskFailedError:
        await ctx.send('Task failed')
        return
    await ctx.send(result)

import random
from time import sleep

from helpers import add_async_command, process_deferred_task
from celeryconf import celery_app


@celery_app.task(bind=True)
def delay_echo(self, message):
    sleep(10)
    return f'{self.request.hostname} says "{message}"'


@add_async_command
async def echo(ctx, message: str):
    """Replies with your message."""
    await process_deferred_task(ctx, delay_echo.delay(message))


@add_async_command
async def roll(ctx, dice: str):
    """Rolls a die in NdN format."""
    try:
        rolls, limit = map(int, dice.split('d'))
    except Exception:
        await ctx.send('Format has to be in NdN!')
        return

    result = ', '.join(str(random.randint(1, limit)) for r in range(rolls))
    await ctx.send(result)

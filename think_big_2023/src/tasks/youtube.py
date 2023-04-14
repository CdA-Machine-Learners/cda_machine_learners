import asyncio

from celery.exceptions import TaskRevokedError

from apps.youtube_summarizer import YoutubeSummarizer
from helpers import add_async_command, await_task, TaskFailedError
from celeryconf import celery_app


# Split text into less than 2000 characters to avoid Discord's 2000 character limit
def split_text(text, max_len=1990):
    if len(text) <= max_len:
        return [text]

    parts = []
    while len(text) > max_len:
        split_index = max_len
        while text[split_index] != ' ' and split_index > 0:
            split_index -= 1

        if split_index == 0:
            raise ValueError("No whitespace found to split the text.")

        parts.append(text[:split_index])
        text = text[split_index:].strip()

    parts.append(text)
    return parts


@celery_app.task
def youtube_summarizer_task(url: str):
    summarizer = YoutubeSummarizer(url)
    output = summarizer.summarize()
    return output


@add_async_command
async def youtube(ctx, url: str):
    """
    Given a YouTube url, the model will return a summary of the video.
    """
    ctx.defer()
    try:
        output = await await_task(youtube_summarizer_task.delay(url))
    except TaskRevokedError:
        await ctx.send('Task timed out')
        return
    except TaskFailedError:
        await ctx.send('Task failed')
        return

    formatted_output = f'''
**[Youtube Summarizer]**
{ctx.author.mention}
> ```/youtube url:{url}```
'''
    if len(output) + len(formatted_output) < 1988:
        await ctx.send(f'{formatted_output}\n```{output}```')
    else:
        await ctx.send(formatted_output)
        for part in split_text(output):
            await ctx.send(f'```{part}```')

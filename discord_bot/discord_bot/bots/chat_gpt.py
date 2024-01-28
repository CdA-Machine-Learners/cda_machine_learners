'''

Chat with chat gpt

'''

from discord_bot.common import get_nested, mk_logger
import argparse
import logging
import asyncio

import openai
from openai import AsyncOpenAI  # Import AsyncOpenAI for asynchronous operations

log = mk_logger('example', logging.DEBUG)

COMPLETION_ENGINES = []

CHAT_ENGINES = [
    "gpt-3.5-turbo",
    "gpt-4.5-turbo",
]


async def openai_autocomplete(client, engine, text, max_length):
    ''' Stream responses from OpenAI's API as a generator using an Async client. '''
    if engine in COMPLETION_ENGINES:
        response = await client.completions.create(
            model=engine,
            prompt=text,
            max_tokens=max_length,
            stream=True
        )
        for message in response:
            generated_text = message.choices[0].text
            yield generated_text
    elif engine in CHAT_ENGINES:
        response = await client.chat.completions.create(
            model=engine,
            messages=[{"role": "user", "content": text}],
            stream=True,
            max_tokens=max_length
        )
        async for message in response:
            delta = message.choices[0].delta
            if delta.content is not None:
                print(delta.content)
                generated_text = delta.content
                yield generated_text

async def openai_stream_fn(client, proc, prompt, engine, max_length):
    BATCH = 20
    try:
        running_text = f'**Chat prompt:** {prompt}\n\n'
        i = 0
        async for new_text in openai_autocomplete(client, engine, prompt, max_length):
            i += 1
            if len(new_text) == 0:
                continue
            running_text += new_text
            if i % BATCH == 0:
                await asyncio.sleep(1.0)
                await proc.edit(content=running_text)
        if i >= max_length - 1:
            running_text += '... truncated'
        await proc.edit(content=running_text)
    except Exception as e:
        log.error(f'Error: OpenAI, {e}')

def configure(config_yaml):
    parser = argparse.ArgumentParser()
    # These get picked up as `config` in `initialize`
    parser.add_argument('--engine', default=get_nested(config_yaml, ['chat_gpt', 'engine']))
    parser.add_argument('--max_length', default=get_nested(config_yaml, ['chat_gpt', 'max_length']))
    parser.add_argument('--api_key', default=get_nested(config_yaml, ['chat_gpt', 'api_key']))

    # bc this is only concerned with this module's params, do not error if
    # extra params are sent via cli.
    args, _ = parser.parse_known_args()
    return args

def initialize(args, server):
    log.info('Initializing ChatGPT Bot')
    max_length = args.max_length
    engine = args.engine

    client = AsyncOpenAI(
      api_key=args.api_key,
    )

    @server.hybrid_command(name="chat", description="Chat with ChatGPT")
    async def chat(ctx, prompt: str):
        proc = await ctx.reply("Processing...")
        await openai_stream_fn(client, proc, prompt, engine, max_length)

    return server

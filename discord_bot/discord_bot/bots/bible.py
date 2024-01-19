'''



'''

from discord_bot.common import get_nested, mk_logger
import argparse
import logging
import asyncio
# import aiohttp
import httpx
import json
import traceback

log = mk_logger('example', logging.DEBUG)



async def autocomplete(host, text, model, max_length):
    ''' Stream responses from AI server as a generator using an Async HTTP client. '''
    payload = {
        "prompt": text,
        "max_tokens": max_length,
        "stream": True  # This is part of the payload, not the request method
    }
    headers = {"User-Agent": "Async Client"}
    async with httpx.AsyncClient() as client:
        response = await client.post(host, json=payload, headers=headers)

        buffer = b''
        async for chunk in response.aiter_bytes():
            buffer += chunk
            while b'\x00' in buffer:
                message, buffer = buffer.split(b'\x00', 1)
                try:
                    if message:
                        data = json.loads(message.decode("utf-8"))
                        output = data.get("text", [])
                        if output:
                            for text in output:
                                yield text
                except json.JSONDecodeError as e:
                    print(f"JSON decoding failed: {e}")
                    # Handling incomplete or corrupted chunks
                    continue

DISCORD_MAX_LENGTH = 1950
async def stream_fn(host, proc, prompt, model, max_length):
    BATCH = 20
    try:
        i = 0
        running_text = ''
        async for running_text in autocomplete(host, prompt, model, max_length):
            i += 1
            if len(running_text) == 0:
                continue
            if i % BATCH == 0:
                # await asyncio.sleep(1.0)
                await proc.edit(content=running_text)
        if i >= DISCORD_MAX_LENGTH - 1:
            running_text += '... truncated'
        await proc.edit(content=running_text)
    except Exception as e:
        print("Stack trace:", traceback.format_exc())
        print(f'Error: AI Server, {e}')

def configure(config_yaml):
    parser = argparse.ArgumentParser()
    # These get picked up as `config` in `initialize`
    parser.add_argument('--host', default=get_nested(config_yaml, ['bible', 'host']))
    parser.add_argument('--model', default=get_nested(config_yaml, ['bible', 'model']))
    parser.add_argument('--max_length', default=get_nested(config_yaml, ['bible', 'max_length']))


    # bc this is only concerned with this module's params, do not error if
    # extra params are sent via cli.
    args, _ = parser.parse_known_args()
    return args

def initialize(args, server):
    log.info('Initializing Bible Bot')
    max_length = args.max_length
    model = args.model
    host = args.host

    @server.hybrid_command(name="bible", description="Chat with the Bible")
    async def chat(ctx, prompt: str):
        proc = await ctx.reply("Processing...")
        await stream_fn(host, proc, prompt, model, max_length)

    return server

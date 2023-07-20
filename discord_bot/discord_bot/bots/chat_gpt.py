'''

Chat with chat gpt

'''

from discord_bot.common import get_nested, mk_logger
import argparse
import logging
import openai

log = mk_logger('example', logging.DEBUG)

ENGINE = 'gpt-3.5-turbo'
MAX_LENGTH = 10

COMPLETION_ENGINES = [
    "text-davinci-003",
    "text-davinci-002",
    "ada",
    "babbage",
    "curie",
    "davinci",
]

CHAT_ENGINES = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0613",
    "gpt-4",
]


def openai_autocomplete(engine, text, max_length):
    ''' Stream responses from OpenAI's API as a generator. '''
    if engine in COMPLETION_ENGINES:
        response = openai.Completion.create(
          engine=engine,
          prompt=text,
          max_tokens=max_length,
          stream=True
        )
        for message in response:
            generated_text = message['choices'][0]['text']
            yield generated_text
    elif engine in CHAT_ENGINES:
        response = openai.ChatCompletion.create(
          model=engine,
          messages=[{"role": "user", "content": text}],
          stream=True
        )
        for message in response:
            # different json structure than completion endpoint
            delta = message['choices'][0]['delta']
            if 'content' in delta:
                generated_text = delta['content']
                yield generated_text


async def openai_stream_fn(proc, prompt, engine, max_length):
    try:
        # Stream the results to LSP Client
        running_text = ''
        for new_text in openai_autocomplete(engine, prompt, max_length):
            if len(new_text) == 0:
                continue
            running_text += new_text
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
    log.info('Initializing Example Bot')
    max_length = args.max_length
    engine = args.engine
    openai.api_key = args.api_key

    @server.hybrid_command(name="chat", description="Chat with ChatGPT 3.5")
    async def chat(ctx, prompt: str):
        proc = await ctx.reply("Processing...")
        await openai_stream_fn(proc, prompt, engine, max_length)

    return server

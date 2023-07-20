'''

A demonstration of a GPT-maintained state-loop in the form of D&D.

'''

from dataclasses import dataclass
from discord_bot.common import get_nested, mk_logger
import argparse
import logging
import openai
import asyncio
import re
import random

log = mk_logger('game', logging.DEBUG)


COMPLETION_ENGINES = [
    "text-davinci-003",
    # "text-davinci-002",
    # "ada",
    # "babbage",
    # "curie",
    # "davinci",
]

CHAT_ENGINES = [
    "gpt-3.5-turbo",
    # "gpt-3.5-turbo-0613",
    # "gpt-4",
]


def openai_autocomplete(engine, text, max_length):
    ''' NON-Streaming responses from OpenAI's API.'''
    if engine in COMPLETION_ENGINES:
        response = openai.Completion.create(
          engine=engine,
          prompt=text,
          max_tokens=max_length,
          stream=False
        )
        return response
    elif engine in CHAT_ENGINES:
        response = openai.ChatCompletion.create(
          model=engine,
          messages=[{"role": "user", "content": text}],
          stream=False
        )
        return response['choices'][0]['message']['content']


def find_tag(tag: str, doc_lines: [str]):
    ''' Find index of first element that contains `tag`. '''
    ix = 0
    for ix, line in enumerate(doc_lines):
        match = re.search(tag, line)
        if match:
            return ix, match.start(), match.end()
    return None


def find_block(start_tag, end_tag, doc):
    '''Fine the indices of a start/end-tagged block.'''
    if doc is None:
        return None, None
    doc_lines = doc.split('\n')
    s = find_tag(start_tag, doc_lines)
    e = find_tag(end_tag, doc_lines)
    return s, e


def extract_block(start, end, doc):
    '''Extract block of text between `start` and `end` tag.'''
    if doc is None:
        return None
    doc_lines = doc.split('\n')
    if start is None or end is None:
        return None
    if start[0] > end[0] or (start[0] == end[0] and start[2] > end[1]):
        return None
    if start[0] == end[0]:
        return [doc_lines[start[0]][start[2]: end[1]]]
    else:
        block = [doc_lines[start[0]][start[2]:]]  # portion of start line
        block.extend(doc_lines[start[0]+1:end[0]])  # all of middle lines
        block.append(doc_lines[end[0]][:end[1]])  # portion of end line
        return '\n'.join(block)


def start_tag(x):
    return f'<{x}_TAG>'


def end_tag(x):
    return f'</{x}_TAG>'


def get_block(tag, doc):
    s1, s2 = find_block(start_tag(tag), end_tag(tag), doc)
    return extract_block(s1, s2, doc)


STATE = 'STATE'
NEW_STATE = 'NEW_STATE'
REQUEST = 'REQUEST'
RESPONSE = 'RESPONSE'
UPDATES_NEEDED = 'UPDATES_NEEDED'

def initial_state():
    ''' hide it here so it doesn't accidentally mutate. '''
    prefix = 'You will be a Dungeon Master, and you will keep notes via a natural language-based state machine. Keep notes on: items, players, quests, etc.'
    suffix = 'Remember, keep responses brief, invent interesting quests and obstacles, and make sure the state is always accurate and complete, written in proper yaml.'
    game_state =  '''
players:
  greybeard42:
    items:
    location:

quests:

obstacles:

enemies:
'''
    state = State(
        prefix=prefix,
        suffix=suffix,
        running_resp='',
        game_state=game_state
    )

    return state


def get_response(request, state, engine, max_length):
    nl = '\n\n'  # can't do newlines inside f-exprs
    prompt = f'''
{state.prefix + nl if state.prefix else ''}You must assume the role of a finite state machine, but using only natural language.

You will be given state, and a request.

You must return a response, and a new state.

Please format your response like:

{start_tag(RESPONSE)}
your response
{end_tag(RESPONSE)}

{start_tag(UPDATES_NEEDED)}
updates that you'll need to apply to the new state
{end_tag(UPDATES_NEEDED)}

{start_tag(NEW_STATE)}
the new state
{end_tag(NEW_STATE)}

Here is the current state:

{start_tag(STATE)}
{state.game_state}
{end_tag(STATE)}

Here is a transcript of your responses so far:
{state.running_resp}

Here is the current request:
    {request}{nl + state.suffix if state.suffix else ''}
'''.strip()

    return openai_autocomplete(engine, prompt, max_length)


@dataclass
class State:
    prefix: str
    suffix: str
    running_resp: str
    game_state: str


state = initial_state()


async def step_game(ctx, proc_reply, state, engine, max_length, author, prompt):
    prompt_author = f'{author}: {prompt}'
    x = get_response(prompt_author, state, engine, max_length)

    # Try extracting new_state
    new_game_state = get_block(NEW_STATE, x)
    if new_game_state is None:
        new_game_state = get_block(STATE, x)

    # Try extracting response
    resp = get_block(RESPONSE, x)
    if resp is None:
        resp = '<AI response invalid>'
        log.warn(f'INVALID RESPONSE: \n{x}')
        ctx.send(resp)
        return

    if new_game_state is not None and resp is not None:
        state.game_state = new_game_state
        state.running_resp = f'{state.running_resp.strip()}\n\n{resp.strip()}'
        log.debug(f'STATE: {state.game_state}')
        log.debug(f'STATE: {state.running_resp}')
        author_resp = f'**{author}**: {prompt.strip()}\n**response**: {resp.strip()}'
        await proc_reply.edit(content=author_resp)


def configure(config_yaml):
    parser = argparse.ArgumentParser()
    # These get picked up as `config` in `initialize`
    parser.add_argument('--engine', default=get_nested(config_yaml, ['game', 'engine']))
    parser.add_argument('--max_length', default=get_nested(config_yaml, ['game', 'max_length']))
    parser.add_argument('--api_key', default=get_nested(config_yaml, ['game', 'api_key']))

    # bc this is only concerned with this module's params, do not error if
    # extra params are sent via cli.
    args, _ = parser.parse_known_args()
    return args


def initialize(args, server):
    log.info('Initializing Game Bot')
    max_length = args.max_length
    engine = args.engine
    openai.api_key = args.api_key

    @server.hybrid_command(name="game", description="You're now in the game. What will you do?")
    async def game(ctx, prompt: str):
        author = ctx.author.name
        proc_reply = await ctx.reply("**{author}**: {prompt.strip()}.  *Processing...*")
        await step_game(ctx, proc_reply, state, engine, max_length, author, prompt)

    @server.hybrid_command(name="restart_game", description="WARNING: this resets the game for everyone.")
    async def restart_game(ctx, password):
        if password.lower().strip() == '4321pass':
            state = initial_state()
            await ctx.reply("State has been reset.")
        else:
            author = ctx.author.name
            heckles = [
                f'{author} is naughty and tried to reset the game.',
                f'{author}, how dare you try and reset the game.',
                f'{author} is a 1337 haxor.',
                f'WARNING a wild {author} is on the loose, resting games and such!',
                f'Oh hi {author}. I see youd like to reset the game.',
                f'What did the cake say to the {author}? **Donut** reset the game!',
            ]
            await ctx.reply(random.choice(heckles))

    return server

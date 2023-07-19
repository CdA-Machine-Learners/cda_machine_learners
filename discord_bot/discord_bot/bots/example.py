'''

Example bot, demonstrates how to add a bot.

'''

from discord_bot.common import get_nested, mk_logger
import argparse
import logging

log = mk_logger('example', logging.DEBUG)

def configure(config_yaml):
    parser = argparse.ArgumentParser()
    # These get picked up as `config` in `initialize`
    parser.add_argument('--greeting', default=get_nested(config_yaml, ['example', 'greeting']))

    # bc this is only concerned with this module's params, do not error if
    # extra params are sent via cli.
    args, _ = parser.parse_known_args()
    return args

def initialize(args, server):
    log.info('Initializing Example Bot')
    # Config

    # Command: hello
    @server.hybrid_command(name="hello", description="Say hi to user")
    async def hello(ctx, prompt: str):
        proc = await ctx.reply("Processing...")
        user_name = ctx.author.name
        welcome_message = args.greeting.format(name=user_name)
        await ctx.send(welcome_message)

    return server

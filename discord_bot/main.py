from discord.ext import commands
import discord
from discord_bot import bot
from discord_bot import config
import importlib
from discord_bot.common import mk_logger
import logging
import sys

logging.basicConfig(
    # filename='log_file.log', # TODO: feature loggers still seem to report on stdout
    stream=sys.stdout,
    level=logging.DEBUG,
)

log = mk_logger('main', logging.DEBUG)

def load_module(module_name, config_yaml, server):
    ''' Useful for loading just the modules referenced in the config. '''
    module = importlib.import_module(module_name)
    log.info(f'Loading: {module_name}')
    print(f'Loading: {module_name}')

    if hasattr(module, 'configure'):
        log.info(f'Configuring: {module_name}')
        args = module.configure(config_yaml)

    if hasattr(module, 'initialize'):
        log.info(f'Initializing: {module_name}')
        server = module.initialize(args, server)
    return server

description = '''
A panoply of cool AI Tools.
'''.strip()

def main():
    args, config_yaml, parser = config.get_args()

    intents = discord.Intents.default()
    intents.members = True
    intents.message_content = True
    server = commands.Bot(command_prefix='?', description=description, intents=intents)

    for module_name in args.modules:
        server = load_module(module_name, config_yaml, server)

    # queue = asyncio.Queue()

    # # Command: read_to_me qr gen
    # @server.hybrid_command(name="chat", description="ChatGPT 3.5")
    # async def chat(ctx, speaker, *, prompt: str):
    #     proc = await ctx.reply("Processing...")
    #     await queue.put(QueueMsg(ctx, proc, speaker, prompt))

    # Runs once the server has logged in successfully
    @server.event
    async def on_ready():
        log.info(f'Logged in as {server.user.name} (ID: {server.user.id})')
        log.info('Syncing commands for tab completion...')
        await server.tree.sync()

    # Command: hello
    @server.hybrid_command(name="health",
                           description="Check health of the bot.")
    async def health(ctx):
        await ctx.send(f'Bot is functioning, and running bots: {args.modules}')

    # This runs the server and blocks until the server process is closed
    server.run(args.discord_token)

if __name__ == '__main__':
    main()

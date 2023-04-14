import discord
from discord.ext import commands
from dotenv import load_dotenv
import os
from session import get_session, close_session

description = """
An example bot to showcase the discord.ext.commands extension module.
There are a number of utility commands being showcased here.
"""

load_dotenv('../.env')
TOKEN = os.getenv('DISCORD_BOT_TOKEN')

intents = discord.Intents.default()
intents.members = True
intents.message_content = True
bot = commands.Bot(command_prefix='?', description=description, intents=intents)


@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('Syncing commands for tab completion...')
    await bot.tree.sync()
    bot.session = get_session() # to allow async requests
    print('done')
    print('------')


@bot.event
async def on_disconnect():
    print('Disconnected')
    await close_session()


'''

A Discord Bot

USAGE:
  OPENAI_API_KEY=$(cat .openai_token) TOKEN=$(cat .discord_token) python src/main.py

'''
from bot import bot, TOKEN

# This import registers the commands in the bot
from tasks import *


if __name__ == '__main__':
    bot.run(TOKEN)

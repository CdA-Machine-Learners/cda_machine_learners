from PIL import Image
from discord.ext import commands
import discord, asyncio, aiohttp, io, uuid, os, sys, re

sys.path.append(os.path.abspath('../'))
import settings, requests


def connect():
    # Create a bot instance
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix='/', intents=intents)

    # Setup our API
    publish_commands()

    return bot


def publish_commands():
    app_id = settings.DISCORD['BOT_ID']
    token = settings.DISCORD['BOT_TOKEN']

    print("Registering commands")

    # This is an example CHAT_INPUT or Slash Command, with a type of 1
    json = {
        "name": "read_to_me",
        "type": 1,
        "description": "Convert text to audio with inflections",
        "options": [
            {
                "name": "speaker",
                "description": "Pick a speaker",
                "type": 3,
                "required": True,
                "choices": [
                    {
                        "name": "Man",
                        "value": "man"
                    },
                    {
                        "name": "Woman",
                        "value": "woman"
                    },
                ]
            },
            {
                "name": "prompt",
                "description": "AI prompt",
                "type": 3,
                "required": True
            },
        ]
    }

    # For authorization, you can use either your bot token
    headers = {
        "Authorization": f"Bot {token}"
    }
    url = f"https://discord.com/api/v10/applications/{app_id}/commands"
    r = requests.post(url, headers=headers, json=json)
    print( r.status_code )


# Function to actually send the image
async def send_url_to_channel(ctx, bot, msg, url):
    filename = url.split('/')[-1]

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.read()
                fp = io.BytesIO(data)
                fp.seek(0)
                return await ctx.send(msg, file=discord.File(fp=fp, filename=filename))

    print(f"Error fetching image from URL: {url}")
    return None


# Function to actually send the image
async def send_image_to_channel(ctx, image: Image, url="", prompt=""):
    # Create a BytesIO object
    byte_stream = io.BytesIO()
    image.save(byte_stream, format='PNG')
    byte_stream.seek(0)

    # Build the response and send it
    file = discord.File(byte_stream, 'image.png')
    content = f'**Vanity QR Code**\n\nURL:\n> {url}\nPrompt:\n> {prompt}'
    return await ctx.reply(file=file, content=content)


# Function to actually send the image
async def send_audio_to_channel(ctx, audio: str, prompt=""):
    # Create a BytesIO object
    try:
        with open(audio, "rb") as byte_stream:
            byte_stream.seek(0)

            # Build the response and send it
            file = discord.File(byte_stream, 'audio.mp3')
            content = f'**Read To Me**\n\nPrompt:\n> {prompt[:64]}...'
            return await ctx.reply(file=file, content=content)

    # Errors...
    except:
        return await ctx.reply(f"Error reading audio file: {audio}")

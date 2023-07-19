'''




'''
from PIL import Image
from discord.ext import commands
import aiohttp
import asyncio
import discord
import importlib
import io
import os
import re
import requests
import subprocess
import sys
import uuid

class QueueMsg:
    def __init__(self, ctx, proc, speaker, prompt):
        self.ctx = ctx
        self.processing = proc
        self.speaker = speaker
        self.prompt = prompt


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
    except:
        return await ctx.reply(f"Error reading audio file: {audio}")


##################################################

# async def loop(bot: commands.Bot, queue: asyncio.Queue):
#     print('Starting main bot loop')
#     while True:
#         # Pull valid messages
#         if (msg := await queue.get()) is None:
#             continue

#         # Get my speaker
#         speaker_embeddings = spkr_e.get(msg.speaker, spkr_e['man'])

#         # Generate speech
#         print("Generating speech")
#         fn = audio.text_to_speech( processor, model, speaker_embeddings, vocoder, msg.prompt) #, update_status )

#         # Send the image to the user
#         await msg.processing.delete()
#         await send_audio_to_channel(msg.ctx, fn, msg.prompt )

#     # Do something with the image
#     queue.task_done()

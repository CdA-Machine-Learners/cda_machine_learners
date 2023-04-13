'''

TEST FRM JOSH

A Discord Bot

USAGE:
  TOKEN=$(cat .discord_token) python src/main.py

'''

import aiohttp
import discord
from discord.ext import commands
import openai
import os
import random
from transformers import pipeline

TOKEN = os.getenv('TOKEN')
if TOKEN is None:
    raise Exception('TOKEN env var must be set to a valid Discord token.')

openai.api_key = os.getenv('OPENAI_API_KEY')
if openai.api_key is None:
    raise Exception('OPENAI_API_KEY env var must be set to a valid openai api key.')

description = '''An example bot to showcase the discord.ext.commands extension
module.
There are a number of utility commands being showcased here.'''

intents = discord.Intents.default()
intents.members = True
intents.message_content = True

# Commands can be prefixed by '?'
# Hybrid commands (only use these, really) can be prefixed by `/` and will be tab-completed in the web UI.
bot = commands.Bot(command_prefix='?', description=description, intents=intents)


@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('Syncing commands for tab completion...')
    await bot.tree.sync()
    bot.session = aiohttp.ClientSession() # to allow async requests
    print('done')
    print('------')

@bot.hybrid_command()
async def add(ctx, left: int, right: int):
    """Adds two numbers together."""
    await ctx.send(left + right)

@bot.hybrid_command()
async def roll(ctx, dice: str):
    """Rolls a dice in NdN format."""
    try:
        rolls, limit = map(int, dice.split('d'))
    except Exception:
        await ctx.send('Format has to be in NdN!')
        return

    result = ', '.join(str(random.randint(1, limit)) for r in range(rolls))
    await ctx.send(result)


##################################################
# Stable Diffusion

import json
import requests
import io
import base64
from PIL import Image

url = "http://127.0.0.1:7860"


@bot.hybrid_command()
@discord.ext.commands.guild_only() # don't respond on DMs
async def image(ctx, prompt: str):
    """
    Given text, the model will return a generated image.

    task: https://huggingface.co/tasks/text-to-image
    model: TODO
    size: TODO
    dataset: TODO
    source: TODO
    """

    # endpoints must respond in <3 sec, unless they defer first. This
    # shows in the UI as "thinking..."
    await ctx.defer()

    print(prompt)

    payload = {
        "prompt": prompt,
        "steps": 40,

        "width": 512,
        "height": 512,
        # "firstpass_width": 512,
        # "firstpass_height": 512,

        "negative_prompt": "blurry",
    }

    async with bot.session.post(url=f'{url}/sdapi/v1/txt2img', json=payload) as response:
        r = await response.json()

        for i in r['images']:
            image = io.BytesIO(base64.b64decode(i.split(",",1)[0]))
            file=discord.File(image, 'image.png')
            txt = f'`/image` {prompt}'
            await ctx.reply(file=file, content=txt)

##################################################
# Sentiment Analysis

@bot.hybrid_command()
@discord.ext.commands.guild_only()
async def sentiment_analysis(ctx, prompt: str):
    """
    Given text, the model will return a polarity (positive, negative,
    neutral) or a sentiment (happiness, anger).

    task: https://huggingface.co/tasks/text-classification
    model: distilbert-base-uncased-finetuned-sst-2-english
    size: 268 MB
    dataset: https://huggingface.co/datasets/sst2
    source: https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
    """
    await ctx.defer()
    model = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    await ctx.send(f'`[Text Classification]` {model(prompt)}')

##################################################
# Text Generation

@bot.hybrid_command()
@discord.ext.commands.guild_only()
async def text_generation(ctx, prompt: str):
    """
    Given text, the model will return generated text.

    task: https://huggingface.co/tasks/text-generation
    model: gpt2
    size: 548 MB
    dataset:
        - https://github.com/openai/gpt-2/blob/master/domains.txt
        - https://huggingface.co/datasets/openwebtext
    source: https://huggingface.co/gpt2
    """
    await ctx.defer()
    model = pipeline('text-generation', model='gpt2')
    await ctx.send(f'`[Text Generation]` {model(prompt)}')

##################################################
# ChatGPT4

import openai

@bot.hybrid_command()
@discord.ext.commands.guild_only()
async def gpt4_chat(ctx, prompt: str):
    """
    Given a chat conversation, the model will return a chat completion response.

    task: https://huggingface.co/tasks/conversational
    model: chatgpt4
    size: unknown
    dataset: unknown
    source: closed
    """
    await ctx.defer()
    completion = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[
            # The system message helps set the behavior of the assistant
            {'role': 'system', 'content': 'You are a very knowledgable entity.'},
            # The user messages help instruct the assistant
            {'role': 'user', 'content': prompt},
            # The assistant messages help store prior responses
            # (provides context or desired behavior)
            # {'role': 'assistant', 'content': 'TODO!'},
        ],
    )
    await ctx.send(f'`[Conversation GPT4]` {completion.choices[0].message}')


##################################################
# Run

bot.run(TOKEN)

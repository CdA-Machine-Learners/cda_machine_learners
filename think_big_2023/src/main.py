'''

A Discord Bot

USAGE:
  OPENAI_API_KEY=$(cat .openai_token) TOKEN=$(cat .discord_token) python src/main.py

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
    """Provide detailed text prompt for the image you want"""

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
async def sentiment(ctx, prompt: str):
    """Predict the polarity (positive, negative, neutral) or a sentiment (happiness, anger)."""
    await ctx.defer()
    model = pipeline('sentiment-analysis',
                     model='distilbert-base-uncased-finetuned-sst-2-english'
    )
    out = model(prompt)[0]
    fmt = f'''
**`[Sentiment Analysis]`** 
{prompt}
*Prediction:* {out['label']}
*Confidence:* {out['score']:>0.4f}
'''
    await ctx.send(fmt)

    
##################################################
# Text Continuation

@bot.hybrid_command()
@discord.ext.commands.guild_only()
async def continuation(ctx, prompt: str):
    """Given text, the model will predict how it might have continued."""
    await ctx.defer()
    model = pipeline('text-generation', model='gpt2')
    out = model(prompt) 
    fmt = f'''
**`[Text Continuation]`** 
*{prompt}* 
{out[0]['generated_text']}
'''
    await ctx.send(fmt)


##################################################
# ChatGPT4

import openai

@bot.hybrid_command()
@discord.ext.commands.guild_only()
async def chat(ctx, prompt: str):
    """Chat with a robot. Ask it for a poem, or historical fact, or a joke!"""
    await ctx.defer()
    completion = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
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
    out = f'''
**`[ChatGPT]`** 
*Q:* {prompt}
*A:* {completion.choices[0].message['content']}
'''
    await ctx.send(out)

    
##################################################
# Youtube Summarizer

from youtube_summarizer.summarizer import YoutubeSummarizer

# Will hold summaries for previously seen URLs.
cache = {}

@bot.hybrid_command()
@discord.ext.commands.guild_only()
async def youtube(ctx, youtube_url: str = 'https://www.youtube.com/watch?v=dC1-qgR7YO0'):
    """Provide a URL of a Youtube video you'd like summarized, or hit enter to accept the default.    """
    await ctx.defer()

    try:
        # Check cache
        if url in cache:
            out = cache[url]

        # Generate summary fresh
        else:
            yt = YoutubeSummarizer(url, debug=False)
            out = yt.summarize()
            cache[url] = out
            
        fmt = f'''
**`[Youtube Summarizer]`**
*URL:* {url}
{out}
'''
        await ctx.send(fmt)
    except Exception as e:
        await ctx.send(f'Exception: {str(e)}')


##################################################
# Run

bot.run(TOKEN)


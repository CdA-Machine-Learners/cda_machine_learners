import random

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch, sys
import soundfile as sf
import numpy as np
import subprocess

import nltk, sys

from PIL import Image
from core import audio
from discord.ext import commands
import discord, asyncio, aiohttp, io, uuid, os, sys, re

from discord_bot import comms

sys.path.append(os.path.abspath('../'))
import settings, requests, nltk


class QueueMsg:
    def __init__(self, ctx, proc, speaker, prompt):
        self.ctx = ctx
        self.processing = proc
        self.speaker = speaker
        self.prompt = prompt


async def process_request( bot: commands.Bot, queue: asyncio.Queue ):
    nltk.download( 'punkt')  # Download the necessary resources for sentence tokenization

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    # load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    # speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    man_spkr = np.load("cmu_us_ksp_arctic-wav-arctic_b0087.npy")
    #wom_spkr = np.load("cmu_us_bdl_arctic-wav-arctic_a0009.npy") Guys voice and he sucks!!!

    spkr_e = {
        "man": torch.tensor(man_spkr).unsqueeze(0),
        "woman": torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0),
    }

    def update_status( idx, total ):
        msg.processing.edit(content=f"Processing chunk {idx+1} of {total}")

    print("Starting processor")
    while True:
        # Pull valid messages
        if (msg := await queue.get()) is None:
            continue

        # Get my speaker
        speaker_embeddings = spkr_e.get(msg.speaker, spkr_e['man'])

        # Generate speech
        print("Generating speech")
        fn = audio.text_to_speech( processor, model, speaker_embeddings, vocoder, msg.prompt) #, update_status )

        # Send the image to the user
        await msg.processing.delete()
        await comms.send_audio_to_channel(msg.ctx, fn, msg.prompt )

    # Do something with the image
    queue.task_done()


def main():
    nltk.download( 'punkt')  # Download the necessary resources for sentence tokenization
    bot = comms.connect()
    queue = asyncio.Queue()

    # Command: read_to_me qr gen
    @bot.hybrid_command(name="read_to_me", description="Convert text to audio with inflections")
    async def read_to_me(ctx, speaker, *, prompt: str):
        proc = await ctx.reply("Processing...")
        await queue.put(QueueMsg(ctx, proc, speaker, prompt))

    # Runs once the bot has logged in successfully
    @bot.event
    async def on_ready():
        print(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
        print('------')
        asyncio.create_task(process_request(bot, queue))

    # This runs the bot and blocks until the bot process is closed
    bot.run(settings.DISCORD['BOT_TOKEN'])

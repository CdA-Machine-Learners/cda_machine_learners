import random

from core import qrcode, auto4, s3, post_proc, cda_discord

from PIL import Image
from discord.ext import commands
import discord, asyncio, webuiapi, aiohttp, io, uuid, os, sys, re, concurrent.futures

sys.path.append(os.path.abspath('../'))
import settings, requests


class QueueMsg:
    def __init__(self, ctx, proc, url, prompt):
        self.ctx = ctx
        self.prompt = prompt
        self.processing = proc

        if re.search(r'^https?://', url) is None:
            url = f'https://{url}'
        self.url = url


def generate_qr_code( api: webuiapi, url: str, prompt: str ):
    # Render the ideal QR code
    qr_code = qrcode.render(url)
    qr_code_1px = qrcode.render(url, box_size=1, border=0)
    qr_code.save('/tmp/qr.png')
    # qrcode.is_valid( qr_code, msg.url)

    # Start building QR codes
    img = auto4.generate(api, qr_code, prompt)

    # Process the corners
    #post_proc.draw_corners(img)
    shifted = post_proc.value_shift( img, qr_code_1px )

    # Confirm the image is valid
    # qrcode.is_valid( img, msg.url )

    return img, shifted

async def process_image( api: webuiapi, bot: commands.Bot, queue: asyncio.Queue ):
    print("Image processor online...")
    print()
    while True:
        # Pull valid messages
        if (msg := await queue.get()) is None:
            continue
            
        # Execute the command
        with concurrent.futures.ThreadPoolExecutor() as pool:
            img0, img = await asyncio.get_running_loop().run_in_executor(
                            pool,
                            generate_qr_code, # working function that runs threaded
                            api, msg.url, msg.prompt ) # args to pass to the function


        await msg.processing.delete()

        # Send the image to the user
        content = f'**Vanity QR Code**\n\nURL:\n> {msg.url}\nPrompt:\n> {msg.prompt}'
        await cda_discord.send_image_to_channel(msg.ctx, img, content)
        await cda_discord.send_image_to_channel(msg.ctx, img0, "Original")

    # Let any task join commands actually finish
    queue.task_done()


def main():
    api = auto4.connect()
    queue = asyncio.Queue()
    
    # Create a bot instance
    async def init_ex(bot: commands.Bot):
        asyncio.create_task(process_image(api, bot, queue))

    bot = cda_discord.connect( callback=init_ex )

    # Command: vanity qr gen
    @bot.hybrid_command(name="vanity", description="Create a vanity QR code")
    async def vanity(ctx, url: str, *, prompt: str):
        proc = await ctx.reply("Processing...")
        await queue.put(QueueMsg(ctx, proc, url, prompt))

    # This runs the bot and blocks until the bot process is closed
    bot.run(settings.DISCORD['BOT_TOKEN'])

# Run the main function
if __name__ == "__main__":
    main()
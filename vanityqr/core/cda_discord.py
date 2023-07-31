from PIL import Image
from discord.ext import commands
import discord, aiohttp, io


def connect(ints=['message_content'], callback=None):
    # Create a bot instance
    intents = discord.Intents.default()
    for i in ints:
        intents.__setattr__(i, True)

    bot = commands.Bot(command_prefix='/', intents=intents)
    
    # Runs once the bot has logged in successfully
    @bot.event
    async def on_ready():
        print(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
        if callback is not None:
            print("Running callback...")
            await callback(bot)
        print('Syncing commands for tab completion...')
        await bot.tree.sync()

    # Command: health
    @bot.hybrid_command(name="health", description="Check health of the bot.")
    async def health(ctx):
        await ctx.send(f'Bot is functioning')

    return bot


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
async def send_image_to_channel(ctx, image: Image, content="" ):
    # Create a BytesIO object
    byte_stream = io.BytesIO()
    image.save(byte_stream, format='PNG')
    byte_stream.seek(0)

    # Build the response and send it
    file = discord.File(byte_stream, 'image.png')
    return await ctx.reply(file=file, content=content)


# Function to actually send the image
async def send_audio_to_channel(ctx, audio: str, content=""):
    # Create a BytesIO object
    try:
        with open(audio, "rb") as byte_stream:
            byte_stream.seek(0)

            # Build the response and send it
            file = discord.File(byte_stream, 'audio.mp3')
            return await ctx.reply(file=file, content=content)

    # Errors...
    except:
        return await ctx.reply(f"Error reading audio file: {audio}")

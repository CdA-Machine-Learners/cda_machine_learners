from PIL import Image
from diffusers import DiffusionPipeline
from discord.ext import commands
import asyncio, concurrent.futures, torch
import cda_discord, settings


class QueueMsg:
    def __init__(self, ctx, proc, prompt):
        self.ctx = ctx
        self.processing = proc
        self.prompt = prompt


def run_sdxl( base_pipe, refiner_pipe, prompt ):
    neg = "(worst quality, bad quality:1.2)(easynegative), (worst quality:2), (low quality:2), (normal quality:2),watermark, signature, lowres, ((monochrome)), ((grayscale)), cropped, signature, watermark, framed, border, grain, dust, film grain"

    # Run this to skip the refiner loop
    # return base_pipe(prompt=prompt, negative_prompt=neg ).images[0]

    # First XL pass
    latent_image = base_pipe(prompt=prompt, negative_prompt=neg, output_type="latent" ).images
    # Refine the latent_image
    return refiner_pipe(prompt=prompt, negative_prompt=neg, image=latent_image).images[0]


async def process_request( bot: commands.Bot, queue: asyncio.Queue ):
    print("Building base pipe")
    base_pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16,
        use_safetensors=True, variant="fp16",
        use_auth_token=settings.HUGGING_FACE_TOKEN,
    )
    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    base_pipe.to("cuda")

    print("Building refiner pipe")
    refiner_pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-0.9",
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
        use_auth_token=settings.HUGGING_FACE_TOKEN,
    )
    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    refiner_pipe.to("cuda")

    print("Starting processor")
    while True:
        # Pull valid messages
        if (msg := await queue.get()) is None:
            continue

        # Execute the command
        with concurrent.futures.ThreadPoolExecutor() as pool:
            image = await asyncio.get_running_loop().run_in_executor(
                    pool,
                    run_sdxl, # working function that runs threaded
                    base_pipe, refiner_pipe, msg.prompt ) # args to pass to the function

        # Send the image to the user
        await msg.processing.delete()

        content = f'**SD XL**\n\nPrompt:\n> {msg.prompt}'
        await cda_discord.send_image_to_channel(msg.ctx, image, content )

    # Do something with the image
    queue.task_done()


def main():
    # Setup my custom queue
    queue = asyncio.Queue()

    # Create a bot instance
    async def init_ex(bot: commands.Bot):
        asyncio.create_task(process_request(bot, queue))

    bot = cda_discord.connect( callback=init_ex )

    # Define my commands
    @bot.hybrid_command(name="sdxl", description="Run SD XL.")
    async def sdxl(ctx, *, prompt: str):
        proc = await ctx.reply("Processing...")
        await queue.put(QueueMsg(ctx, proc, prompt))

    # This runs the bot and blocks until the bot process is closed
    bot.run(settings.DISCORD_BOT_TOKEN)


# Run main!
main()
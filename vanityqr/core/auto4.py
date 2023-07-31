import io

from PIL import Image, ImageDraw
import numpy as np
from urllib3.exceptions import NewConnectionError

from core import qrcode

import webuiapi


def connect():
    try:
        # Create my api access
        api = webuiapi.WebUIApi(sampler="DPM++ 2M Karras", steps=30)
        options = api.get_options()
        #options['sd_model_checkpoint'] = 'revAnimated_v122.safetensors [4199bcdd14]'
        #options['sd_model_checkpoint'] = 'ghostmix_v20Bakedvae.safetensors [e3edb8a26f]'
        api.set_options(options)

        return api
        
    except NewConnectionError:
        pass
    except ConnectionError:
        pass
    except ConnectionRefusedError:
        pass

    print("Failed to connect to the Auto11111.")
    print("Did you forget to start it?")
    exit(1)


def generate( api, qr_code, prompt, seed=-1, start=0.2 ):
    print(qr_code.size)
    width, height = qr_code.size

    unit1 = webuiapi.ControlNetUnit(
        input_image=qr_code,
        model='control_v11f1e_sd15_tile [a371b31b]',
        #guidance_start=0.35,#start,
        #guidance_end=0.7,
        #weight=0.5,
        guidance_start=0.18,
        guidance_end=0.9,
        weight=1,
        processor_res=width,
    )

    # Generate the image
    result1 = api.img2img(
        images=[qr_code],
        prompt=prompt,
        negative_prompt="cartoon, painting, illustration, (worst quality, low quality, normal quality:2), (nsfw)",
        #negative_prompt="ugly, disfigured, low quality, blurry, (nsfw)",
        #width=width,
        #height=height,
        cfg_scale=7,
        controlnet_units=[unit1],
        steps=30,
        sampler_index='DPM++ 2M Karras',
        denoising_strength=0.75,
        seed=seed,
        # styles=["anime"],
#                      enable_hr=True,
#                      hr_scale=2,
#                      hr_upscaler=webuiapi.HiResUpscaler.Latent,
#                      hr_second_pass_steps=20,
#                      hr_resize_x=1536,
#                      hr_resize_y=1024,
    )

    return result1.image

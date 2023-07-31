import io

from PIL import Image, ImageDraw
import numpy as np

from core import qrcode

from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import RoundedModuleDrawer
from qrcode.image.styles.colormasks import RadialGradiantColorMask


import webuiapi


qr_code = qrcode.render("https://www.google.com", box_size=18, err=qrcode.constants.ERROR_CORRECT_H)
qr_code.save("qr_code.png")

api = webuiapi.WebUIApi()#sampler="DPM++ 2M Karras", steps=40)

options = api.get_options()
options['sd_model_checkpoint'] = 'ghostmix_v20Bakedvae.safetensors [e3edb8a26f]'
api.set_options(options)

unit1 = webuiapi.ControlNetUnit(
    input_image=qr_code,
    #module='tile_resample',
    model='control_v11f1e_sd15_tile [a371b31b]',
    #guidance=0.87,
    guidance_start=0,#.23,
    guidance_end=0.9,
    weight=0.8,
    #control_mode=2,
)
#)

for i in range(4):
    unit1.guidance_start = 0.1 + i * 0.08
    result1 = api.img2img(
                        images=[qr_code],
                        prompt="a cubism painting of a town with a lot of houses in the snow with a sky background, Andreas Rocha, matte painting concept art, a detailed matte painting",
                        negative_prompt="ugly, disfigured, low quality, blurry, nsfw",
                        #styles=["anime"],
        #seed=917,
                        cfg_scale=7,
                        controlnet_units=[unit1],
                        steps=40,
                        sampler_index='DPM++ 2M Karras',
                        denoising_strength=0.75,
    #                      sampler_index='DDIM',
    #                      steps=30,
    #                      enable_hr=True,
    #                      hr_scale=2,
    #                      hr_upscaler=webuiapi.HiResUpscaler.Latent,
    #                      hr_second_pass_steps=20,
    #                      hr_resize_x=1536,
    #                      hr_resize_y=1024,
    #                      denoising_strength=0.4,
                        )

    result1.image.save(f"aaa{i}.png")


x = {
    "init_images": [
        "string"
    ],
    "resize_mode": 0,
    "denoising_strength": 0.75,
    "image_cfg_scale": 0,
    "mask": "string",
    "mask_blur": 4,
    "inpainting_fill": 0,
    "inpaint_full_res": True,
    "inpaint_full_res_padding": 0,
    "inpainting_mask_invert": 0,
    "initial_noise_multiplier": 0,
    "prompt": "",
    "styles": [
        "string"
    ],
    "seed": -1,
    "subseed": -1,
    "subseed_strength": 0,
    "seed_resize_from_h": -1,
    "seed_resize_from_w": -1,
    "sampler_name": "string",
    "batch_size": 1,
    "n_iter": 1,
    "steps": 50,
    "cfg_scale": 7,
    "width": 512,
    "height": 512,
    "restore_faces": False,
    "tiling": False,
    "do_not_save_samples": False,
    "do_not_save_grid": False,
    "negative_prompt": "string",
    "eta": 0,
    "s_min_uncond": 0,
    "s_churn": 0,
    "s_tmax": 0,
    "s_tmin": 0,
    "s_noise": 1,
    "override_settings": {},
    "override_settings_restore_afterwards": True,
    "script_args": [],
    "sampler_index": "Euler",
    "include_init_images": False,
    "script_name": "string",
    "send_images": True,
    "save_images": False,
    "alwayson_scripts": {}
}


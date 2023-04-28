#!/usr/bin/python

import torch, sys, re, os
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DiffusionPipeline

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} prompt_file")
    exit(0)

# Comment for a stupid amount of logging info
os.environ["MIOPEN_ENABLE_LOGGING"] = "0"
os.environ["MIOPEN_ENABLE_LOGGING_CMD"] = "0"
os.environ["MIOPEN_LOG_LEVEL"] = "1"

# Defaults
opt = {
    "width": 512,
    "height": 512,
}

# Select your model
model_id = "Lykon/NeverEnding-Dream" # Super real
#model_id = ("Lykon/DreamShaper", "5_beta2_BakedVae") # Super real
#model_id = "andite/anything-v4.0" # Fantisy
#model_id = "iriscope/cartoonavatar"
#model_id = "runwayml/stable-diffusion-v1-5" # Ugly

# Setup the pretrained args
pt_args = { "safety_checker": None }
if isinstance( model_id, tuple ):
    pt_args['variant'] = model_id[1]
    model_id = model_id[0]

# Setup teh pipe line
pipeline = StableDiffusionPipeline.from_pretrained(model_id, **pt_args)
#pipeline = DiffusionPipeline.from_pretrained(model_id, **pt_args)

# Optional specific pipeline
#pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

# Send the GPU, AMD ROCm but still says cuda
pipeline.to("cuda")

# Load prompts from promp file
print("")
print(f"Loading prompts: {sys.argv[1]}")
with open(sys.argv[1]) as handle:
    negative_prompt = ''
    output = re.sub('/', '_', sys.argv[1])
    prompts = handle.readlines()

    # Iterate through my prompts
    idx = 0
    for prompt in prompts:
        if re.search('[A-Za-z]', prompt) is None:
            continue
        if re.search("^#", prompt):
            print(prompt)
            continue

        # Update the negative prompt?
        if re.search('^-', prompt) is not None:
            negative_prompt = re.sub('^-', '', prompt)
            print(f"")
            print(f"Negative Prompt: {negative_prompt}")
            print(f"")
            continue

        # Run the prompt
        print(prompt)
        image = pipeline(prompt, negative_prompt=negative_prompt, **opt).images[0]
        # save the output
        print(f"    {output}_{idx}.png\r\n")
        image.save(f"{output}_{idx}.png")
        idx += 1

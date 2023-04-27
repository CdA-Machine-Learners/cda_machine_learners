# Usage

    ./main prompts_example1
    # Outputs PNG files of the prompt

# Changing the model

Inside the code at the top, you'll see a few example model repos to choose from. Uncomment the one you want and wait for the multi GB download. If you want to run a specific variant, set the model_id to tuple, with the repo as the first element and the variant as the second. Example **DreamShaper** inside main.py

# Customizing params

opt = xxx at the top of the file will take any params that are passed into the prompt call. Currently just width and height are given. For details on the possible params:

https://huggingface.co/docs/diffusers/v0.16.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline

# Prompt file

There are 3 prompt_files. Please look at those for examples.

**Description of the prompt file format**
* If a line begins with '-', then that is a negative prompt. Negative prompts are used repeatedly until another negative prompt is specified. Typically you'll have 1 negative prompt at the top of your file.
* If the line begins with a '#', then that line is a comment. Comments and printed to the screen during rendering
* Otherwise, the entire line is considered a prompt. You can have as many prompts as you like. Each prompt will output one image: {prompt_filename}_{index}.png

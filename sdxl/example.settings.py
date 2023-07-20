# Only needed if the send_url_to_channel function is used
S3_ACCESS = {
    "MODE": "DEV",
    "BUCKET": "beenthere",
    "REGION": "sfo3",
    "HOST": "digitaloceanspaces.com",

    "EXTRA_ARGS": "public-read",
    "ACCESS_KEY": "XXX",
    "SECRET_KEY": "XXX",
}

# Inside of the Discord Developer Portal, create a bot and paste the token here
DISCORD_BOT_TOKEN   = "XXX"
DISCORD_BOT_ID      = 2222222

# Go to https://huggingface.co/settings/tokens, create a token, and paste it here
# Also, goto https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9 and make sure to signup for the model
HUGGING_FACE_TOKEN = "hf_XXX"
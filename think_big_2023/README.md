# Think Big 2023

## Goals:

  * a single discord channel that the bot responds to, so we don't get flooded in other channels.
  * Should we open the bot up to private messages?
  * The bots:
    * /t2midjourney <prompt> is there a simple way to intercept these messages and have my paid account mirror the request? That could be cool.
    * /t2i <prompt> text 2 image, StableD
    * /i2i <url> <prompt>, img 2 img, StableD
    * /t2t <prompt>, ChatGPT
    * /t2v <prompt> some text 2 vid model, I have yet to experiment here
    * /t2langchain if anyone has any cool ideas

## Usage

Just get a copy of the `TOKEN` and go for it!

```sh

TOKEN=$(cat .discord_token) python src/main.py

```

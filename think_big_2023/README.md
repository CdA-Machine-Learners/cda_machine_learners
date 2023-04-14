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

### Setup
Copy the `.env_sample` file to `.env` and fill in the values.

```sh
cp .env_sample .env
```

Make sure you have Docker and Docker Compose installed.

```sh
docker-compose build
```

### Running

To run the bot in the background, use the following command:
```sh
docker-compose up -d
```
Remove the -d flag to run in the foreground and see logs.

### Running with Multi-Threading
To achieve multi-threading, we scale the number of celery workers we have.
Since python is single-threaded, this is the easiest way.
Replace `NUM_WORKERS` with the number of workers you want to run.
One or two workers short of the number of cores you have is a good starting point.
```sh
docker-compose up -d --scale celery_worker=NUM_WORKERS
```
**WARNING:** Make sure you have enough RAM to run the number of workers you want.
LLMs are memory intensive. You can change the allocated amount of RAM in the Docker desk

### Stopping
To stop a container that is running in the background, use the following command:
```sh
docker-compose down
```
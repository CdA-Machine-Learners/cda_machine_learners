import re

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import soundfile as sf
import numpy as np
import subprocess

import nltk, uuid

import settings

nltk.download('punkt')  # Download the necessary resources for sentence tokenization


def text_to_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences


def exec( cmd ):
    print(cmd)
    subprocess.run(cmd.split(' '))


def text_to_chunks( blob ):
    chunks = []

    # Build out the text
    current = ""
    for sent in text_to_sentences(blob):
        if len(current + sent) < 500:
            current = f"{current}. {sent}"
        else:
            chunks.append(f"{current}.")
            current = sent

    if len(current) > 0:
        chunks.append(current)

    return chunks


def text_to_speech( processor, model, speaker_embeddings, vocoder, blob, cb=None ):
    if cb is None:
        cb = lambda a,b: print(f"Processing chunk {a+1} of {b}")

    chunks = text_to_chunks( blob )

    # Hacky AF but create a working directory
    uuid_str = str(uuid.uuid4())
    base = f"{settings.SCRATCH}/{uuid_str}"
    exec(f"mkdir {settings.SCRATCH}/{uuid_str}")

    # Generate speech
    with open(f"{base}/concat.txt", "w") as handle:
        for idx, chunk in enumerate(chunks):
            cb( idx, len(chunks) )

            inputs = processor(text=chunk, return_tensors="pt")
            speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

            filename = f"{base}/speech{idx}.wav"
            sf.write( filename, speech.numpy(), samplerate=16000)
            #if idx > 0: handle.write(f"file 'pause.wav'\n") # Hack to get the timing right between chunks
            handle.write(f"file '{filename}'\n")

    # Concatenate the speech
    #exec("sox -n -r 44100 -c 2 pause.wav trim 0.0 1.0")
    exec( f"ffmpeg -f concat -safe 0 -i {base}/concat.txt -c copy {base}/output.wav")

    # Convert to mp3
    audio_file = f"audio.mp3"
    exec( f"ffmpeg -i {base}/output.wav -vn -ar 44100 -ac 2 -b:a 192k {base}/{audio_file}")

    # Add the video
    #exec( f"ffmpeg -i {audio_file} -filter_complex movie=video.mp4:loop=0,setpts=N/FRAME_RATE/TB -shortest audio.mp4")

    # Clean up
    print("Clean up")
    kill = ["rm", f"concat.txt", "output.wav"] #, "pause.wav"]
    for idx, chunk in enumerate(chunks):
        kill.append( f"speech{idx}.wav" )

    #subprocess.run( kill )

    return f"{base}/{audio_file}"

import re

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch, sys
import soundfile as sf
import numpy as np
import subprocess

import nltk, sys

nltk.download('punkt')  # Download the necessary resources for sentence tokenization


def text_to_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences


def exec( cmd ):
    print(cmd)
    subprocess.run(cmd.split(' '))

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")


# load xvector containing speaker's voice characteristics from a dataset
#embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
#speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
spkr = np.load("cmu_us_ksp_arctic-wav-arctic_b0087.npy")
#spkr = np.load("cmu_us_bdl_arctic-wav-arctic_a0009.npy")
speaker_embeddings = torch.tensor(spkr).unsqueeze(0)

# Build out the text
chunks = []
with open(sys.argv[1]) as handle:
    blob = handle.read()
    blob = re.sub(r'\n', ' ', blob)
    blob = re.sub(r'\r', ' ', blob)

    current = ""
    for sent in text_to_sentences(blob):
        if len(current + sent) < 500:
            current = f"{current}. {sent}"
        else:
            chunks.append(f"{current}.")
            current = sent

    if len(current) > 0:
        chunks.append(current)

# Generate speech
with open("concat.txt", "w") as handle:
    for idx, chunk in enumerate(chunks):
        print(f"Processing chunk {idx+1} of {len(chunks)}")
        inputs = processor(text=chunk, return_tensors="pt")
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

        filename = f"speech{idx}.wav"
        sf.write( filename, speech.numpy(), samplerate=16000)
        #if idx > 0: handle.write(f"file 'pause.wav'\n") # Hack to get the timing right between chunks
        handle.write(f"file '{filename}'\n")

# Concatenate the speech
#exec("sox -n -r 44100 -c 2 pause.wav trim 0.0 1.0")
exec( "ffmpeg -f concat -safe 0 -i concat.txt -c copy output.wav")

# Convert to mp3
audio_file = f"{sys.argv[1]}.mp3"
exec( f"ffmpeg -i output.wav -vn -ar 44100 -ac 2 -b:a 192k {audio_file}")

# Add the video
exec( f"ffmpeg -i {audio_file} -filter_complex movie=video.mp4:loop=0,setpts=N/FRAME_RATE/TB -shortest {sys.argv[1]}.mp4")

# Clean up
print("Clean up")
kill =["rm", "concat.txt", "output.wav"] #, "pause.wav"]
for idx, chunk in enumerate(chunks):
    kill.append( f"speech{idx}.wav" )

subprocess.run( kill )

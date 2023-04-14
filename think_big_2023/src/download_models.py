from transformers import GPT2LMHeadModel, GPT2Tokenizer
from os import environ


# This file is run on docker build to download and cache models

def download_gpt2_model(model_name='gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    path = f"{environ.get('MODEL_DIR', './models')}/{model_name.replace('/', '-')}/"
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


if __name__ == '__main__':
    download_gpt2_model()

'''

Test Transformers, Find What Gives High TOK/S

----------
DEPENDENCIES:

pip install tqdm  matplotlib torch unidecode nltk
pip install -e "$HOME/_/lib/transformers"  # git clone this
pip install -e "$HOME/_/lib/accelerate"  # git clone this
pip install packaging ninja
pip install flash-attn --no-build-isolation
pip install bitsandbytes

'''

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from unidecode import unidecode
import json
import nltk
import os
import re
import torch
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

MODEL_PATH = "~/_/models/phi-2"
MODEL_PATH = os.path.expanduser(MODEL_PATH)

try:
    model_loaded
    print('model already loaded')
except:
    print('loading model')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        # load_in_8bit=True,
        trust_remote_code=False,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_loaded = True
    print('done loading model')


# Phi-2's Chat Template
if 'phi-2' in MODEL_PATH:
    tokenizer.chat_template = '''{% for message in messages %}
{% if message['role'].lower() == 'user' %}
{{ "Alice: " + message['content'] }}
{% else %}
{{ "Bob: " + message['content'] }}
{% endif %}
{% endfor %}'''


def not_special(tok_id):
    return (
        tok_id != tokenizer.eos_token_id and
        tok_id != tokenizer.bos_token_id and
        tok_id != tokenizer.pad_token_id and
        tok_id != tokenizer.unk_token_id
    )

def count_tokens(batch_toks):
    return len(list(filter(lambda x: not_special(x), batch_toks.flatten().tolist())))


##################################################
# Data

data_path = os.path.expanduser('~/_/cda_machine_learners/bots/toy_data_ml_dialog.jsonl')
data = []
with open(data_path, 'r') as f:
    raws = f.readlines()
    for raw in raws:
        data.append(json.loads(raw))
print(f'loaded data, {len(data)} rows')

# Because the dialogs end with 'assistant', continue with a 'user' query.
keep_going = [{'role': 'user', 'content': 'Please explain in detail.'}]


##################################################
#

def test_batch_size(batch_size):
    print(f'Computing batch size={batch_size}', end='')

    # Prepare input
    inp = [tokenizer.apply_chat_template(
        x + keep_going,
        tokenize=False, return_tensors=False, add_generation_prompt=True
    ) for x in data[:batch_size]]
    tokenized = tokenizer(inp, padding='longest', return_tensors='pt', add_special_tokens=True)
    tokenized['attention_mask'] = tokenized['attention_mask'].to('cuda')
    tokenized['input_ids'] = tokenized['input_ids'].to('cuda')

    # Generate output and measure time
    start_time = time.time()
    out_toks = model.generate(
        **tokenized,
        max_new_tokens=32,  # VARIABLE
        use_cache=True,  # (huge slowdown without)
    )
    elapsed_time = time.time() - start_time

    # trim off the query prefix (model.generate returns it)
    out_toks = out_toks[:, tokenized['input_ids'].shape[1]:]

    # Count tokens and calculate tokens per second
    num_toks = count_tokens(out_toks)
    toks_per_second = num_toks / elapsed_time
    print(f', time={elapsed_time}, tok/s={toks_per_second}')

    return tokenized['input_ids'], out_toks, toks_per_second

# Test different batch sizes
batch_sizes = [1, 2, 4, 8, 16]
results = []
for bs in tqdm(batch_sizes):
    inp_toks, out_toks, tok_per_s = test_batch_size(bs)
    results.append(tok_per_s)


##########
# Plotting

plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, results, marker='o')
plt.xlabel('Batch Size')
plt.ylabel('Tokens per Second')
plt.title('Tokens per Second vs. Batch Size')
plt.grid(True)
plt.show()


##########
# Demo of what it's generating

for i, o in zip(inp_toks, out_toks):
    i = tokenizer.decode(i)
    o = tokenizer.decode(o)
    print('----------')
    print('QUESTION:')
    print(i)
    print('ANSWER:')
    print(o)

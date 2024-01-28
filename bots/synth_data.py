'''

Generate Synthetic Data

DEPS:
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
from nltk.tokenize import sent_tokenize

source_paths = [
    'meditations.txt'
]

'''

Get prompts working nicely: Batching, Templating

DEPS:
pip install packaging ninja torch
pip install flash-attn --no-build-isolation

'''

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from unidecode import unidecode
import json
import nltk
import os
import re
import torch
from nltk.tokenize import sent_tokenize
from typing import List, Generator
import time
import sys

# Sliding window of N sentences, moving forward by a certain stride of sentences.
WINDOW = 3
STRIDE = 3
BATCH_SIZE = 1
MAX_NEW_TOKENS = 128


##################################################
# Clean up prep

def remove_text_after_substring(text, substring):
    # Find the index where the substring occurs
    index = text.find(substring)

    # If the substring is found, cut the text up to that point
    if index != -1:
        return text[:index]
    else:
        return text


def clean(text):
    text = re.sub(r' +', ' ', text)
    return text


##################################################
# Load Model

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
        # torch_dtype=torch.float16,
        load_in_8bit=True,
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

def count_tokens(batch_toks):
    return len(list(filter(lambda x: x != tokenizer.eos_token, batch_toks.flatten().tolist())))


##################################################
# Prompt

def prompt(context, passage):
    dialog = [
        {
            'role':'user',
            'content': f"""\
Please read the following text, and simulate a dialog between a Student and Teacher. The Student asks intelligent questions about the passage, covering all the important knowledge from the passage. The Teacher gives long and thoughtful answers. Let's warm up with a short passage:

Context

{context}

Passage

Stoicism became particularly fashionable in the Roman period. Although never identifying as a Stoic himself, Cicero, who studied philosophy in Athens and endeavored to popularize it at Rome, engaged extensively with Stoic theory in his philosophical works and modeled his On Proper Functions (De Officiis) on Panaetius’ treatise of the same name. During the Imperial era, several prominent public figures were associated with the Stoic school. Appointed to the court of Emperor Augustus in the late 1st century BCE was the Stoic philosopher Arius Didymus, and a generation later Seneca served as advisor to Nero."""
        },
        {
            'role':'assistant',
            'content': """\
Understood. I should simulate a dialog between a Teacher and Student, and the dialog should be supported by the provided passage. Here is a hypothetical conversation:

[Student] Was Cicero a Stoic?

[Teacher] Indeed, although Stoicism was becoming fashionable at that time in Rome, Cicero never identified as a Stoic. He studied philosophy in Athens, and endeavored to popularize it in Rome. He did however engage with Stoic theory in his philosophical works and modeled his On Proper Functions (De Officiis) on Panaetius' treatise of the same name.

[Student] I heard that Stoic philosophers were once employed by leaders. Do you know of any examples of this?

[Teacher] Yes! Interest in Stoicism climbed during the Roman period, and several stoic philosophers reported to the emperors. For example, during the Imperial era, in the late 1st century BCE, Arius Didymus was appointed to the court of Emperor Augustus. A generation later, Seneca served as advisor to Nero."""
        },
        {
            'role':'user',
            'content': f"""
That's great, good job imagining that dialog. It sticks to the given details well, which is critical ie to ground the discussion in facts from the passage. For our next passage, allow the Student to ask more open ended questions, and allow the Teacher to give longer and more detailed answers, as if in a lecture. Here is the next passage for you to convert to dialog:

Context:

{context}

Passage:

{passage}"""
        },
        {
            'role':'assistant',
            'content': f"""
I see. I will focus on having the Student ask more broad questions, and the Teacher will answer open ended questions at great length, and in great detail like part of a lecture. Above all we should ground the discussion on the original passage and not infer too much. Now let's generate a dialog for your new passage:

[Student] """
        },
    ]
    p = tokenizer.apply_chat_template(
        dialog, tokenize=False, return_tensors=False, add_generation_prompt=True
    )

    if 'phi-2' not in MODEL_PATH:
        # Remove end token from the chat template, to allow assistant to keep
        # writing, itself. Phi doesn't have this on the end.
        # p = p[:-len(' </s>')]
        p = p[:-len(f' {tokenizer.eos_token}')]
    return p


##################################################
# Load Book

def tok(text):
    return tokenizer.encode(text, add_special_tokens=False)

def dtok(x):
    return tokenizer.decode(x)

def newline(x):
    '''Escape newlines'''
    return x.replace('\n', '\\n')

BOS = tokenizer.bos_token_id # beginning of string
EOS = tokenizer.eos_token_id # end of string
INST = tok("[INST]")         # user Start
_INST = tok("[/INST]")       # user End


def batched_scan(sentences: List[str], window_size: int, stride: int, batch_size: int) -> Generator[List[str], None, None]:
    """
    A generator that yields batches of windows created from the sentences list.

    :param sentences: List of sentences to process.
    :param window_size: The size of each window.
    :param stride: The stride between windows.
    :param batch_size: The number of windows in each batch.
    :return: Yields lists of joined window strings, each list of length `batch_size`.
    """
    window_batch = []
    for i in range(0, len(sentences) - (window_size - 1), stride):
        window = ' '.join(sentences[i:i + window_size])
        window_batch.append(window)

        if len(window_batch) == batch_size:
            yield window_batch
            window_batch = []

    # Yield any remaining windows in the last batch
    if window_batch:
        yield window_batch


def generate_batch(batch: List[str]):
    tqdm.write(f'Computing batch size={len(batch)}', end='')

    # Prepare input
    tokenized = tokenizer(batch, padding='longest', return_tensors='pt', add_special_tokens=True)
    tokenized['attention_mask'] = tokenized['attention_mask'].to('cuda')
    tokenized['input_ids'] = tokenized['input_ids'].to('cuda')

    # Generate output and measure time
    start_time = time.time()
    out_toks = model.generate(
        **tokenized,
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True,  # (huge slowdown without)
    )
    elapsed_time = time.time() - start_time

    # trim off the query prefix (model.generate returns it)
    out_toks = out_toks[:, tokenized['input_ids'].shape[1]:]

    # Count tokens and calculate tokens per second
    num_toks = count_tokens(out_toks)
    toks_per_second = num_toks / elapsed_time
    tqdm.write(f', time={elapsed_time}, new_toks={num_toks}, tok/s={toks_per_second}')

    return tokenized['input_ids'], out_toks, toks_per_second


def extract_dialog_pairs(dialog):
    """
    Extracts pairs of questions and responses from a dialog and returns them as a list of dictionaries.
    The keys in the dictionary are 'role' (either 'user' or 'assistant') and 'content' (text of the dialog).

    :param dialog: A string containing the dialog.
    :return: A list of dictionaries with the dialog pairs.
    """
    # Regular expression pattern to match the dialog pairs
    pattern = r"\[(student|teacher)\](.*?)(?=\[student\]|\[teacher\]|\<\/s\>|$)"

    # Finding all matches
    matches = re.findall(pattern, dialog, re.DOTALL | re.IGNORECASE)

    dialog_pairs = []
    for role, content in matches:
        if role.lower() == 'teacher':
            role = 'assistant'
        elif role.lower() == 'student':
            role = 'user'
        dialog_pairs.append({'role': role, 'content': content.strip()})

    return dialog_pairs

def format_book_info(book_json):
    lines = []
    if 'title' in book_json:
        lines.append(f"Title: {book_json['title']}")
    if 'authors' in book_json:
        authors = ', '.join(book_json['authors'])
        lines.append(f"Authors: {authors}")
    if 'faith' in book_json:
        lines.append(f"Faith: {book_json['faith']}")
    if 'publication_date' in book_json:
        lines.append(f"Publication Date: {book_json['publication_date']}")
    return '\n'.join(lines)


##################################################
# Go!

total_errors = 0

for source_path in tqdm(source_paths):

    ##########
    # Load Book

    book_path = os.path.expanduser(source_path)
    json_path = book_path + '.json'
    output_path = book_path + '.synth01.jsonl'
    with open(book_path, 'r') as f:
        raw_book = f.read()
    with open(json_path, 'r') as f:
        book_json = json.loads(f.read())
    raw_book = unidecode(raw_book)  # cleans up eg greek chars that aren't
    # book = raw_book[100000:105000]
    book = raw_book
    book = clean(book)
    book = remove_text_after_substring(book, "End of Project Gutenberg's").strip()
    book = remove_text_after_substring(book, "End of the Project Gutenberg").strip()
    book = book.replace('\n', ' ')
    sentences = sent_tokenize(book)
    book_loaded = True
    print(f'done loading book, len(sentences) = {len(sentences)}')


    ##########
    # Generate some synth data!

    context = format_book_info(book_json)

    # Total number of windows that can be generated
    total_windows = (len(sentences) - (WINDOW - 1) + STRIDE - 1) // STRIDE

    # Total number of batches
    total_batches = (total_windows + BATCH_SIZE - 1) // BATCH_SIZE

    for batch in tqdm(batched_scan(sentences, WINDOW, STRIDE, BATCH_SIZE), total=total_batches):
        try:
            formatted_batch = [prompt(context, x) for x in batch]
            inp_toks, out_toks, tok_per_s = generate_batch(formatted_batch)

            # Save Results
            with open(output_path, 'a') as f:
                for passage, out_tok in zip(batch, out_toks):
                    o = tokenizer.decode(out_tok)
                    o = '[Student] ' + o
                    dialog = extract_dialog_pairs(o)

                    tqdm.write('----------')
                    tqdm.write(f'tok/s: {tok_per_s}')
                    tqdm.write('PASSAGE:')
                    tqdm.write(passage[:100], end=f'\n...{len(passage)} chars...\n')
                    tqdm.write(passage[-100:])
                    tqdm.write('DIALOG:')
                    # tqdm.write(str(dialog))
                    tqdm.write(o)

                    if len(dialog) >= 2:
                        json.dump({
                            'passage': passage,
                            'dialog': dialog,
                        }, f)
                        f.write('\n')
        except Exception as e:
            print(f'ERROR ERROR: {e}')
            if total_errors > 6:
                sys.exit(1)
                BRK
            total_errors += 1

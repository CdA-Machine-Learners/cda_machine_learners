'''

Training the model

----------
SETUP:

pip install git+https://github.com/huggingface/transformers
pip install git+https://github.com/huggingface/accelerate
pip install git+https://github.com/huggingface/peft
pip install bitsandbytes datasets scipy tensorboard

accelerate config

----------
DATA PREP:

https://huggingface.co/learn/nlp-course/chapter7/6

> Getting rid of all the chunks that are smaller than the context size wasn’t a big issue here because we’re using small context windows. As you increase the context size (or if you have a corpus of short documents), the fraction of chunks that are thrown away will also grow. A more efficient way to prepare the data is to join all the tokenized samples in a batch with an eos_token_id token in between, and then perform the chunking on the concatenated sequences. As an exercise, modify the tokenize() function to make use of that approach. Note that you’ll want to set truncation=False and remove the other arguments from the tokenizer to get the full sequence of token IDs.

- Triangular Mask
- return_overflowing_tokens
- return_length



----------
RUNNING IT:

python t04_peft.py

tensorboard --logdir logs

'''


from accelerate import FullyShardedDataParallelPlugin, Accelerator
from datasets import load_dataset, Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import matplotlib.pyplot as plt
import torch
import os
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import transformers
from datetime import datetime
import json
import re
print()


##########
#

RUN_NUMBER = 2
project = f"bible-{RUN_NUMBER}"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

if os.path.exists(output_dir):
    raise Exception(f"ERROR: {output_dir} already exists")


##########
# Model

# model_path = "models/mistral/Mistral-7B-v0.1"
model_path = "models/mistral/Mistral-7B-Instruct-v0.2"

try:
    already_loaded
    print('model already loaded')
except:
    print('loading model')
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    # Multi-gpu fix
    #   https://github.com/huggingface/accelerate/issues/1840#issuecomment-1683105994
    device_index = Accelerator().process_index
    device_map = {"": device_index}


    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        # device_map='cuda:0'
        # device_map='auto',
        # torch_dtype=torch.bfloat16,
        # device_map='balanced'
        device_map=device_map
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    already_loaded = True
    print('model loaded')

##########
# Dataset

CONTEXT_LENGTH = 1024

def join_generator(xss):
    for xs in xss:
        for x in xs:
            yield x

def chunked_stream(list_of_list_of_tokens, chunk_size):
    token_stream = join_generator(list_of_list_of_tokens)
    # Yield chunks of size CHUNK_SIZE
    chunk = []
    for token in token_stream:
        chunk.append(token)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []

def prepare_string(j):
    x = json.dumps(j)
    x = x.strip()
    x = re.sub(r'HCSB', 'CSB', x)
    x = re.sub(r' *CSB', '', x)
    x = re.sub(r'Christian Standard Bible', 'Bible', x)
    x = json.loads(x)

    if len(x) == 0:
        return ''
    try:
        x = tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=False)
    except Exception as e:
        # eg: jinja2.exceptions.TemplateError: Conversation roles must alternate user/assistant/user/assistant/...
        return ''
    return x

def stream_jsonl(paths, key):
    """
    Read all paths, and stream out the key from within.
    """
    for path in paths:
        with open(path, 'r') as file:
            raws = file.readlines()
            for raw in raws:
                j = json.loads(raw)[key] # get "dialog"
                yield prepare_string(j)

def tokenize_jsonls(directory_path, tokenizer, chunk_size):
    tokens = map(lambda x: tokenizer(x)['input_ids'], stream_jsonl(directory_path, key='dialog'))
    yield from chunked_stream(tokens, chunk_size)

try:
    X # force error, so reload anyways
    already_loaded_data
except:
    data_paths = [
        '../data/bible/christian_standard_bible.txt.synth01.jsonl',
        '../data/bible/christian_standard_bible.txt.synth02.jsonl',
    ]
    xs = tokenize_jsonls(data_paths, tokenizer, chunk_size=CONTEXT_LENGTH)
    xs = list(xs)
    xs = [{'input_ids': x} for x in  xs]
    already_loaded_data = True

full_dataset = Dataset.from_list(xs)
full_dataset = full_dataset.shuffle(seed=152)
full_dataset = full_dataset.train_test_split(test_size=0.15)
tokenized_train_dataset = full_dataset['train']
tokenized_val_dataset   = full_dataset['test']

print('Example Data:')
print(tokenizer.decode(xs[0]['input_ids']))

# # @@@@@@@@@@
# txt = 'a b c d e f'
# testo = tokenizer(
#     ['a b c dog elephant f g h i j',
#      'x y z'],
#     # truncation=True,
#     # return_overflowing_tokens=True,
#     # return_length=True,
#     # max_length=5,
# )
# xx = chunked_stream(testo['input_ids'], chunk_size=3)
# xx = [tokenizer.decode(x) for x in xx]
# print(list(xx))
# # @@@@@@@@@@


##########
# Visualization

# def plot_data_lengths(tokenize_train_dataset, tokenized_val_dataset):
#     lengths  = [len(x['input_ids']) for x in tokenized_train_dataset]
#     lengths += [len(x['input_ids']) for x in tokenized_val_dataset]
#     print(len(lengths))

#     # Plotting the histogram
#     plt.figure(figsize=(10, 6))
#     plt.hist(lengths, bins=20, alpha=0.7, color='blue')
#     plt.xlabel('Length of input_ids')
#     plt.ylabel('Frequency')
#     plt.title('Distribution of Lengths of input_ids')
#     plt.show()

# plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)


##########
# Prep model

# model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


##########
# Prep Config

config = LoraConfig(
    r=128, # lora rank
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)


##########
# Accelerate

# fsdp_plugin = FullyShardedDataParallelPlugin(
#     state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
#     optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
# )
# accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

# accelerator = Accelerator()
# model, tokenized_train_dataset, tokenized_val_dataset = accelerator.prepare(
#     model,
#     tokenized_train_dataset,
#     tokenized_val_dataset
# )


##################################################
# Go

# print('CUDA Device Count:', torch.cuda.device_count())
# if torch.cuda.device_count() > 1: # If more than 1 GPU
#     model.is_parallelizable = True
#     model.model_parallel = True


trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=100,
        per_device_train_batch_size=4,
        # gradient_accumulation_steps=1,

        # max_steps=601,
        num_train_epochs=13,

        learning_rate=1e-5, # Want a small lr for finetuning
        # bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=5,             # training loss (i think)
        logging_dir=output_dir,      # Directory for storing logs

        save_strategy="epoch",       # Save the model checkpoint every logging step
        # save_strategy="steps",       # Save the model checkpoint every logging step
        # save_steps=100,              # Save checkpoints every x steps

        evaluation_strategy="epoch", # Evaluate the model every logging step
        # evaluation_strategy="steps", # Evaluate the model every logging step
        # eval_steps=100,              # Evaluate and save checkpoints every x steps

        do_eval=True,                # Perform evaluation at the end of training
        report_to="tensorboard",
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"  # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()


# ##################################################
# # Test it out

# checkpoint_dir = output_dir + '/' + 'checkpoint-1600'

# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,
# )

# base_model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     quantization_config=bnb_config,
#     device_map="auto",
#     trust_remote_code=True,
# )

# tokenizer = AutoTokenizer.from_pretrained(model_path, add_bos_token=True, trust_remote_code=True)

# ft_model = PeftModel.from_pretrained(base_model, checkpoint_dir)

# eval_prompt = "So there we were."

# model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

# ft_model.eval()
# with torch.no_grad():
#     out = ft_model.generate(**model_input, max_new_tokens=300, repetition_penalty=1.5)
#     print(tokenizer.decode(out[0], skip_special_tokens=True))



##################################################


# from transformers import AutoModelForCausalLM
# from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

# model_path = "/home/josh/_/models/mistral/Mistral-7B-Instruct-v0.1"

# peft_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
# )

# model = AutoModelForCausalLM.from_pretrained(model_path)
# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()
# # output: trainable params: 2359296 || all params: 1231940608 || trainable%: 0.19151053100118282

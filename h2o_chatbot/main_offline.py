import torch, sys, datetime
from h2oai_pipeline import H2OTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3",
    use_fast=False,
    padding_side="left",
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3",
    torch_dtype=torch.float16,
    device_map={"": "cuda:0"},
    trust_remote_code=True,
)
generate_text = H2OTextGenerationPipeline(model=model, tokenizer=tokenizer)

with open(sys.argv[1], "r") as handle:
    passage = handle.read()

now = datetime.datetime.now()
res = generate_text(
        f"Summerize the room described in this paragraph:\n{passage}",
    min_new_tokens=2,
    max_new_tokens=1024,
    do_sample=False,
    num_beams=1,
    temperature=float(0.3),
    repetition_penalty=float(1.2),
    renormalize_logits=True
)
print("")
dt = datetime.datetime.now() - now
print(f"Result: {dt.seconds}:{dt.microseconds}")
print(res[0]["generated_text"])

now = datetime.datetime.now()
res = generate_text(
        f"Summerize the room described in this paragraph:\n{passage}",
    min_new_tokens=2,
    max_new_tokens=1024,
    do_sample=False,
    num_beams=1,
    temperature=float(0.3),
    repetition_penalty=float(1.2),
    renormalize_logits=True
)
print("")
dt = datetime.datetime.now() - now
print(f"Result: {dt.seconds}:{dt.microseconds}")
print(res[0]["generated_text"])


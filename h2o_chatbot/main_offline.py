import torch
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

res = generate_text(
    "Why is drinking water so healthy?",
    min_new_tokens=2,
    max_new_tokens=1024,
    do_sample=False,
    num_beams=1,
    temperature=float(0.3),
    repetition_penalty=float(1.2),
    renormalize_logits=True
)
print(res[0]["generated_text"])


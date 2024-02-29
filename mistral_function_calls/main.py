from transformers import AutoModelForCausalLM, AutoTokenizer
import json, datetime

FUNCTION_METADATA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "This function gets the current weather in a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city, e.g., San Francisco"
                    },
                    "state": {
                        "type": "string",
                        "description": "Pick the correct state best on the well know city name."
                    },
                },
                "required": ["city", "state"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_clothes",
            "description": "This function provides a suggestion of clothes to wear based on the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "temperature": {
                        "type": "string",
                        "description": "The temperature, e.g., 15 C or 59 F"
                    },
                    "condition": {
                        "type": "string",
                        "description": "The weather condition, e.g., 'Cloudy', 'Sunny', 'Rainy'"
                    }
                },
                "required": ["temperature", "condition"]
            }
        }
    }
]

messages = [
    {
        "role": "function_metadata",
        "content": json.dumps(FUNCTION_METADATA)
    },
    {
        "role": "user",
        "content": "Whats the weather link in Seattle Wa this weekend?"
    },
]


device = "cuda"
#device = "cpu"

# Load up the model and teh tokenzier
model = AutoModelForCausalLM.from_pretrained('Trelis/Mistral-7B-Instruct-v0.2-function-calling-v3', trust_remote_code=True, load_in_8bit=True, device=device)#, torch_dtype=torch.float16)
#model.half()
#model.to(device)

tokenizer = AutoTokenizer.from_pretrained("Trelis/Mistral-7B-Instruct-v0.2-function-calling-v3", trust_remote_code=True)

now = datetime.datetime.now()

# Tokenize, run the module and decode
model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)

print()
print(f"Generated {datetime.datetime.now() - now}")
print(decoded[0])

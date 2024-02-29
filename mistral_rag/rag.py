import os
import torch, datetime
from transformers import (
  AutoTokenizer, 
  AutoModelForCausalLM, 
  BitsAndBytesConfig,
  pipeline,
  AutoConfig
)

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders import AsyncChromiumLoader, WebBaseLoader

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain

import nest_asyncio

def inline_vectorstore( url ):
    loader = WebBaseLoader(url)
    # Split documents

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000,
                                                   chunk_overlap=20)
    documents = text_splitter.split_documents(loader.load())

    # Load chunked documents into the FAISS index
    db = FAISS.from_documents(documents, 
                              HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))

    return db.as_retriever()


device = "cuda"  # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

#messages = [
#    {"role": "user", "content": "What is your favourite condiment?"},
#    {"role": "assistant",
#     "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
#    {"role": "user", "content": "Do you have mayonnaise recipes?"}
#]

#encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

#model_inputs = encodeds.to(device)
#model.to(device)

#generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
#decoded = tokenizer.batch_decode(generated_ids)
#print(decoded[0])

#return tokenizer, model


import nest_asyncio
nest_asyncio.apply()

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=1000,
    do_sample=True,
    device="cuda"
)

mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

retriever = inline_vectorstore( 'https://www.gutenberg.org/cache/epub/84/pg84.txt')

# Create prompt template
prompt_template = """
### [INST] Instruction: Answer the question based the excerpt from this book. Here is context to help:

{context}

### QUESTION:
{question} [/INST]
 """

# Create prompt from prompt template 
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Create llm chain 
llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

rag_chain = ( 
 {"context": retriever, "question": RunnablePassthrough()}
    | llm_chain
)

print()
print("Starting the actual query")
print()
#for query in ["Where do we first meet Frankenstein's monster?", "How does Frankenstein feel about his monster?"]:
for query in ["What were the rain drops described as?", "What is one character name mentioned?"]:
    now = datetime.datetime.now()
    result = rag_chain.invoke( query )

    print(query)
    print(result['text'])
    print(datetime.datetime.now() - now)
    print()

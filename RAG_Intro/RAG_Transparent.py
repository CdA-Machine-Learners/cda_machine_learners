# load required library
import torch
import datetime
import chromadb
import PyPDF2


import nltk
nltk.download('punkt')  # Sentence tokenizer
from nltk.tokenize import sent_tokenize

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

chroma_client = chromadb.Client()

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading the model...", end='')
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",
                                             quantization_config=quantization_config)
print("done.")

#print("Converting the model to half precision...", end='')
#model.half()  # Convert the model to half precision
#print("done.")

#print("Moving the model to the GPU...", end='')
#model.to('cuda')  # Move the model to the GPU
#print("done.")

print("Ensure the model is on the GPU...")
device = next(model.parameters()).device
print("Model is on device:", device)

print("Loading the tokenizer...", end='')
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.pad_token = tokenizer.eos_token
print("done.")

#model_kwargs = {'device': 'cuda'}
#embeddings = HuggingFaceEmbeddings(model_kwargs=model_kwargs)

model.config.pad_token_id = model.config.eos_token_id

def split_into_chunks(text, sentences_per_chunk=5):
    """
    Split the text into chunks, each containing a specified number of sentences.
    :param text: The input text to split.
    :param sentences_per_chunk: Number of sentences per chunk.
    :return: A list of text chunks.
    """
    sentences = sent_tokenize(text)
    chunks = [' '.join(sentences[i:i + sentences_per_chunk]) for i in range(0, len(sentences), sentences_per_chunk)]
    return chunks

def load_pdf_into_db(pdf_path):
    # Load the PDF file
    #loader = PyPDFLoader(pdf_link, extract_images=False)
    
    #pages = loader.load_and_split()

    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)
        
        # Initialize an empty list to hold the text of each page
        pages_text = ""

        for page_number in range(min(10, len(pdf_reader.pages))):
            page = pdf_reader.pages[page_number]
            # Process the page
            pages_text += page.extract_text()

    chunks = split_into_chunks(pages_text, sentences_per_chunk=10)
    
    inputs = tokenizer(chunks, padding=True, truncation=True, return_tensors="pt")
    inputs.to('cuda')

    collection=chroma_client.get_or_create_collection(name="test_index")

    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # This is a tuple of hidden states from all layers

        # To get the last layer's hidden states, you can do:
        last_hidden_states = hidden_states[-1]

        # Then you can average these last hidden states as you were attempting to do
        embeddings = last_hidden_states.mean(dim=1)

        # Store data into database
        # TODO: Add metadatas here:
        collection.add(embeddings=embeddings,
                       documents=chunks,
                       ids=[])

    results = collection.query(
        query_texts=["Can you describe the character Captain Ahab in Moby Dick?"],
        n_results=2
    )

    print(results)
    #db=Chroma.from_documents(chunks,embedding=embeddings,persist_directory="test_index")
    #db.persist()

def load_text_into_db(text_file_link):
    # Load the text file
    with open(text_file_link, 'r', encoding='utf-8') as file:
        text = file.read()

    # Wrap the text in an object with a 'page_content' attribute
    class Page:
        def __init__(self, content):
            self.page_content = content
            self.metadata = {
                'page': 1,
                'source': text_file_link
            }

    # Create a Page object containing the entire text
    page = Page(text)
    
    # Now, consider this Page object as the content for 'pages'
    pages = [page]

    # Split data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(pages)

    # Store data into database
    db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="test_index")
    db.persist()

#load_pdf_into_db("source_docs/frankenstein.pdf")
#load_pdf_into_db("source_docs/gullivers_travels.pdf")
load_pdf_into_db("source_docs/moby_dick.pdf")
#load_pdf_into_db("source_docs/tale_of_two_cities.pdf")
#load_pdf_into_db("source_docs/treasure_island.pdf")

#load_text_into_db("source_docs/Frankenstein.txt")
#load_pdf_into_db("source_docs/frankenstein.pdf")
#load_pdf_into_db("https://journalofbigdata.springeropen.com/counter/pdf/10.1186/s40537-014-0007-7.pdf")

# Load the database
vectordb = Chroma(persist_directory="test_index", embedding_function = embeddings)

# Load the retriver
# set k to 3 means retrieve top 3 results
retriever = vectordb.as_retriever(search_kwargs = {"k" : 15})
qna_prompt_template="""

{context}

"""


# Start an input loop
while True:
    # Get input prompt from the user
    prompt = input(f"\033[1m" + "Prompt: " + "\033[0m")

    # Check if the user wants to exit
    if prompt.lower() == 'exit':
        break

    now = datetime.datetime.now()

    # Take the user input and call the function to generate output
    context = retriever.get_relevant_documents(prompt)

    builtPrompt = "### [INST] Instruction: You will be provided with a prompt or question as well as excerpts from different pieces of literature. "
    builtPrompt += "These excerpts are retrieved by a retriever model that finds texts from literature related to the prompt. "
    builtPrompt += "You are to respond as if you are the one who retrieved the excerpts. Your response represents the entire retrieval and reponse process. "
    builtPrompt += "Your task is to review the excerpts from various books and do your best to answer the prompt or question, citing the page numbers that are provided with each excerpt. "
    builtPrompt += "If the excerpts do not contain enough information to answer the question properly, then you must return 'Not enough information.'\n\n"
    
    for document in context:
        builtPrompt += f"{document.metadata['source']} (page {document.metadata['page']}):\n"
        builtPrompt += f"{document.page_content}\n\n"
    
    builtPrompt += f"### Question: {prompt} [/INST]"

    print(f"Time to retrieve related source materials: {datetime.datetime.now() - now}")
    now = datetime.datetime.now()

    #answer = (chain.invoke({"input_documents": context, "question": question}, return_only_outputs=True))['output_text']

    encoded_input = tokenizer.encode(builtPrompt, return_tensors='pt').to('cuda')

    #print(builtPrompt)

    # Generate a sequence of tokens in response to the input prompt
    generated_ids = model.generate(
        encoded_input,
        max_new_tokens=1000,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,  # Adjust temperature for creativity/diversity of the response
        top_k=50,  # Adjust top_k for top-k sampling
        top_p=0.95,  # Adjust top_p for nucleus sampling
        do_sample=True,  # Enable sampling to generate diverse responses
        num_return_sequences=1  # Number of sequences to generate
    )

    # Decode the generated ids to text
    decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print(f"Time to generate response: {datetime.datetime.now() - now}")

    print()
    print("\033[1m" + "Assistant: " + "\033[0m" + f"{decoded_output}")
    print()
    #print("Here is the relevant source material:")
    #for document in context:
        #print(f"{document.metadata['source']} (page {document.metadata['page']}):")
        #print(f"{document.page_content}")
        #print()


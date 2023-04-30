'''

Inspired by BabyAGI

Usage:

1. Deps

  pip install openai numpy

2. OpenAI Key

  # .secrets.yml
  OPENAI_API_KEY: "sk-..."
  OPENAI_ORGANIZATION: "org-..."

3. Run it

  python babyagi.py "How can I make a successful AI Meetup?"


TODO:

  * add web search tool via SerpAPI

  * add `requests` tool

  * Help it terminate. Right now it runs and just keeps filling up its task queue

'''

from collections import deque
from typing import Dict, List
import numpy as np
import openai
import os
import sys
import time
import yaml


##################################################
# Inputs

if len(sys.argv) == 2:
    OBJECTIVE = sys.argv[1]
else:
    OBJECTIVE = "How can I make a successful AI Meetup?"

YOUR_FIRST_TASK = "Develop a task list."


##################################################
# Params

# LLM_MODEL = 'gpt-3.5-turbo'
LLM_MODEL = 'text-davinci-003'
OPENAI_TEMPERATURE = 0.8
EMBEDDING_LENGTH = 1536

# openai
secrets_path = '.secrets.yml'
with open(secrets_path, 'r') as f:
    raw = f.read()
    secrets = yaml.safe_load(raw)
openai.organization = secrets['OPENAI_ORGANIZATION']
openai.api_key = secrets['OPENAI_API_KEY']


##################################################
# UTIL

def fmtstr(text,
         color=None,
         color_256=None,
         bg_color=None,
         bg_color_256=None,
         bold=False,
         underline=False,
         ):
    ''' format strings with color and boldness for terminal output '''
    color_codes = {
        'black': '30',
        'red': '31',
        'green': '32',
        'yellow': '33',
        'blue': '34',
        'magenta': '35',
        'cyan': '36',
        'white': '37'
    }

    if color_256 is not None and 0 <= color_256 <= 255:
        text = f"\033[38;5;{color_256}m{text}"
    elif color and color.lower() in color_codes:
        text = f"\033[{color_codes[color.lower()]}m{text}"

    if bg_color_256 is not None and 0 <= bg_color_256 <= 255:
        text = f"\033[48;5;{bg_color_256}m{text}"
    elif bg_color and bg_color.lower() in color_codes:
        text = f"\033[{int(color_codes[bg_color.lower()]) + 10}m{text}"

    if bold:
        text = f"\033[1m{text}"

    if underline:
        text = f"\033[4m{text}"

    # Reset all attributes at the end of the string
    text = f"{text}\033[0m"

    return text

def dblinput(prompt):
    ''' `input`, but only returns after 2 newlines '''
    result = []
    print(prompt, end="")
    while True:
        current_input = input()
        if len(result) > 1 and result[-2] == '' and result[-1] == '':
            break
        result.append(current_input)
    return "\n".join(result).strip()

def llm(
    prompt: str,
    model: str = LLM_MODEL,
    temperature: float = OPENAI_TEMPERATURE,
    max_tokens: int = 100,
):
    ''' make calls to OpenAI's API, recover from certain errors '''
    while True:
        try:
            # Completion API
            if not model.lower().startswith("gpt-"):

                response = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return response.choices[0].text.strip()

            #  Chat API
            else:
                messages = [{"role": "system", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=1,
                    stop=None,
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            error_msg = {
                'RateLimitError': "The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again.",
                'Timeout': "OpenAI API timeout occurred. Waiting 10 seconds and trying again.",
                'APIError': "OpenAI API error occurred. Waiting 10 seconds and trying again.",
                'APIConnectionError': "OpenAI API connection error occurred. Check your network settings, proxy configuration, SSL certificates, or firewall rules. Waiting 10 seconds and trying again.",
                'InvalidRequestError': "OpenAI API invalid request. Check the documentation for the specific API method you are calling and make sure you are sending valid and complete parameters. Waiting 10 seconds and trying again.",
                'ServiceUnavailableError': "OpenAI API service unavailable. Waiting 10 seconds and trying again.",
            }
            if e in error_msg:
                print(e)
                print(error_msg[e.__class__.__name__])
                print(e.__class__.__name__)
                time.sleep(10)
        else:
            break


##################################################
# VECTOR DB

def cosine_similarity(vector, matrix):
    ''' cosine similarity of `vector` to each row of `matrix` '''
    # Normalize the input vector
    vector_norm = np.linalg.norm(vector)
    if vector_norm == 0:
        raise ValueError("Input vector has zero magnitude.")
    normalized_vector = vector / vector_norm

    # Normalize each row in the input matrix
    matrix_norms = np.linalg.norm(matrix, axis=1).reshape(-1, 1)
    zero_norm_rows = np.isclose(matrix_norms, 0)
    if np.any(zero_norm_rows):
        raise ValueError("One or more rows in the input matrix have zero magnitude.")
    normalized_matrix = matrix / matrix_norms

    # Calculate the cosine similarity between the vector and each row in the matrix
    cosine_similarities = np.dot(normalized_matrix, normalized_vector)

    return cosine_similarities


class Index:
    def __init__(self, embedding_len):
        self.embedding_len = embedding_len
        self.db = {}
        self.np_index = None # vectors into db
        self.np_embedding = None

    def upsert_multiple(self, xs):
        '''
        Args:
          xs: [(id, embedding, metadata)]
        '''
        for i, e, m in xs:
            self.db[i] = (e, m)
        self.rebuild_index()

    def rebuild_index(self):
        '''
        If new things have been added to `self.db`, the index needs to be rebuilt
        out of it.
        '''
        ix_embs = [(i, x[0]) for i, x in self.db.items()] # trim off metadata, keep db-ixs
        self.np_index = np.array([x[0] for x in ix_embs])
        self.np_embedding = np.array([x[1] for x in ix_embs])

    def query(self, embedding, top_k : int):
        '''
        Get the `top_k` nearest neighbors of `embedding`.
        '''
        if self.np_embedding is None:
            return []

        sim = cosine_similarity(embedding, self.np_embedding)
        ixs = list(reversed(np.argsort(sim)))[:top_k]
        out = []
        for i in ixs:
            db_i = self.np_index[i]
            out.append({
                'score': sim[i],
                'index': db_i,
                'embedding': self.db[db_i][0],
                'metadata': self.db[db_i][1],
            })
        return out


##################################################
#


# Task list
task_list = deque([])


index = Index(EMBEDDING_LENGTH)

def add_task(task: Dict):
    task_list.append(task)

def get_ada_embedding(text):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]


##############################
# TASK CREATION

task_creation_prompt = """
You are a task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective},
The last completed task has the result: {result}.
This result was based on this task description: {task_description}.
These are incomplete tasks: {task_list}.
Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks.
Return the tasks as an array.
""".strip()

def task_creation_agent(objective: str, result: Dict, task_description: str, task_list: List[str]):
    prompt = task_creation_prompt.format(
        objective=objective,
        result=result,
        task_description=task_description,
        task_list=', '.join(task_list),
    )
    response = llm(prompt)
    new_tasks = response.split('\n')
    return [{"task_name": task_name} for task_name in new_tasks]


##############################
# PRIORITIZATION

prioritization_prompt = """
You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: {task_names}.
Consider the ultimate objective of your team:{OBJECTIVE}.
Do not remove any tasks.

Return the result as a numbered list, like:
#. First task
#. Second task
Start the task list with number {next_task_id}.
""".strip()

def prioritization_agent(this_task_id:int):
    global task_list
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id)+1
    prompt = prioritization_prompt.format(
        task_names=task_names,
        OBJECTIVE=OBJECTIVE,
        next_task_id=next_task_id,
    )
    response = llm(prompt)
    new_tasks = response.split('\n')
    task_list = deque()
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            task_list.append({"task_id": task_id, "task_name": task_name})


##############################
# EXECUTION

execution_prompt = """
You are an AI who performs one task based on the following objective: {objective}
.
Take into account these previously completed tasks: {context}

Your task: {task}

Response:
""".strip()

def execution_agent(objective:str, task: str) -> str:
    #context = context_agent(index="quickstart", query="my_search_query", n=5)
    context=context_agent(query=objective, n=5)
    print("\n*******RELEVANT CONTEXT******\n")
    print(context)
    prompt = execution_prompt.format(
        objective=objective,
        context=context,
        task=task
    )
    return llm(prompt)


##############################
# CONTEXT

def context_agent(query: str, n: int):
    query_embedding = get_ada_embedding(query)
    results = index.query(query_embedding, top_k=n) # , include_metadata=True)
    #print("***** CONTEXT RESULTS *****")
    #print(results)
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    return [(str(item['metadata']['task'])) for item in sorted_results]


##################################################
# RUN

print(fmtstr("\n*****OBJECTIVE*****\n", color='blue', bold=True))
print(OBJECTIVE)

# Add the first task
first_task = {
    "task_id": 1,
    "task_name": YOUR_FIRST_TASK
}
add_task(first_task)

# Main Loop
task_id_counter = 1
while True:
    if task_list:
        # Print the task list
        print(fmtstr("\n*****TASK LIST*****\n", color='magenta', bold=True))
        for t in task_list:
            print(str(t['task_id'])+": "+t['task_name'])

        # Step 1: Pull the first task
        task = task_list.popleft()
        print(fmtstr("\n*****NEXT TASK*****\n", color='green', bold=True))
        print(str(task['task_id'])+": "+task['task_name'])

        # Send to execution function to complete the task based on the context
        result = execution_agent(OBJECTIVE, task["task_name"])
        this_task_id = int(task["task_id"])
        print(fmtstr("\n*****TASK RESULT*****\n", color='red', bold=True))
        print(result)

        # Step 2: Enrich result and store in Pinecone
        enriched_result = {'data': result}  # This is where you should enrich the result if needed
        result_id = f"result_{task['task_id']}"
        vector = enriched_result['data']  # extract the actual result from the dictionary
        index.upsert_multiple([(result_id,
                        get_ada_embedding(vector),
                        {
                            "task": task['task_name'],
                            "result": result
                        })])

    # Step 3: Create new tasks and reprioritize task list
    new_tasks = task_creation_agent(OBJECTIVE,
                                    enriched_result,
                                    task["task_name"],
                                    [t["task_name"] for t in task_list]
                                    )

    for new_task in new_tasks:
        task_id_counter += 1
        new_task.update({"task_id": task_id_counter})
        add_task(new_task)
    prioritization_agent(this_task_id)

    time.sleep(1)  # Sleep before checking the task list again


##########
# DEBUG

for k, v in index.db.items():
    print('##################################################')
    print(k)
    print(v[1]['task'])
    print(v[1]['result'])

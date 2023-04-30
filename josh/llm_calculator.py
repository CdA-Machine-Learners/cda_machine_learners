'''

Augment an LLM with a calculator

1.Human: Ask a question

2. Either:
  A. answer question
  B. emit special syntax to use a calculator, then with that context, generate an answer
    CALCULATE: 2 + 4

'''

from collections import deque
from typing import Dict, List
import numpy as np
import openai
import os
import sys
import time
import yaml
import re

##################################################
# Inputs

if len(sys.argv) == 2:
    OBJECTIVE = sys.argv[1]
else:
    # OBJECTIVE = 'If I have two carrots, and someone hands me 5 thousand carrots, how many carrots do I have?'
    # OBJECTIVE = 'A person is 53 years old. What is their age raised to the quarter power?'
    # OBJECTIVE = 'What is the meaning of life?'
    # OBJECTIVE = 'What is the sum of 1 and two, raised to the 4th power, then divided by 9?'
    # OBJECTIVE = 'How many licks does it take to get to the center of a tootsie pop?'
    # OBJECTIVE = 'How many letters are in this word: abcdefghij?'
    # OBJECTIVE = 'Why do birds suddenly appear?'
    # OBJECTIVE = 'Tell me about ribosomes.'
    # OBJECTIVE = 'How many ribosomes does a cell have?'
    OBJECTIVE = 'I was given fifteen yeast cells last tuesday. They doubled. Then doubled again. How many do I have now?'


##################################################
# Params

# LLM_MODEL = 'gpt-3.5-turbo'
LLM_MODEL = 'text-davinci-003'
OPENAI_TEMPERATURE = 0.9

# openai
if os.path.split(os.getcwd())[1] == 'sandbox': # for emacs repl
    secrets_path = '../.secrets.yml'
else:
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


##############################
# CALCULATOR AGENT

calculator_prompt = '''
You are an oracle machine that knows everything, and you know how to use the output of a calculator for properly answering questions. You have been given the following objective, and the following context:

The question:
  {objective}

The calculator input:
  {calculation_str}


The calculator output:
  {calculation}

The correct answer to the question, incorporating the calculation:
'''.strip()

def calculator_clean(x:str) -> str:
    x = re.sub(r'[^0-9\.\+\-\*\/\^\%\(\)]', '', x)
    x = re.sub(r'\^', '**', x) # exponent
    return x

def calculator_agent(objective: str, calculation_str: str) -> str:
    ''' Returns calculator results from SerpAPI '''
    # 1. Do the calculation
    clean = calculator_clean(calculation_str)
    calculation = eval(clean)

    # 2. Run the LLM with the result
    prompt = calculator_prompt.format(
            objective=objective,
            calculation_str=calculation_str,
            calculation=calculation,
        )
    print('CALCULATION PROMPT')
    print(fmtstr(prompt, color='red'))
    response = llm(prompt)
    return response

##############################
# MAIN AGENT

main_prompt = '''
You are an oracle machine that knows everything except how to calculate numbers. You are good at answering questions knowledgably, and with brevity. You answer questions, unless they require calculations. If a question deals with numbers, instead of answering the question, you will output a snippet of syntax that a calculator could use, and only that syntax instead of a full answer.

##########
# Example 1:

INPUT:
  How many apples do I have if I start with one and then I am given 2?

YOUR RESPONSE:
  CALCULATE: 1 + 2


##########
# Example 2:

INPUT:
  What is 1 thousand raised to the half power?

YOUR RESPONSE:
  CALCULATE: 1000^0.5


##########
# Example 3:

INPUT:
  What is the color of the sky?

YOUR RESPONSE:
  blue


##########
# Example 4:

INPUT:
  Where does a cell keep its DNA?

YOUR RESPONSE:
  in its nucleus


##########
# Example 5:

INPUT:
  What is 1 + 1?

YOUR RESPONSE:
  CALCULATE: 1 + 1

##########
# Actual Question

You must not attempt to answer questions that deal with calculations. You must output the syntax `CALCULATE: ...`.

If the question does not deal with numbers or quantities, just answer it directly.

INPUT: {objective}
'''.strip()

def main_agent(objective: str) -> str:
    # 1. First pass
    first_pass = llm(
        main_prompt.format(objective=objective)
    )
    # 2. Use calculator if necessary
    calc_strs = re.findall(r'CALCULATE: (.*)$', first_pass)
    if len(calc_strs) > 0:
        print(f'needs calculator: {calc_strs[0]}')
        return calculator_agent(objective, calc_strs[0])
    else:
        return first_pass


##################################################
# RUN

print(fmtstr("\n*****OBJECTIVE*****\n", color='blue', bold=True))
print(OBJECTIVE)

print(fmtstr("\n*****RESULT*****\n", color='green', bold=True))
print(main_agent(OBJECTIVE))

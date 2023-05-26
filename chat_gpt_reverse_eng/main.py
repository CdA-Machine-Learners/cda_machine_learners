#!/usr/bin/python3

from revChatGPT.V3 import Chatbot

import sys, os

api_key = os.getenv('API_KEY')

if len(sys.argv) < 2 or api_key is None:
    print('Goto: https://chat.openai.com/api/auth/session')
    print('export API_KEY="xxxxxx"')
    print("Usage: python main.py <PROMPT>")
    exit(1)

chatbot = Chatbot(api_key=api_key)

# Is it a file? Can we load it up?
prompt = ' '.join(sys.argv[1:])
try:
    with open(prompt) as handle:
        prompt = handle.read()
except:
    pass

# Do it up!
print("")
for word in chatbot.ask_stream( prompt ):
    sys.stdout.write(word)
    sys.stdout.flush()

print("")
print("")

# Libraries
from langchain.prompts.prompt import PromptTemplate
from langchain import ConversationChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
import re
import os

# Function to Split questions
def questions_brekdown(ai_response):
    pattern = r'^(?:\d+\.|\-|\*).*?(\n(?![\d+\.\-\*]).*|$)'
    matches = re.finditer(pattern, ai_response, re.MULTILINE | re.DOTALL)
    split_items = [match.group(0).strip() for match in matches]
    return split_items

# Load Open AI Key
os.environ['OPENAI_API_KEY'] = "OPEN_AI_KEY"

# Load LLMs
gpt3 = ChatOpenAI(temperature=0.6,model='gpt-3.5-turbo-16k')

# Retrieve AI's supporting questions
print("Welcome, I am an assistant to develop feedback for inquiries, just send me your request")
request = input("You: ")

# Template
template = """Act like you are a chatgpt prompt builder.
The user is about to send you a query, your task is to numerically list questions to ask the user, in order to clarify and know if the request was well understood

Current conversation:
{history}
Human: {input}
Prompt Builder:  
"""

PROMPT = PromptTemplate(input_variables=["history","input"], template=template)

# Set first ConversationChain
conversation = ConversationChain(
  prompt=PROMPT,
  llm=gpt3,
  verbose=False,
  memory=ConversationBufferMemory(ai_prefix="Prompt Builder"))

ai_response = conversation(request)

ai_questions_list = questions_brekdown(ai_response["response"])
print("\nHere are some questions to help building a better prompt\nPlease, answer one question at a time, I will be taking notes\n")
number_of_questions = len(ai_questions_list)
print("There are "+str(number_of_questions)+" questions")

# Loop thru questions, getting user input
user_answers = ""
ai_questions = ""
question_number = 1
for question in ai_questions_list:
    print("\n"+question)
    ai_questions += question+ "\n"
    user_input = input("You: ")
    user_answers += str(question_number) + ". " + user_input + "\n"
    question_number += 1

# Send answers
answers_prompt = conversation("I have answered your questions suggested to improve the original inquiry, please see below\n"+user_answers+"\nsummarize the chat so far, please")
#don't do anything yet, more instructions to come

print("\n"+answers_prompt["response"])

# Request final prompt
final_prompt = conversation("Correct, now use all the information so far and return me a prompt that I can use to ask ChatGPT, the prompt should contain details to improve my initial request")

print("\nGreat, I have all I need\n")

print(final_prompt['response'])

try:
  prompt_to_gpt4 = re.findall(r'"([^"]*)"', final_prompt['response'])
  prompt_to_gpt4 = prompt_to_gpt4[0]
except:
  prompt_to_gpt4 = final_prompt['response']

#print(prompt_to_gpt4)
#ADD INFORMATION LOOP TO SEE IF THE USER IS HAPPY WITH FINAL OUTCOME

# Ask final prompt to GPT4
# Load GPT4
gpt4 = OpenAI(temperature=0.6, model="gpt-4")

# Initiate
template = """Question: {question}
Answer: """
prompt = PromptTemplate(
        template=template,
    input_variables=['question']
)

# user question
question = prompt_to_gpt4

llm_chain = LLMChain(
  llm=gpt4,
  prompt=prompt)

print("### GPT4 Output ###")
print(llm_chain.run(question))

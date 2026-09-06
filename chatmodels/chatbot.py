from dotenv import load_dotenv
load_dotenv()

# from langchain_mistralai import ChatMistralAI
# from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
# model = ChatMistralAI(model = "mistral-small-2506", temperature=0.9)


# messages = [
#     SystemMessage(content="you are a funny AI agent")
# ]

# print("--------------- welcome, type 0 to exit the application ---------------")
# while(True):

#     prompt = input("you : ")
#     messages.append(HumanMessage(content=prompt))
#     if(prompt == "0"): 
#         break

#     response  = model.invoke(messages)
#     messages.append(AIMessage(content=response.content))
#     print("Bot :", response.content)

#no tokens left

#using deepseek model from huggingface
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731"
)
model = ChatHuggingFace(llm=llm)

print("Choose your AI mode")
print("press 1 for Angry mode")
print("press 2 for funny mode")
print("press 3 for sad mode")
print("press 4 for normal mode")
choice = int(input("Enter mode : "))
if(choice == 1):
    mode = "you are an angry AI agent. You respond aggressively and impatiently."
elif(choice == 2):
    mode = "you are a very funny AI agent. You respond with humor and jokes."
elif(choice == 3):
    mode = "You are sad AI agent. You respond with sadness."
elif(choice == 4):
    mode = "You are normal AI agent. You respond normally as you do."

messages = [
    SystemMessage(content=mode)
]

print("--------------- welcome, type 0 to exit the application ---------------")

while(True):
    prompt = input("you : ")
    messages.append(HumanMessage(prompt))
    if(prompt == "0"):
        break

    response = model.invoke(messages)
    messages.append(AIMessage(content = response.content))
    print("Bot :", response.content)
print(messages)
from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(model = "mistral-small-2506", temperature=0.9)

messages = []

print("--------------- welcome, type 0 to exit the application ---------------")
while(True):

    prompt = input("you : ")
    messages.append(prompt)
    if(prompt == "0"): 
        break

    response  = model.invoke(messages)
    messages.append(response.content)
    print("Bot :", response.content)

print(messages)
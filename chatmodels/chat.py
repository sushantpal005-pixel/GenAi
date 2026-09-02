from dotenv import load_dotenv

load_dotenv()   #to use api keys stored in .env file

#using gpt
#using init_chat_model
#from langchain.chat_models import init_chat_model
#model = init_chat_model("gpt-4.1")

#using model class
#from langchain_openai import ChatOpenAI
#model = ChatOpenAI(model = "gpt-5")

#using gemini model 3.5 bcoz it is free
#using init_chat_model

# from langchain.chat_models import init_chat_model
# model = init_chat_model(
#     model ="gemini-3.5-flash-lite", 
#     model_provider="google_genai"
# )
# response = model.invoke("give me a paragraph on Machine learning")
# print(response.content)

#using model class 

# from langchain_google_genai import ChatGoogleGenerativeAI
# model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")
# response = model.invoke("give me a paragraph on Machine Learning")
# print(response.content)

#using mistralai 
#using model class

from langchain_mistralai import ChatMistralAI
model = ChatMistralAI(model="mistral-small-2603", temperature=0.9, max_tokens=20)
response = model.invoke("write a poem on AI")
print(response.content)


#from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
#using gemini embedding model bcoz it is free
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    
)

texts = [
    "Hello this is Sushant",
    "I am Full stack web developer",
    "Currently learning GenAI"
]
vector = embeddings.embed_documents(texts)
print(vector)
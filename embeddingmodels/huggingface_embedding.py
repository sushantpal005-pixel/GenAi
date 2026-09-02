from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name= "sentence-transformers/all-MiniLM-L6-v2"
)


texts = [
    "Hello this is Sushant",
    "I am Full stack web developer",
    "Currently learning GenAI"
]

vector = embedding.embed_documents(texts)

print(vector)
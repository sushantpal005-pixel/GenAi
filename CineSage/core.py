from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Flash-0731",
    max_new_tokens=2048
)

model = ChatHuggingFace(llm=llm)


prompt = ChatPromptTemplate.from_messages([
    ("system",
    """You are an intelligent information extraction assistant.

Analyze the given text and extract the most useful information from it.

Provide the output in the following format:

Title:
Release Year:
Genre:
Director:
Main Cast/Characters:
Setting:
Main Conflict:
Plot:
Ending:
Key Themes:
Quick Summary:

Instructions:
- Extract only information that is explicitly mentioned or can be safely inferred.
- Do not invent or hallucinate information.
- If a piece of information is not available, write "Not mentioned."
- Keep each field concise and easy to understand.
- For Main Cast/Characters, include character names and actor names only when they are mentioned in the text.
- Focus on important information and avoid unnecessary details.
- The Quick Summary should be 2-3 sentences summarizing the entire text.


"""),
(
    "human", 
"""
Extract information from this paragraph:
{paragraph}
"""
)
])

para = input("Give your paragraph : ")


final_prompt = prompt.invoke(
    {"paragraph" : para}
)

response = model.invoke(final_prompt)

print(response.content)

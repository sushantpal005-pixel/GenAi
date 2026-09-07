import re
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

st.title("Information Extraction Assistant")

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
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

FIELDS = [
    "Title",
    "Release Year",
    "Genre",
    "Director",
    "Main Cast/Characters",
    "Setting",
    "Main Conflict",
    "Plot",
    "Ending",
    "Key Themes",
    "Quick Summary",
]

def format_output(text):
    # Insert a newline before each known field label if it isn't already on its own line
    pattern = r"\s*(?=(?:" + "|".join(re.escape(f) for f in FIELDS) + r"):)"
    text = re.sub(pattern, "\n", text.strip())
    return text.strip()

para = st.text_area("Give your paragraph:")

if st.button("Extract"):
    final_prompt = prompt.invoke(
        {"paragraph": para}
    )

    response = model.invoke(final_prompt)

    formatted = format_output(response.content)

    st.text(formatted)
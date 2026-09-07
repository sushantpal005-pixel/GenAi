import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()


# ---------- Schema ----------
class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str


parser = PydanticOutputParser(pydantic_object=Movie)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
Extract movie information from the paragraph.
{format_instructions}
"""),
    ("human", "{paragraph}")
])


# ---------- Cache the model so it isn't rebuilt on every rerun ----------
@st.cache_resource
def load_model():
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
        max_new_tokens=2048
    )
    return ChatHuggingFace(llm=llm)


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Movie Info Extractor", page_icon="🎬", layout="centered")

st.title("🎬 Movie Info Extractor")
st.write("Paste a paragraph describing a movie, and this app will extract structured details from it.")

para = st.text_area(
    "Paragraph about a movie",
    height=200,
    placeholder="e.g. Inception is a 2010 sci-fi thriller directed by Christopher Nolan, starring Leonardo DiCaprio and Elliot Page..."
)

extract_clicked = st.button("Extract Movie Info", type="primary", disabled=not para.strip())

if extract_clicked:
    with st.spinner("Loading model and extracting..."):
        try:
            model = load_model()

            final_prompt = prompt.invoke({
                "paragraph": para,
                "format_instructions": parser.get_format_instructions()  # note: call it, don't pass the method itself
            })

            response = model.invoke(final_prompt)

            # Try to parse into the structured Movie object
            try:
                movie = parser.parse(response.content)
            except Exception:
                movie = None

        except Exception as e:
            st.error(f"Something went wrong while calling the model: {e}")
            movie = None
            response = None

    if movie is not None:
        st.success("Extraction successful!")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Release Year", movie.release_year if movie.release_year else "Unknown")
        with col2:
            st.metric("Rating", movie.rating if movie.rating is not None else "N/A")

        st.subheader(movie.title)
        st.write(f"**Director:** {movie.director or 'Unknown'}")
        st.write(f"**Genre(s):** {', '.join(movie.genre) if movie.genre else 'Unknown'}")
        st.write(f"**Cast:** {', '.join(movie.cast) if movie.cast else 'Unknown'}")
        st.write("**Summary:**")
        st.write(movie.summary)

        with st.expander("Raw JSON"):
            st.json(movie.model_dump())

    elif response is not None:
        st.warning("Couldn't parse the model's output into the structured format. Showing raw response instead:")
        st.code(response.content)

st.divider()
st.caption("Powered by LangChain + HuggingFace (DeepSeek-V4-Flash) + Pydantic structured output.")
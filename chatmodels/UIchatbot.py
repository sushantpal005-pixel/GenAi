import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

st.set_page_config(page_title="AI Mode Chat Bot", page_icon="🤖", layout="centered")

MODES = {
    "😡 Angry mode": "you are an angry AI agent. You respond aggressively and impatiently.",
    "😂 Funny mode": "you are a very funny AI agent. You respond with humor and jokes.",
    "😢 Sad mode": "You are sad AI agent. You respond with sadness.",
    "🙂 Normal mode": "You are normal AI agent. You respond normally as you do.",
}

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #1f1c2c 0%, #928dab 100%);
    }
    .title-text {
        text-align: center;
        color: white;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .subtitle-text {
        text-align: center;
        color: #e0e0e0;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V4-Flash-0731"
    )
    return ChatHuggingFace(llm=llm)


st.markdown('<p class="title-text">🤖 AI Mode Chat Bot</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Choose a personality and start chatting</p>', unsafe_allow_html=True)

if "mode_selected" not in st.session_state:
    st.session_state.mode_selected = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Mode selection screen ----------------
if not st.session_state.mode_selected:
    with st.container(border=True):
        st.subheader("Choose your AI mode")
        choice = st.radio(
            "Select a mode:",
            list(MODES.keys()),
            label_visibility="collapsed",
        )
        if st.button("Start Chat 🚀", use_container_width=True):
            st.session_state.messages = [SystemMessage(content=MODES[choice])]
            st.session_state.selected_mode_label = choice
            st.session_state.mode_selected = True
            st.rerun()

# ---------------- Chat screen ----------------
else:
    model = load_model()

    st.info(f"Active mode: **{st.session_state.selected_mode_label}**")

    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)

    prompt = st.chat_input("you :")

    if prompt:
        st.session_state.messages.append(HumanMessage(prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        response = model.invoke(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=response.content))

        with st.chat_message("assistant"):
            st.markdown(response.content)

    if st.button("🔄 Change mode"):
        st.session_state.mode_selected = False
        st.session_state.messages = []
        st.rerun()
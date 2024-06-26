import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from streamlit_chat import message
import requests

st.set_page_config(
    page_title="Streamlit Chat - Demo",
    page_icon=":robot:"
)


st.header("Streamlit Chat - Demo")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


def generate_response(prompt):
    """Generates a response using LLM."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                 temperature=0.1,
                                 max_tokens=100,
                                 top_p=0.9,
                                 google_api_key="AIzaSyAwFJZW0I1gA_954Wih96vDb3T0b-L9p84")

    return llm.invoke(prompt)


def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = generate_response(user_input).content

    st.session_state.past.append(user_input)

    st.session_state.generated.append(output)

if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        # message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


st.subheader("We are Online👨‍🏭")

def generate_response(prompt):
    """Generates a response using LLM."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                 temperature=0.1,
                                 max_tokens=100,
                                 top_p=0.9,
                                 google_api_key="AIzaSyAwFJZW0I1gA_954Wih96vDb3T0b-L9p84")

    return llm.invoke(prompt).content

# # One Logic to manage the chat conversation between user and assistant vanishing previous messages
# def chat():
#     """
#     Manages a chat conversation between user and assistant.

#     Sets up the initial messages and handles user input.
#     Generates a response using the LLM and updates the messages.
#     """

#     # Get user input and handle it if provided
#     if prompt := st.chat_input("Say something..."):
#         # Display the user's message
#         with st.chat_message("user"):
#             st.write(prompt)

#         # Display the assistant's response
#         with st.chat_message("assistant"):
#             # Show a loading spinner while generating the response
#             with st.spinner("Thinking..."):
#                 # Generate the response using the LLM
#                 response = generate_response(prompt)
#                 st.write(response)

# Second Logic to manage the chat conversation between user and assistant displaying previous messages
def chat():
    """
    Manages a chat conversation between user and assistant.

    Keeps track of the conversation history in the session state.
    """

    # Initialize the conversation history if it doesn't exist
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hello We are Online"}
        ]

    # Get user input and handle it if provided
    if question := st.chat_input("Say something..."):
        # Add user's message to the conversation history
        st.session_state.messages.append({"role": "user", "content": question})

    # Display each message in the conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If the last message is not from the assistant, generate a response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Generate a response using the LLM
                response = generate_response(question)
                st.write(response)
                # Add the assistant's response to the conversation history
                st.session_state.messages.append(
                    {"role": "assistant", "content": response})


if __name__ == '__main__':
    chat()

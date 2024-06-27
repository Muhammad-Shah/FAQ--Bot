import os
import pinecone
import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Pinecone
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from prompt import system_prompt
from pprint import pprint

# Load environment variables from .env file
load_dotenv('.env')

# Access your API key
GOOGLE_API = os.getenv("GOOGLE_API")
HF_API = os.getenv('HF_TOKEN')
PINECONE_API_KEY = os.getenv('PINECONE_API')


def load_data(file_path, jq_schema):
    loader = JSONLoader(
        file_path=file_path,
        jq_schema=jq_schema,
        text_content=False)

    data = loader.load()
    data = [Document(page_content=doc.page_content) for doc in data]
    return data


@st.cache_resource
def create_embeddings(model_name):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

# def create_embeddings(model_name, hf_api_key, texts):
#     # Initialize Hugging Face Inference API embeddings with LangChain
#     embeddings = HuggingFaceInferenceAPIEmbeddings(
#         api_key=hf_api_key,
#         model_name=f"sentence-transformers/{model_name}"
#     )
#     # Generate embeddings for each text
#     results = [embeddings.embed_query(text) for text in texts]
#     return results


@st.cache_resource
def create_chroma_db(file_path, jq_schema, persist_directory, _embeddings):
    if os.path.isdir(os.getcwd() + '/vectorstore'):
        db = Chroma(persist_directory=persist_directory,
                    embedding_function=_embeddings)
    else:
        data = load_data(file_path=file_path, jq_schema=file_path)
        db = Chroma.from_documents(
            data, embeddings, persist_directory=persist_directory)
    return db


# @st.cache_resource
def create_pinecone_db(data, _embeddings):
    # Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create a Pinecone index from documents
    index_name = 'pinecone'
    index = pc.create_index(
        name=index_name,
        service_type=ServiceType.SIMILARITY,
        dimension=1536,  # Assuming embeddings is a numpy array
        repository=Repository.DENSE,
        data=docs_split
    )

    return index


def create_llm(model, temperature, max_tokens, top_p, GOOGLE_API):
    llm = ChatGoogleGenerativeAI(model=model,
                                 temperature=temperature,
                                 max_tokens=max_tokens,
                                 top_p=top_p,
                                 google_api_key=GOOGLE_API)
    return llm


def create_prompt(system_prompt):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{Question}"),
        ]
    )
    return prompt


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(retriever, prompt, llm):
    rag_chain = (
        {"context": retriever | format_docs, "Question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def create_correction_prompt_template(query):
    template = PromptTemplate(
        input_variables=["query"],
        template="Please correct the following user query as a standalone question for grammar and spelling if there are any errors. Otherwise, return the same query: `{query}`",
    )
    return template.format(query=query)


def process_query(rag_chain, query, correction_prompt_template, llm):
    standalone_question = llm.invoke(correction_prompt_template).content
    result = rag_chain.invoke(standalone_question)
    return result


# Efficiently call the functions for RAG
def rag_pipeline(file_path, jq_schema, embeddings, persist_directory, model, temperature, max_tokens, top_p, GOOGLE_API, system_prompt, query):

    # Create the database
    db = create_chroma_db(file_path, jq_schema, persist_directory, embeddings)

    # # Create the database
    # db = create_pinecone_db(data, embeddings)

    # Create the LLM
    llm = create_llm(model, temperature, max_tokens, top_p, GOOGLE_API)

    # Create the prompt
    prompt = create_prompt(system_prompt)

    # Create the RAG chain
    rag_chain = create_rag_chain(db.as_retriever(), prompt, llm)

    # Create correction prompt template
    correction_prompt_template = create_correction_prompt_template(query)

    # Process the query
    result = process_query(
        rag_chain, query, correction_prompt_template, llm)

    return result


# Example usage
# file_path = "data/FAQ.json"
# jq_schema = ".[]"
# model_name = "sentence-transformers/all-mpnet-base-v2"
# persist_directory = "./vectorstore"
# model = "gemini-1.5-flash"
# temperature = 0.1
# max_tokens = 512
# top_p = 0.9
# query = "How can I create an account?"

# result = rag_pipeline(file_path, jq_schema, model_name,
#                       persist_directory, model, temperature,
#                       max_tokens, top_p, GOOGLE_API, system_prompt, query)
# pprint(result)

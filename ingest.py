import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from pprint import pprint
from prompt import system_prompt
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Pinecone
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from dotenv import load_dotenv
import streamlit as st
import os
from langchain_chroma import Chroma

# Load environment variables from .env file
load_dotenv('.env')

# Access your API key
GOOGLE_API = os.getenv("GOOGLE_API")
HF_API = os.getenv('HF_TOKEN')
PINECONE_API_KEY = os.getenv('PINECONE_API')


def load_data():
    """
    Load data from a file using the provided file path and JSON schema.

    Args:
        file_path (str): The path to the file containing the data.
        jq_schema (str): The JSON schema to be used for loading the data.

    Returns:
        list: A list of Document objects created from the loaded data.
    """
    loader = JSONLoader(
        file_path="data/FAQ.json",
        jq_schema='.[]',
        text_content=False)

    data = loader.load()
    data = [Document(page_content=doc.page_content) for doc in data]
    return data


@st.cache_resource
def create_embeddings(model_name):
    """
    Creates an instance of `HuggingFaceEmbeddings` with the specified `model_name` and caches it for future use.

    Args:
        model_name (str): The name of the Hugging Face model to use for embedding creation.

    Returns:
        HuggingFaceEmbeddings: The cached instance of `HuggingFaceEmbeddings` with the specified `model_name`.
    """
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
def create_chroma_db(persist_directory, _embeddings):
    """
    Creates a Chroma database using the given file path, JSON schema, persist directory, and embedding function.

    Args:
        file_path (str): The path to the file containing the data.
        jq_schema (str): The JSON schema to be used for loading the data.
        persist_directory (str): The directory where the database will be persisted.
        _embeddings (HuggingFaceEmbeddings): The embedding function to use.

    Returns:
        Chroma: The created Chroma database.
    """
    if os.path.isdir(os.getcwd() + '/vectorstore'):
        db = Chroma(persist_directory=persist_directory,
                    embedding_function=_embeddings)
    else:
        data = load_data()
        db = Chroma.from_documents(
            data, _embeddings, persist_directory=persist_directory)
    return db


# @st.cache_resource
def create_pinecone_db(data, _embeddings):
    """
    Creates a Pinecone index from the given data using the provided embeddings.

    Args:
        data (List[str]): A list of documents to be indexed.
        _embeddings (PineconeEmbeddings): The embeddings to be used for indexing.

    Returns:
        PineconeIndex: The created Pinecone index.

    Raises:
        PineconeError: If there is an error creating the index.

    Note:
        The dimension of the index is assumed to be 1536, which is the dimension of the embeddings.
        The repository used is DENSE.
    """
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
    """
    Creates a ChatGoogleGenerativeAI object with the given parameters.

    Args:
        model (str): The name of the model to use for the ChatGoogleGenerativeAI.
        temperature (float): The temperature parameter for the ChatGoogleGenerativeAI.
        max_tokens (int): The maximum number of tokens to generate for the ChatGoogleGenerativeAI.
        top_p (float): The top-p parameter for the ChatGoogleGenerativeAI.
        GOOGLE_API (str): The API key for the ChatGoogleGenerativeAI.

    Returns:
        ChatGoogleGenerativeAI: The created ChatGoogleGenerativeAI object.
    """
    llm = ChatGoogleGenerativeAI(model=model,
                                 temperature=temperature,
                                 max_tokens=max_tokens,
                                 top_p=top_p,
                                 google_api_key=GOOGLE_API)
    return llm


def create_prompt(system_prompt):
    """
    Creates a chat prompt template using the given system prompt.

    Args:
        system_prompt (str): The system prompt to be used in the chat prompt template.

    Returns:
        ChatPromptTemplate: The created chat prompt template.
    """
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
    """
    Creates a RAG chain using the specified retriever, prompt, and llm objects.
    Args:
        retriever: The retriever object used in the RAG chain.
        prompt: The prompt object used in the RAG chain.
        llm: The llm object used in the RAG chain.
    Returns:
        The created RAG chain.
    """
    rag_chain = (
        {"context": retriever | format_docs, "Question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def create_correction_prompt_template(query):
    """
    Creates a correction prompt template for a given user query.

    Args:
        query (str): The user query to be corrected.

    Returns:
        PromptTemplate: The created correction prompt template.

    The correction prompt template is a PromptTemplate object that prompts the user to correct the given query as a standalone question for grammar and spelling errors. If there are no errors, the same query is returned.

    Example:
        >>> create_correction_prompt_template("What is the weather today?")
        <PromptTemplate: Please correct the following user query as a standalone question for grammar and spelling if there are any errors. Otherwise, return the same query: {query}>

    """
    template = PromptTemplate(
        input_variables=["query"],
        template="Please correct the following user query as a standalone question for grammar and spelling if there are any errors. Otherwise, return the same query: `{query}`",
    )
    return template.format(query=query)


def process_query(rag_chain, query, correction_prompt_template, llm):
    """
    Process a user query by invoking a correction prompt template to check for grammar and spelling errors. If there are no errors, the same query is returned. 

    Args:
        rag_chain (RAGChain): The RAG chain used to generate a response to the query.
        query (str): The user query to be processed.
        correction_prompt_template (PromptTemplate): The correction prompt template used to check for grammar and spelling errors.
        llm (ChatGoogleGenerativeAI): The ChatGoogleGenerativeAI object used to generate a response to the query.

    Returns:
        str: The processed query with any grammar and spelling errors corrected.
    """
    standalone_question = llm.invoke(correction_prompt_template).content
    result = rag_chain.invoke(standalone_question)
    return result


# Efficiently call the functions for RAG
def rag_pipeline(retriever, model, temperature, max_tokens, top_p, GOOGLE_API, system_prompt, query):
    """
    Runs a pipeline to process a user query using a Retrieval-Augmented Generation (RAG) approach.

    Args:
        file_path (str): The path to the file containing the data to be ingested.
        jq_schema (str): The JSON query schema used to extract data from the file.
        embeddings (Embeddings): The embeddings model used to encode the data.
        persist_directory (str): The directory where the database will be persisted.
        model (str): The name of the language model used for generation.
        temperature (float): The temperature parameter for the language model.
        max_tokens (int): The maximum number of tokens to generate for the language model.
        top_p (float): The top-p parameter for the language model.
        GOOGLE_API (str): The API key for the Google API.
        system_prompt (str): The system prompt used for the RAG chain.
        query (str): The user query to be processed.

    Returns:
        str: The processed query with any grammar and spelling errors corrected.
    """

    # # Create the database
    # db = create_pinecone_db(data, embeddings)

    # Create the LLM
    llm = create_llm(model, temperature, max_tokens, top_p, GOOGLE_API)

    # Create the prompt
    prompt = create_prompt(system_prompt)

    # Create the RAG chain
    rag_chain = create_rag_chain(retriever, prompt, llm)

    # Create correction prompt template
    correction_prompt_template = create_correction_prompt_template(query)

    # Process the query
    result = process_query(
        rag_chain, query, correction_prompt_template, llm)

    return result


# # Example usage for Testing
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

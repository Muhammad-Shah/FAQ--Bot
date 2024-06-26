import json
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from pathlib import Path
from pprint import pprint


from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document

loader = JSONLoader(
    file_path='FAQ.json',
    jq_schema='.[]',
    text_content=False)

data = loader.load()

# Process the data
data = [Document(page_content=doc.page_content) for doc in data]

# Print the processed data
pprint(data[:10])
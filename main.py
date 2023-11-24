import getpass
import os

import weaviate

client = weaviate.Client(
    url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(ADMIN_WEAVIATE_API_KEY)
)

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

os.environ["WEAVIATE_API_KEY"] = ADMIN_WEAVIATE_API_KEY
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]


from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate

from langchain.document_loaders import TextLoader

loader = TextLoader("tsne.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

db = Weaviate.from_documents(docs, embeddings, weaviate_url=WEAVIATE_URL, by_text=False)

query = "What sne stands for?"
docs = db.similarity_search(query)

print(docs[0].page_content)
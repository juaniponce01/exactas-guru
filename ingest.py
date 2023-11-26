import os
import click
from typing import List
from utils import xlxs_to_csv
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from constants import SOURCE_DIRECTORY
from langchain.vectorstores.weaviate import Weaviate
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import weaviate

# auth_config = weaviate.AuthApiKey(api_key=os.environ['WEAVIATE_API_KEY'])

# client = weaviate.Client(
#   url="https://rag-agent-cluster-wfzijl47.weaviate.network",
#   auth_client_secret=auth_config
# )


def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf8")
    elif file_path.endswith(".pdf"):
        loader = PDFMinerLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    return loader.load()[0]


def load_documents(source_dir: str) -> List[Document]:
    # Loads all documents from source documents directory
    all_files = os.listdir(source_dir)
    docs = []
    for file_path in all_files:
        if file_path[-4:] == 'xlsx':
            for doc in xlxs_to_csv(f"{source_dir}/{file_path}"):
                docs.append(load_single_document(doc))
        elif file_path[-4:] in ['.txt', '.pdf', '.csv']:
            docs.append(load_single_document(f"{source_dir}/{file_path}"))
    return docs
    # return [load_single_document(f"{source_dir}/{file_path}") for file_path in all_files if
    #         file_path[-4:] in ['.txt', '.pdf', '.csv']]
    

@click.command()
@click.option('--device_type', default='cuda', help='device to run on, select gpu, cpu or mps')
def main(device_type, ):
    # load the instructorEmbeddings
    if device_type in ['cpu', 'CPU']:
        device='cpu'
    elif device_type in ['mps', 'MPS']:
        device='mps'
    else:
        device='cuda'

    #  Load documents and split in chunks
    print(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    print(f"Split into {len(texts)} chunks of text")

    print(texts[0].page_content)
    
    # Create embeddings
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                                            #    , model_kwargs={"device": device})
    # embeddings = OpenAIEmbeddings()
    
    db = Weaviate.from_documents(texts, embeddings, weaviate_url=os.environ['WEAVIATE_URL'], by_text=False)
    
    # llm = LlamaCpp(
    #     streaming = True,
    #     modelpath = os.environ["MODEL_PATH"],
    #     tempertaure = 0.75,
    #     top_p = 1,
    #     verbose = True,
    #     n_ctx = 4096
    # )
    
    llm = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(search_kwargs={'k': 2}))
    
    query = "why does t-sne use t-distribution?"
    
    print(qa.run(query))


if __name__ == "__main__":
    main()
    
    
    # docs = db.similarity_search(query)
    # print(docs[0].page_content)



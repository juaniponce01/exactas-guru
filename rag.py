import os
import torch

from textwrap import fill
from IPython.display import Markdown, display

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    )

from langchain import PromptTemplate
from langchain import HuggingFacePipeline

from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.chains import ConversationalRetrievalChain

from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

import warnings
warnings.filterwarnings('ignore')

class RAG():
    def __init__(self, model_name, source_directory):
        self.model_name = model_name
        self.source_directory = source_directory
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quantization_config
        )

        generation_config = GenerationConfig.from_pretrained(model_name)
        generation_config.max_new_tokens = 1024
        generation_config.temperature = 0.0001
        generation_config.top_p = 0.95
        generation_config.do_sample = True
        generation_config.repetition_penalty = 1.15
        
        pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=True,
            generation_config=generation_config,
        )

        self.llm = HuggingFacePipeline(
            pipeline=pipeline,
            )
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="thenlper/gte-large",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        loader = PyPDFDirectoryLoader(self.source_directory)
        self.documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
        self.texts = text_splitter.split_documents(self.documents)
        
        self.vector_store = Chroma.from_documents(self.texts, self.embeddings, persist_directory="db")
        
        custom_template = """You are a computer science AI Assistant that helps computer science students. You must provide readable answers, well structured and with much information as possible. Given the
        following conversation and a follow up question, rephrase the follow up question
        to be a standalone question. At the end of standalone question add this
        'Answer the question in the language the input is provided'. If you do not know the answer reply with 'I am sorry, I dont have enough information'.
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:
        """

        self.custom_question_prompt = PromptTemplate.from_template(custom_template)
        
    def __load_pdfs(self):
        loader = PyPDFDirectoryLoader(self.source_directory)
        self.documents = loader.load()
    
    def __split_documents(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
        self.texts = text_splitter.split_documents(self.documents)
        
    def refresh_vector_store(self):
        self.__load_pdfs()
        self.__split_documents()
        self.vector_store = Chroma.from_documents(self.texts, self.embeddings, persist_directory="db")
        
    def querying(self, query, history):
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 2}),
            memory=memory,
            condense_question_prompt=self.custom_question_prompt,
        )

        result = qa_chain({"question": query})
        return result["answer"].strip()
    
    






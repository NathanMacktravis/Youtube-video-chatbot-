import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain.chains import LLMChain
import whisper
import chainlit as cl
from langchain.schema.runnable.config import RunnableConfig


# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = "ytberag"
YOUTUBE_AUDIO = "/Users/nathanwandji/Downloads/youtube_audio.mp3"
TRANSCRIPTION_FILE = "transcription.txt"

## Create the model 
def create_model():
    return ChatOpenAI(
        openai_api_key=openai_api_key,
        model="gpt-3.5-turbo", 
        streaming=True
    )

## The prompt 
def create_prompt():
    template = (
            [
                (
                    "system",
                    "You are a RAG specialized in transcript analysis and you answer the user's questions based on what is in it. The original transcript has been divided into several subtexts and the context provided to you is the closest to the user's question. If you can't answer the question user based on this context, answ
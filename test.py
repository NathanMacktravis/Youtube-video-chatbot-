import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
import whisper
import chainlit as cl

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
        model="gpt-3.5-turbo"
    )

## The prompt 
def create_prompt():
    template = """
        Answer the question based on the context below. If you can't 
        answer the question, reply "I don't know".

        Context: {context}

        Question: {question}
    """
    return ChatPromptTemplate.from_template(template)

## The parser 
def create_parser():
    return StrOutputParser()

## The transcriptor 
def transcriptor():
    # Check if the transcription file already exists
    if not os.path.exists(TRANSCRIPTION_FILE):
        whisper_model = whisper.load_model("base")
        transcription = whisper_model.transcribe(YOUTUBE_AUDIO, 
                                                 fp16=False)["text"].strip()
        with open(TRANSCRIPTION_FILE, "w") as file:
            file.write(transcription)

## Splitting the transcription
def splitter():
    text_documents = TextLoader(TRANSCRIPTION_FILE).load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=2
    )
    documents = text_splitter.split_documents(text_documents)
    return documents

## The vector store
def vector_store():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    documents = splitter()

    return PineconeVectorStore.from_documents(
        embedding=embeddings, 
        documents=documents, 
        index_name=index_name
    )

# The main function 
def modes_response (): 
    model = create_model()
    prompt = create_prompt()
    pinecone = vector_store()
    parser = create_parser()

    # Create the chain 
    chain = (
        {
            "context": pinecone.as_retriever(),
            "question": RunnablePassthrough()
        }
        | prompt
        | model
        | parser
    )

    # Using the chain
    response = chain.invoke("what are the exploits in physic ?")
    print(response)

    return response 

## Chainlit  code 
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)


if __name__ == "__main__":
    main()

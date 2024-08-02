# Youtube-video-chatbot
# YouTube Video Chatbot

This repository contains an application that allows you to interact with a chatbot that answers questions based on the content of a YouTube video. The application utilizes multiple tools and libraries to transcribe audio, generate embeddings, and query a language model to provide accurate responses.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Tools Used](#tools-used)
  - [Model](#model)
  - [Parser](#parser)
  - [Vector Store](#vector-store)
  - [Text Splitter](#text-splitter)
  - [Chainlit Interface](#chainlit-interface)
  - [Notebook](#notebook)
- [Running the Application](#running-the-application)
  - [Running with Chainlit](#running-with-chainlit)
  - [Running with Jupyter Notebook](#running-with-jupyter-notebook)


## Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/NathanMacktravis/Youtube-video-chatbot-
   cd /Youtube-video-chatbot-
   ```

2. **Create and activate a Python virtual environment:**

   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   Create a `.env` file in the root directory and add your API keys:

   ```
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

## Usage

### Tools Used

#### Model

- **OpenAI's GPT-3.5-Turbo Model**: The core language model used for generating responses. The model is instantiated using `ChatOpenAI` from the `langchain_openai` library.

#### Parser

- **StrOutputParser**: This parser is used to convert the model's output into a simple string format. It ensures that the model's response is correctly formatted before being sent back to the user.

#### Vector Store

- **Pinecone Vector Store**: Pinecone is used to store and retrieve text embeddings. It is particularly useful for retrieving context relevant to the user's query from the transcribed YouTube video.

#### Text Splitter

- **RecursiveCharacterTextSplitter**: This tool splits the transcribed text into smaller chunks that can be easily processed by the model. It ensures that the context provided to the model does not exceed its input limitations.

#### Chainlit Interface

- **Chainlit**: This is the interface used to interact with the chatbot. It allows for a smooth and interactive chat experience.

#### Notebook

- **RAG Notebook**: The notebook `rag_app.ipynb` contains the implementation for running the application in a more interactive environment. It is useful for testing, understanding what i did and developp purposes.

## Running the Application

### Running with Chainlit

To run the application using the Chainlit interface:

1. **Ensure your virtual environment is activated.**
2. **Run the application:**

   ```sh
   chainlit run app.py
   ```

   This command will start the Chainlit server, and you can interact with the chatbot through the web interface that Chainlit provides.

### Running with Jupyter Notebook

To run the application in a Jupyter Notebook:

1. **Ensure your virtual environment is activated.**
2. **Start Jupyter Notebook:**

   ```sh
   jupyter notebook
   ```

3. **Open the `rag_app.ipynb` notebook** and follow the instructions within to execute the cells. This notebook provides a step-by-step guide to transcribing the video, generating embeddings, and querying the language model.


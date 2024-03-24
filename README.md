# Therapy Chatbot Assistant

The Therapy Chatbot Assistant is a project aimed at providing personalized therapy support through a conversational interface. Leveraging OpenAI's API and Mistral(local implementation) for natural language processing (NLP), this chatbot utilizes a RAG (Retrieval-Augmented Generation) implementation of Large Language Models (LLMs) for accurate responses.
I implemted the assistant using Langchain which is a framework that simplifies the creation of applications using large language models.
I also used Chainlit to build a production ready Conversational AI interface.

## Features

- **Personalized Therapy Support**: Users can interact with the chatbot to receive personalized therapy guidance and support.
  
- **RAG Implementation**: The chatbot utilizes a RAG implementation of LLMs to retrieve and generate accurate responses from a PDF of therapy-related information.

- **Conversational History**: Integration with Langchain ensures retaining and building upon past interactions, resulting in coherent and engaging dialogue across conversations.
.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Stephen-Echessa/therapy-chatbot.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Obtain OpenAI API keys:
    - Visit [OpenAI API](https://openai.com/api/) to obtain API keys.
    - Add the keys to your environment variables or directly to the code.

## Usage

1. Run the chatbot:
    ```bash
    chainlit run therapy_gpt_final_model.py -w
    ```

2. Interact with the chatbot by typing your queries and receiving responses.

## Testing

The project includes additional folders for testing the OpenAI API and also Mistral as NLP chatbots. Refer to the respective folders and files for testing scripts and results.

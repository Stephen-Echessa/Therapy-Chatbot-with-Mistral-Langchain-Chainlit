from langchain_community.llms import OpenAI
from langchain.vectorstores import Weaviate
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
import chainlit as cl
import os

prompt_template = """
   ### [INST]
   Instruction: You are an expert at answering NLP questions.
   When a user thanks you, please make sure to thank them back.
   When a query by the user is out of context, please make sure to state that you have not been trained to answer that question.
   Here is context to help: {context}
   Follow-up Instructions:
   {chat_history}
   ##QUESTION:
   {question}
   [/INST]
"""

def set_prompt():
    prompt = PromptTemplate(
        input_variables=['context','question', 'chat_history'],
        template=prompt_template
    )
    return prompt

def conversation_chain(llm, prompt, retriever):
    memory = ConversationBufferMemory(llm=llm, memory_key='chat_history', output_key='answer', return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, 
                                         memory=memory,
                                         retriever=retriever,
                                         combine_docs_chain_kwargs={'prompt': prompt}
    )
    return chain

def load_llm():
    llm = OpenAI(api_key=os.environ['OPENAI_API_KEY'],temperature=0.7)
    return llm

def preprocess_documents():
    loader = PyPDFLoader('../data/Reading 2 Text Analytics for Beginners using NLTK_240116_161801.pdf')
    data = loader.load()
    text_gen = ''
    for page in data:
        text_gen += page.page_content
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1024,
        chunk_overlap = 5
    )
    chunks_gen = text_splitter.split_text(text_gen)
    docs = [Document(page_content=t) for t in chunks_gen]
    return docs
    
def chatbot():
    embedding_model = HuggingFaceBgeEmbeddings(model_name='BAAI/bge-small-en-v1.5')
    docs = preprocess_documents()
    vector_store = Weaviate.from_documents(docs, 
                                embedding_model, 
                                weaviate_url = 'http://localhost:8080'
    )
    retriever = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={'k': 2}
    )
    llm = load_llm()
    prompt = set_prompt()
    return conversation_chain(llm, prompt, retriever)

def final_result(query):
    bot = chatbot()
    response = bot.invoke(query)
    
@cl.on_chat_start
async def start():
    chain = chatbot()
    msg = cl.Message(content='Bot is starting...')
    await msg.send()
    msg.content = 'Welcome... I am a Test NLP Question Answering Chatbot. How may I help you?'
    await msg.update()
    
    cl.user_session.set('chain', chain)
    
@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get('chain')
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True
    )
    cb.answer_reached = True
    response = await chain.acall(message.content, callbacks=[cb])
    await cl.Message(response['answer']).send()
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory.buffer import ConversationBufferMemory
import chainlit as cl
import os

@cl.on_chat_start
def setup_multiple_chains():
    llm = OpenAI(api_key=os.environ['OPENAI_API_KEY'], temperature=0.7)
    conversation_memory = ConversationBufferMemory(
        memory_key='chat_history',
        max_len=50,
        return_messages=True
    )
    prompt_template = """
    ### [INST]
        Instruction: You are an expert at answering NLP-related questions.
        ##QUESTION:
        {question}
        Chat History: {chat_history}
        Answer:
        [/INST]
    """
    prompt = PromptTemplate(
        input_variables=['chat_history', 'question'],
        template=prompt_template
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt, memory=conversation_memory)
    cl.user_session.set('llm_chain', llm_chain)
    
@cl.on_message
async def query_llm(message: cl.Message):
    llm_chain = cl.user_session.get('llm_chain')
    response = await llm_chain.acall(message.content,
                                     callbacks=[
                                         cl.AsyncLangchainCallbackHandler()
                                     ])
    await cl.Message(response['text']).send()

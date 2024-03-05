from fastapi import FastAPI
import asyncio
from fastapi.responses import StreamingResponse

from handler import MyCustomHandler
from threading import Thread

from queue import Queue

from langchain.schema.messages import HumanMessage
from langchain_community.llms import Ollama
from langchain.chains import LLMChain

app = FastAPI()
streamer_queue = Queue()
my_handler = MyCustomHandler(streamer_queue)

llm = Ollama(model='mistral:7b-instruct-q4_K_M', temperature=0.7, callbacks=[my_handler])

from typing import List, Tuple

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.
tavily_api_key = os.getenv("TAVILY_API_KEY")

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Create the tool
search = TavilySearchAPIWrapper(tavily_api_key="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")
description = """You are a highly skilled and empathetic doctor.
                                    Your primary role is to diagnose ailments based on the symptoms described by the patient.
                                    You should ask relevant questions to gather enough information about the patient's condition.
                                    Once you have enough information, use the tool to provide a possible diagnosis and suggest appropriate treatments or medications.
                                    However, you should always remind the patient that while you can provide advice based on their symptoms, they should seek professional medical help for a definitive diagnosis and treatment.
                                    Remember to maintain a professional and caring tone throughout the conversation.
                                """
tavily_tool = TavilySearchResults(api_wrapper=search, description=description)

tools = [tavily_tool]

llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-pro", google_api_key=gemini_api_key)

prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm_with_tools = llm.bind(functions=tools)


def _format_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


agent = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        #"agent_scratchpad": lambda x: format_to_openai_function_messages(
         #   x["intermediate_steps"]
        #),
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)


class AgentInput(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]] = Field(
        ..., extra={"widget": {"type": "chat", "input": "input", "output": "output"}}
    )


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True).with_types(
    input_type=AgentInput
)

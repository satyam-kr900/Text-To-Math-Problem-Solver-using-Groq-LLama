from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
load_dotenv()
model=ChatOpenAI(model='gpt-4o')
from langchain.agents import create_react_agent,AgentExecutor

from langchain import hub
prompt = hub.pull(
    "hwchase17/react",
)

search_tool=DuckDuckGoSearchRun()

agent=create_react_agent(llm=model,tools=[search_tool],prompt=prompt)

agent_executer=AgentExecutor(agent=agent,tools=[search_tool],verbose=True,handle_parsing_errors=True)

response=agent_executer.invoke({'input':'3 ways to visit goa to pune'})

print(response['output'])
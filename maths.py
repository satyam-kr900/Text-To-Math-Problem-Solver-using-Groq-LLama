import streamlit as st

from langchain_core.prompts import PromptTemplate

from langchain_groq import ChatGroq

from dotenv import load_dotenv

from langchain.chains import LLMChain, LLMMathChain

from langchain_community.utilities import WikipediaAPIWrapper

from langchain.agents.agent_types import AgentType

from langchain.agents import Tool, initialize_agent

from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

import re

load_dotenv()

st.title("Text To Math Problem Solver using Groq LLama")

groq_api = st.sidebar.text_input(
    'Please Enter Your Groq API key',
    type='password'
)

if not groq_api:

    st.info("Please Enter Your Groq API key:")

    st.stop()

model = ChatGroq(
    model="qwen/qwen3.8-27b",
    groq_api_key=groq_api
)


# initialize Agents

wikipedia_wrapper = WikipediaAPIWrapper()

wikipedia_tool = Tool(
    name='wikipedia',
    func=wikipedia_wrapper.run,
    description='Agent used for searching over the internet to find various information'
)

math_chain = LLMMathChain.from_llm(llm=model)


def math_tool_func(question):

    math_expr = ''.join(
        re.findall(r'[\d\.\+\-\*\/\^\%\(\)]+', question)
    )

    return math_chain.run(math_expr)


calculator = Tool(
    name='calculator',
    func=math_tool_func,
    description='Tool used for answering math related questions. Only input mathematical expression needed'
)

prompt = '''you are an agent Tasked with solving user mathematics problems.
Logically arrive at the solution and display it point wise for the question below.
Question: {question}

Answer:'''


prompt_template = PromptTemplate(
    template=prompt,
    input_variables=['question']
)

chain = LLMChain(
    prompt=prompt_template,
    llm=model
)

Reasoning = Tool(
    name='Reasoning Tool',
    func=chain.run,
    description='A Tool used for answering logic based and reasoning questions'
)


# Build the Agent

assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, Reasoning],
    llm=model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)


if "messages" not in st.session_state:

    st.session_state['messages'] = [
        {
            'role': 'assistant',
            'content': 'Hi! I am Math chatbot who can answer all your maths questions'
        }
    ]


for msg in st.session_state.messages:

    st.chat_message(msg['role']).write(msg['content'])


question = st.text_area("Please Ask Your question:")

if st.button("Find My Answer"):

    if question:

        with st.spinner("Generating Response..."):

            st.session_state.messages.append({
                'role': 'user',
                'content': question
            })

            st.chat_message('user').write(question)

        if re.search(r'^[\d\.\+\-\*\/\^\%\(\)]+$', question):

            response = calculator.invoke(question)

        else:

            st_cb = StreamlitCallbackHandler(
                st.container(),
                expand_new_thoughts=False
            )

            response = assistant_agent.invoke(
                {"input": question},
                callbacks=[st_cb]
            )

            response = response["output"]

        st.session_state.messages.append({
            'role': 'assistant',
            'content': response
        })

        st.chat_message('assistant').write(response)

    else:

        st.warning("Please Enter The Questions")







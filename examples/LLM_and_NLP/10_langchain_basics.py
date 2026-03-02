"""
10. LangChain Basics Example

LLM applications using LangChain
"""

print("=" * 60)
print("LangChain Basics")
print("=" * 60)

# ============================================
# 1. LangChain Structure (Code Example)
# ============================================
print("\n[1] LangChain Basic Structure")
print("-" * 40)

langchain_basic = '''
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Prompt template
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")

# Output parser
parser = StrOutputParser()

# Chain composition (LCEL)
chain = prompt | llm | parser

# Execute
result = chain.invoke({"topic": "programming"})
print(result)
'''
print(langchain_basic)


# ============================================
# 2. Prompt Templates
# ============================================
print("\n[2] Prompt Template Examples")
print("-" * 40)

try:
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

    # Basic template
    template = PromptTemplate(
        input_variables=["product"],
        template="Write a marketing slogan for {product}."
    )
    print(f"Basic template: {template.format(product='smartphone')}")

    # Chat template
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{question}")
    ])
    messages = chat_template.format_messages(question="What is Python?")
    print(f"\nChat template: {messages}")

except ImportError:
    print("langchain not installed (pip install langchain langchain-core)")


# ============================================
# 3. Few-shot Prompt
# ============================================
print("\n[3] Few-shot Prompt")
print("-" * 40)

fewshot_code = '''
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
]

example_template = PromptTemplate(
    input_variables=["word", "antonym"],
    template="Word: {word}\\nAntonym: {antonym}"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="Give the antonym of each word:",
    suffix="Word: {input}\\nAntonym:",
    input_variables=["input"]
)

prompt = few_shot_prompt.format(input="big")
'''
print(fewshot_code)


# ============================================
# 4. Output Parser
# ============================================
print("\n[4] Output Parser")
print("-" * 40)

parser_code = '''
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="Name")
    age: int = Field(description="Age")

parser = JsonOutputParser(pydantic_object=Person)

# Add format instructions to prompt
format_instructions = parser.get_format_instructions()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract person info. {format_instructions}"),
    ("human", "{text}")
]).partial(format_instructions=format_instructions)

chain = prompt | llm | parser
result = chain.invoke({"text": "John is 25 years old"})
# {'name': 'John', 'age': 25}
'''
print(parser_code)


# ============================================
# 5. Chain (LCEL)
# ============================================
print("\n[5] LCEL Chain")
print("-" * 40)

lcel_code = '''
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Sequential chain
chain = prompt | llm | parser

# Parallel chain
parallel = RunnableParallel(
    summary=summary_chain,
    keywords=keyword_chain
)

# Branching chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

# Execute
result = chain.invoke({"question": "What is AI?"})
'''
print(lcel_code)


# ============================================
# 6. RAG Chain
# ============================================
print("\n[6] RAG Chain")
print("-" * 40)

rag_chain_code = '''
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(texts, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# RAG prompt
template = """Answer based on context:
Context: {context}
Question: {question}
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# Document formatting
def format_docs(docs):
    return "\\n\\n".join(doc.page_content for doc in docs)

# RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI()
    | StrOutputParser()
)

# Execute
answer = rag_chain.invoke("What is machine learning?")
'''
print(rag_chain_code)


# ============================================
# 7. Agent
# ============================================
print("\n[7] Agent")
print("-" * 40)

agent_code = '''
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.tools import tool

# Custom tools
@tool
def calculate(expression: str) -> str:
    """Calculate a math expression."""
    return str(eval(expression))

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Search results for: {query}"

tools = [calculate, search]

# ReAct agent
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Execute
result = executor.invoke({"input": "What is 2 + 2?"})
'''
print(agent_code)


# ============================================
# 8. Conversation Memory
# ============================================
print("\n[8] Conversation Memory")
print("-" * 40)

memory_code = '''
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Memory
memory = ConversationBufferMemory()

# Conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Conversation
response1 = conversation.predict(input="Hi, I'm John")
response2 = conversation.predict(input="What's my name?")
# "Your name is John"

# LCEL Memory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)
'''
print(memory_code)


# ============================================
# 9. Runnable Example
# ============================================
print("\n[9] Runnable Example")
print("-" * 40)

try:
    from langchain_core.prompts import PromptTemplate

    # Test prompt template only
    template = PromptTemplate.from_template(
        "Translate '{text}' to {language}."
    )

    # Formatting
    prompt = template.format(text="Hello", language="Korean")
    print(f"Generated prompt: {prompt}")

    # Input variables
    print(f"Input variables: {template.input_variables}")

except ImportError:
    print("langchain-core not installed")


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("LangChain Summary")
print("=" * 60)

summary = """
Key Patterns:
    # Basic chain
    chain = prompt | llm | output_parser

    # RAG chain
    rag = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

    # Agent
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)

Main Components:
    - PromptTemplate: Prompt composition
    - ChatOpenAI: LLM wrapper
    - OutputParser: Output parsing
    - Retriever: Document retrieval
    - Memory: Conversation history
    - Agent: Tool usage
"""
print(summary)

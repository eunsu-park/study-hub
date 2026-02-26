# 12. 실전 챗봇 프로젝트

## 학습 목표

- 대화형 AI 시스템 설계
- RAG 기반 챗봇 구현
- 대화 관리와 메모리
- 프로덕션 배포 고려사항

---

## 1. 챗봇 아키텍처

### 기본 구조

```
┌─────────────────────────────────────────────────────────────┐
│                       Chatbot System                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  사용자 입력                                                  │
│      │                                                       │
│      ▼                                                       │
│  [의도 분류] ──▶ FAQ / RAG / 일반 대화 분기                   │
│      │                                                       │
│      ▼                                                       │
│  [컨텍스트 검색] ◀── 벡터 DB                                 │
│      │                                                       │
│      ▼                                                       │
│  [프롬프트 구성] ◀── 대화 히스토리                           │
│      │                                                       │
│      ▼                                                       │
│  [LLM 생성]                                                  │
│      │                                                       │
│      ▼                                                       │
│  응답 출력                                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 기본 챗봇 구현

### 간단한 대화 챗봇

```python
from openai import OpenAI

class SimpleChatbot:
    def __init__(self, system_prompt=None):
        self.client = OpenAI()
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.history = []

    def chat(self, user_message):
        # 메시지 구성
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message})

        # API 호출
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7
        )

        assistant_message = response.choices[0].message.content

        # 히스토리 업데이트
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def clear_history(self):
        self.history = []

# 사용
bot = SimpleChatbot("You are a friendly customer support agent.")
print(bot.chat("Hi, I need help with my order."))
print(bot.chat("My order number is 12345."))
```

---

## 3. RAG 챗봇

### 문서 기반 Q&A

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class RAGChatbot:
    def __init__(self, documents, persist_dir="./rag_db"):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.embeddings = OpenAIEmbeddings()

        # 문서 분할
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        # 벡터 스토어
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_dir
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # 대화 히스토리
        self.history = []

        # RAG 체인 구성
        self._setup_chain()

    def _setup_chain(self):
        template = """You are a helpful assistant. Answer based on the context.
If you don't know the answer, say so.

Context:
{context}

Conversation History:
{history}

Question: {question}

Answer:"""

        self.prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        def format_history(history):
            if not history:
                return "No previous conversation."
            return "\n".join([f"{m['role']}: {m['content']}" for m in history[-6:]])

        self.chain = (
            {
                "context": self.retriever | format_docs,
                "history": lambda x: format_history(self.history),
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def chat(self, question):
        response = self.chain.invoke(question)

        # 히스토리 업데이트
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": response})

        return response

    def get_sources(self, question):
        """검색된 소스 문서 반환"""
        docs = self.retriever.invoke(question)
        return [{"content": d.page_content, "metadata": d.metadata} for d in docs]
```

---

## 4. 고급 대화 관리

### 의도 분류

```python
class IntentClassifier:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def classify(self, message, intents):
        prompt = f"""Classify the user message into one of these intents: {intents}

Message: {message}

Intent (only output the intent name):"""

        response = self.llm.invoke(prompt)
        return response.content.strip()

# 사용
classifier = IntentClassifier()
intent = classifier.classify(
    "I want to return my purchase",
    ["order_status", "return_request", "product_inquiry", "general"]
)
# "return_request"
```

### 슬롯 추출

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

class OrderSlots(BaseModel):
    order_id: str = Field(default=None, description="Order ID")
    product_name: str = Field(default=None, description="Product name")
    issue: str = Field(default=None, description="Customer's issue")

class SlotExtractor:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.parser = JsonOutputParser(pydantic_object=OrderSlots)

    def extract(self, message, context=""):
        prompt = f"""Extract information from the message.
{self.parser.get_format_instructions()}

Context: {context}
Message: {message}

JSON:"""

        response = self.llm.invoke(prompt)
        return self.parser.parse(response.content)

# 사용
extractor = SlotExtractor()
slots = extractor.extract("I want to return order #12345, the shirt is too small")
# {'order_id': '12345', 'product_name': 'shirt', 'issue': 'too small'}
```

### 대화 상태 관리

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any

class ConversationState(Enum):
    GREETING = "greeting"
    COLLECTING_INFO = "collecting_info"
    PROCESSING = "processing"
    CONFIRMING = "confirming"
    COMPLETED = "completed"

@dataclass
class ConversationContext:
    state: ConversationState = ConversationState.GREETING
    slots: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)
    missing_slots: List[str] = field(default_factory=list)

class StatefulChatbot:
    def __init__(self):
        self.context = ConversationContext()
        self.required_slots = ["order_id", "issue"]

    def process(self, message):
        # 슬롯 추출
        new_slots = self.extract_slots(message)
        self.context.slots.update({k: v for k, v in new_slots.items() if v})

        # 누락된 슬롯 확인
        self.context.missing_slots = [
            s for s in self.required_slots
            if s not in self.context.slots or not self.context.slots[s]
        ]

        # 상태 전이
        if self.context.missing_slots:
            self.context.state = ConversationState.COLLECTING_INFO
            return self.ask_for_slot(self.context.missing_slots[0])
        else:
            self.context.state = ConversationState.CONFIRMING
            return self.confirm_action()

    def ask_for_slot(self, slot_name):
        prompts = {
            "order_id": "Could you please provide your order number?",
            "issue": "What issue are you experiencing with your order?"
        }
        return prompts.get(slot_name, f"Please provide {slot_name}.")

    def confirm_action(self):
        return f"Let me confirm: Order {self.context.slots['order_id']}, Issue: {self.context.slots['issue']}. Is this correct?"
```

---

## 5. 스트리밍 응답

```python
from openai import OpenAI

class StreamingChatbot:
    def __init__(self):
        self.client = OpenAI()
        self.history = []

    def chat_stream(self, message):
        messages = [{"role": "system", "content": "You are helpful."}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": message})

        stream = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True
        )

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content

        # 히스토리 업데이트
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": full_response})

# 사용
bot = StreamingChatbot()
for chunk in bot.chat_stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

---

## 6. FastAPI 웹 서버

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# 세션 저장소
sessions = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: list = []

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # 세션 가져오기/생성
    if request.session_id not in sessions:
        sessions[request.session_id] = RAGChatbot(documents)

    bot = sessions[request.session_id]

    # 응답 생성
    response = bot.chat(request.message)
    sources = bot.get_sources(request.message)

    return ChatResponse(response=response, sources=sources)

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    if request.session_id not in sessions:
        sessions[request.session_id] = StreamingChatbot()

    bot = sessions[request.session_id]

    def generate():
        for chunk in bot.chat_stream(request.message):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "cleared"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 7. Gradio UI

```python
import gradio as gr

class ChatbotUI:
    def __init__(self):
        self.bot = RAGChatbot(documents)

    def respond(self, message, history):
        response = self.bot.chat(message)
        return response

    def launch(self):
        demo = gr.ChatInterface(
            fn=self.respond,
            title="Document Q&A Chatbot",
            description="Ask questions about your documents",
            examples=["What is this document about?", "Summarize the main points"],
            theme="soft"
        )
        demo.launch()

# 사용
ui = ChatbotUI()
ui.launch()
```

---

## 8. 프로덕션 고려사항

### 에러 처리

```python
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionChatbot:
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def chat(self, message):
        try:
            response = self._generate_response(message)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble processing your request. Please try again."
```

### 토큰/비용 관리

```python
import tiktoken

class TokenManager:
    def __init__(self, model="gpt-3.5-turbo", max_tokens=4000):
        self.encoding = tiktoken.encoding_for_model(model)
        self.max_tokens = max_tokens

    def count_tokens(self, text):
        return len(self.encoding.encode(text))

    def truncate_history(self, history, max_history_tokens=2000):
        """오래된 메시지부터 제거"""
        total_tokens = 0
        truncated = []

        for msg in reversed(history):
            msg_tokens = self.count_tokens(msg['content'])
            if total_tokens + msg_tokens > max_history_tokens:
                break
            truncated.insert(0, msg)
            total_tokens += msg_tokens

        return truncated
```

### 모니터링

```python
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class ChatMetrics:
    session_id: str
    message: str
    response: str
    latency_ms: float
    token_count: int
    timestamp: float

class MonitoredChatbot:
    def __init__(self):
        self.metrics = []

    def chat(self, session_id, message):
        start = time.time()

        response = self._generate(message)

        latency = (time.time() - start) * 1000

        # 메트릭 기록
        metric = ChatMetrics(
            session_id=session_id,
            message=message,
            response=response,
            latency_ms=latency,
            token_count=self.token_manager.count_tokens(response),
            timestamp=time.time()
        )
        self.metrics.append(metric)

        return response

    def get_avg_latency(self):
        if not self.metrics:
            return 0
        return sum(m.latency_ms for m in self.metrics) / len(self.metrics)
```

---

## 9. 전체 시스템 예제

```python
"""
완전한 RAG 챗봇 시스템
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionRAGChatbot:
    def __init__(self, docs_dir, persist_dir="./prod_db"):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        self.embeddings = OpenAIEmbeddings()

        # 문서 로드
        logger.info(f"Loading documents from {docs_dir}")
        loader = DirectoryLoader(docs_dir, glob="**/*.txt")
        documents = loader.load()

        # 청킹
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")

        # 벡터 스토어
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_dir
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

        # 세션 관리
        self.sessions = {}

    def _get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {"history": [], "context": {}}
        return self.sessions[session_id]

    def chat(self, session_id, message):
        session = self._get_session(session_id)

        # 관련 문서 검색
        docs = self.retriever.invoke(message)
        context = "\n\n".join([d.page_content for d in docs])

        # 히스토리 포맷
        history_text = "\n".join([
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
            for m in session["history"][-6:]
        ])

        # 프롬프트
        prompt = f"""You are a helpful assistant. Answer based on the context provided.
If you cannot find the answer in the context, say so honestly.

Context:
{context}

Conversation History:
{history_text}

User: {message}
Assistant:"""

        # LLM 호출
        response = self.llm.invoke(prompt)
        answer = response.content

        # 히스토리 업데이트
        session["history"].append({"role": "user", "content": message})
        session["history"].append({"role": "assistant", "content": answer})

        return answer
```

---

## 정리

### 챗봇 설계 체크리스트

```
□ 용도 정의 (일반 대화 / FAQ / 문서 기반)
□ RAG 필요 여부 결정
□ 대화 히스토리 관리 방식
□ 의도 분류 필요 여부
□ 에러 처리 및 폴백
□ 비용 관리 (토큰 제한)
□ 모니터링 및 로깅
```

### 핵심 패턴

```python
# 기본 챗봇
messages = [system_prompt] + history + [user_message]
response = llm.invoke(messages)

# RAG 챗봇
docs = retriever.invoke(query)
context = format_docs(docs)
response = llm.invoke(prompt.format(context=context, question=query))

# 스트리밍
for chunk in llm.stream(messages):
    yield chunk
```

### 다음 단계

- 실제 서비스 배포 (AWS, GCP)
- A/B 테스트 설정
- 사용자 피드백 수집
- 지속적인 모델 개선

## 연습 문제

### 연습 문제 1: 토큰 제한이 있는 히스토리 잘라내기

`TokenManager.truncate_history` 메서드는 히스토리가 `max_history_tokens`를 초과하면 오래된 메시지를 제거합니다. 그런데 미묘한 버그가 있습니다: 대화 쌍을 깨뜨릴 수 있습니다(예: 어시스턴트 메시지는 유지하면서 해당 사용자 메시지를 삭제). 이 문제를 수정하고 단위 테스트를 추가하세요.

<details>
<summary>정답 보기</summary>

```python
import tiktoken

class TokenManager:
    def __init__(self, model="gpt-3.5-turbo", max_tokens=4000):
        self.encoding = tiktoken.encoding_for_model(model)
        self.max_tokens = max_tokens

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def truncate_history(self, history: list[dict], max_history_tokens: int = 2000) -> list[dict]:
        """
        대화 일관성 유지를 위해 가장 오래된 메시지 쌍(pair)부터 제거합니다.
        고아(orphan) 메시지 방지를 위해 항상 user+assistant 쌍을 함께 제거합니다.
        """
        if not history:
            return history

        # 총 토큰 수 계산
        total_tokens = sum(self.count_tokens(msg['content']) for msg in history)

        if total_tokens <= max_history_tokens:
            return history  # 잘라내기 불필요

        # 가장 오래된 쌍부터 제거 (한 번에 2개 메시지)
        truncated = list(history)
        while truncated and total_tokens > max_history_tokens:
            # 가장 오래된 쌍 제거 (user + assistant)
            if len(truncated) >= 2:
                removed_user = truncated.pop(0)
                removed_assistant = truncated.pop(0)
                total_tokens -= (
                    self.count_tokens(removed_user['content']) +
                    self.count_tokens(removed_assistant['content'])
                )
            else:
                # 메시지 하나만 남으면 제거
                removed = truncated.pop(0)
                total_tokens -= self.count_tokens(removed['content'])

        return truncated


# 단위 테스트
def test_truncate_history():
    tm = TokenManager()

    # 알려진 토큰 수로 대화 구성
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm great!"},
        {"role": "user", "content": "Tell me about Python"},
        {"role": "assistant", "content": "Python is a high-level programming language"},
    ]

    # 테스트 1: 잘라내기 불필요
    result = tm.truncate_history(history, max_history_tokens=1000)
    assert len(result) == 6, "제한 내에서는 모든 메시지 유지"

    # 테스트 2: 가장 오래된 쌍 제거
    result = tm.truncate_history(history, max_history_tokens=25)
    assert len(result) % 2 == 0, "결과는 짝수 개의 메시지(완전한 쌍)여야 함"
    assert result[0]["role"] == "user", "첫 메시지는 user여야 함"

    # 테스트 3: 쌍(pair) 무결성 검증 (고아 어시스턴트 메시지 없음)
    for i in range(0, len(result), 2):
        assert result[i]["role"] == "user", f"메시지 {i}는 user여야 함"
        assert result[i+1]["role"] == "assistant", f"메시지 {i+1}는 assistant여야 함"

    print("모든 테스트 통과!")

test_truncate_history()
```

**원본의 버그:** 원본 구현은 `reversed(history)`로 최신부터 토큰을 세지만, 제한에 도달해도 메시지 쌍 무결성을 보장하지 않습니다. 대응하는 사용자 메시지가 없는 어시스턴트 메시지는 LLM을 혼란스럽게 만듭니다.
</details>

---

### 연습 문제 2: 의도(Intent) 기반 챗봇 라우터

사용자 의도를 `rag_query`(문서로 답변 가능한 질문), `chitchat`(일반 대화), `action_request`(주문과 같은 실제 행동 필요)의 세 가지로 분류하는 챗봇 라우터(router)를 구축하세요. 각 의도를 다른 핸들러(handler)로 라우팅하세요.

<details>
<summary>정답 보기</summary>

```python
from enum import Enum
from dataclasses import dataclass

class Intent(Enum):
    RAG_QUERY = "rag_query"
    CHITCHAT = "chitchat"
    ACTION_REQUEST = "action_request"

@dataclass
class ChatbotResponse:
    intent: Intent
    response: str
    sources: list = None

class RouterChatbot:
    def __init__(self, rag_chatbot=None):
        from openai import OpenAI
        self.client = OpenAI()
        self.rag_chatbot = rag_chatbot  # 선택적 RAG 컴포넌트
        self.history = []

    def classify_intent(self, message: str) -> Intent:
        """Zero-shot LLM 분류로 사용자 의도 파악."""
        prompt = f"""다음 메시지를 정확히 하나의 카테고리로 분류하세요:
- rag_query: 문서/지식 베이스에서 답변이 필요한 질문
- chitchat: 일상 대화, 인사, 일반적인 질문
- action_request: 무언가를 해달라는 요청 (주문, 예약, 취소 등)

메시지: "{message}"

카테고리 (카테고리 이름만 답하세요):"""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20
        )
        result = response.choices[0].message.content.strip().lower()

        mapping = {
            "rag_query": Intent.RAG_QUERY,
            "chitchat": Intent.CHITCHAT,
            "action_request": Intent.ACTION_REQUEST,
        }
        return mapping.get(result, Intent.CHITCHAT)

    def handle_rag(self, message: str) -> ChatbotResponse:
        """문서 기반 질문 처리."""
        if self.rag_chatbot:
            response = self.rag_chatbot.chat(message)
            sources = self.rag_chatbot.get_sources(message)
        else:
            response = "해당 질문에 답변하기 위한 문서에 접근할 수 없습니다."
            sources = []
        return ChatbotResponse(Intent.RAG_QUERY, response, sources)

    def handle_chitchat(self, message: str) -> ChatbotResponse:
        """일상 대화 처리."""
        messages = [
            {"role": "system", "content": "당신은 친근한 대화 어시스턴트입니다."},
            {"role": "user", "content": message}
        ]
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.8
        )
        return ChatbotResponse(Intent.CHITCHAT, response.choices[0].message.content)

    def handle_action(self, message: str) -> ChatbotResponse:
        """행동 요청 처리 (실제 행동 로직의 플레이스홀더)."""
        response = (
            f"'{message}'를 처리하고 싶으신 거군요. "
            "요청을 처리하기 위해 더 자세한 정보가 필요합니다. "
            "추가 정보를 제공해 주시겠어요?"
        )
        return ChatbotResponse(Intent.ACTION_REQUEST, response)

    def chat(self, message: str) -> ChatbotResponse:
        """메인 진입점: 의도 분류 후 라우팅."""
        intent = self.classify_intent(message)

        if intent == Intent.RAG_QUERY:
            result = self.handle_rag(message)
        elif intent == Intent.CHITCHAT:
            result = self.handle_chitchat(message)
        else:
            result = self.handle_action(message)

        # 공유 히스토리 업데이트
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": result.response, "intent": intent.value})

        return result

# 라우팅 테스트
bot = RouterChatbot()

test_messages = [
    "반품 정책이 어떻게 되나요?",   # → rag_query
    "안녕하세요! 잘 지내시나요?",    # → chitchat
    "주문을 취소하고 싶어요",        # → action_request
]

for msg in test_messages:
    result = bot.chat(msg)
    print(f"[{result.intent.value}] {msg}")
    print(f"  → {result.response[:80]}...\n")
```
</details>

---

### 연습 문제 3: 슬롯 정정(Slot Correction) 처리

`StatefulChatbot`을 확장하여 대화 중간에 사용자가 모순된 정보를 제공하는 경우(예: 처음에 주문 #12345를 말했다가 "아, #67890이었어요"라고 수정)를 처리하세요. 봇이 슬롯을 업데이트하고 수정을 확인해야 합니다.

<details>
<summary>정답 보기</summary>

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from openai import OpenAI

class ConversationState(Enum):
    GREETING = "greeting"
    COLLECTING_INFO = "collecting_info"
    CONFIRMING = "confirming"
    COMPLETED = "completed"

@dataclass
class ConversationContext:
    state: ConversationState = ConversationState.GREETING
    slots: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)
    corrections: List[Dict] = field(default_factory=list)  # 정정 이력 추적

class SmartStatefulChatbot:
    """대화 중 슬롯 정정을 감지하는 챗봇."""

    REQUIRED_SLOTS = ["order_id", "issue"]
    SLOT_QUESTIONS = {
        "order_id": "주문 번호를 알려주시겠어요?",
        "issue": "주문에 어떤 문제가 있으신가요?"
    }

    def __init__(self):
        self.client = OpenAI()
        self.context = ConversationContext()

    def extract_slots(self, message: str) -> dict:
        """메시지에서 슬롯 값을 추출합니다."""
        import json
        prompt = f"""메시지에서 주문 정보를 추출하세요. JSON만 반환하세요.
필드: order_id (문자열, 예: "12345"), issue (문자열 설명)
누락된 필드는 null 사용.

메시지: "{message}"
JSON:"""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {"order_id": None, "issue": None}

    def detect_correction(self, new_slots: dict) -> Optional[str]:
        """사용자가 이전에 제공한 슬롯을 정정하는지 확인합니다."""
        for slot, new_value in new_slots.items():
            if new_value and slot in self.context.slots:
                old_value = self.context.slots[slot]
                if old_value and old_value != new_value:
                    return slot  # 이 슬롯이 정정됨
        return None

    def process(self, message: str) -> str:
        self.context.history.append({"role": "user", "content": message})
        new_slots = self.extract_slots(message)

        # 정정 확인
        corrected_slot = self.detect_correction(new_slots)
        if corrected_slot:
            old_value = self.context.slots[corrected_slot]
            new_value = new_slots[corrected_slot]
            self.context.corrections.append({
                "slot": corrected_slot,
                "old": old_value,
                "new": new_value
            })
            self.context.slots[corrected_slot] = new_value
            response = (
                f"알겠습니다! {corrected_slot}을 "
                f"'{old_value}'에서 '{new_value}'로 수정했습니다. "
            )
        else:
            # 일반 슬롯 업데이트
            for slot, value in new_slots.items():
                if value:
                    self.context.slots[slot] = value
            response = ""

        # 아직 누락된 슬롯 확인
        missing = [s for s in self.REQUIRED_SLOTS
                  if not self.context.slots.get(s)]

        if missing:
            self.context.state = ConversationState.COLLECTING_INFO
            response += self.SLOT_QUESTIONS[missing[0]]
        else:
            self.context.state = ConversationState.CONFIRMING
            response += self.confirm_action()

        self.context.history.append({"role": "assistant", "content": response})
        return response

    def confirm_action(self) -> str:
        return (
            f"확인해 드리겠습니다: 주문 #{self.context.slots['order_id']}, "
            f"문제: {self.context.slots['issue']}. 맞으신가요? (네/아니요)"
        )

# 정정 처리 테스트
bot = SmartStatefulChatbot()

print(bot.process("주문 12345 반품하고 싶어요"))   # order_id 설정, issue 질문
print(bot.process("상품이 파손됐어요"))            # issue 설정, 확인 요청
print(bot.process("아, 주문은 67890이었어요"))     # order_id 정정

print(f"\n정정 이력: {bot.context.corrections}")
# [{'slot': 'order_id', 'old': '12345', 'new': '67890'}]
```
</details>

---

## 학습 완료

이것으로 LLM & NLP 학습 과정을 완료했습니다!

### 학습 요약

1. **NLP 기초 (01-03)**: 토큰화, 임베딩, Transformer
2. **사전학습 모델 (04-07)**: BERT, GPT, HuggingFace, 파인튜닝
3. **LLM 활용 (08-12)**: 프롬프트, RAG, LangChain, 벡터 DB, 챗봇

### 다음 단계 추천

- 실제 프로젝트에 적용
- Kaggle NLP 대회 참가
- 최신 LLM 논문 읽기 (Claude, Gemini, Llama)


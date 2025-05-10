import os
import logging
import requests
import json
import traceback
import uuid
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory 

from dotenv import load_dotenv
load_dotenv()

# настраиваем логирование
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("agent_logger") 
logger.setLevel(logging.INFO)

info_handler = logging.FileHandler("logs/agent.log", encoding="utf-8") # общий лог-файл
info_handler.setLevel(logging.INFO)

error_handler = logging.FileHandler("logs/errors.log", encoding="utf-8") # лог-файл для ошибок
error_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

logger.addHandler(info_handler)
logger.addHandler(error_handler)

class State(TypedDict):
    text: str
    classification: str
    sentiment: str
    entities: dict
    summary: str
    knowledge: str
    response: str
    memory: str

llm = ChatOpenAI(
    model="gpt-4.1-nano",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    temperature=0
)

# инициализируем модель и векторное хранилище
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

chat_memory = ConversationBufferMemory(
    llm=llm,
    memory_key="history",
    return_messages=True,
    max_token_limit=1000
)

def run_agent(user_input: str, verbose: bool = False) -> str:
    """
    Запускает агента на основе входного текста пользователя и возвращает ответ.
    """
    state = {"text": user_input}
    try:
        for step in agent.stream(state):
            if verbose:
                print(f"Step: {step}")
                print(f"Data: {agent.get_state(step)}\n")
        final = agent.invoke(state)
        if verbose:
            print(f"Final state: {final}\n")
        return final["response"]
    except Exception as e:
        error_id = str(uuid.uuid4())[:8]  # короткий уникальный ID ошибки
        error_type = type(e).__name__
        trace = traceback.format_exc()
        logger.error(f"[{error_type}] Error ID: {error_id} | Input: {user_input}\n{str(e)}\nTraceback:\n{trace}")
        return f"Извините, произошла внутренняя ошибка (ID: {error_id}). Мы уже работаем над её устранением."

def pre_filter_node(state: State):
    """
    Предварительная фильтрация сообщений.
    """
    text = state["text"].lower()
    if len(text) < 10 or any(p in text for p in ["спасибо", "ок", "ага"]):
        return {"response": "Пожалуйста, уточните ваш запрос — сейчас он слишком короткий или непонятный."}
    if not any(x in text for x in ["заказ", "товар", "доставка", "оплата", "промокод", "возврат", "жалоба", 
                                   "отзыв", "гарантия", "адрес", "оформить", "сертификат", "статус"]):
        return {"response": "Похоже, ваш запрос не относится к поддерживаемым темам. Пожалуйста, уточните."}
    return {}

def classify_node(state: State):
    """
    Классификация сообщения пользователя по категории: Order, Return, Complaint, Feedback, Other.
    """
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Classify the following customer message into a category (Order, Return, Complaint, Feedback, Other):\n\n{text}\n\nCategory:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    classification = llm.invoke([message]).content.strip()

    valid_categories = {"Order", "Return", "Complaint", "Feedback", "Other"}
    if classification not in valid_categories:
        classification = "Other"

    return {"classification": classification}

def sentiment_node(state: State):
    """
    Анализ тональности сообщения: Positive, Neutral, Negative.
    """
    prompt = PromptTemplate(
        input_variables=["text"],
        template="What is the sentiment of the following message? (Positive, Neutral, Negative)\n\n{text}\n\nSentiment:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    sentiment = llm.invoke([message]).content.strip()
    return {"sentiment": sentiment}

def entity_node(state: State):
    """
    Извлечение сущностей из сообщения пользователя (например, ID заказа, адрес, email, etc.).
    """
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Extract all named entities (User ID, User Name, Order ID, Product Category, " \
        "Order Date, Address, Email, Phone Number, Delivery Method, Payment Method, Promocode) " \
        "from the text:\n\n{text}\n\nEntities (comma-separated):"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    raw_output = llm.invoke([message]).content.strip()
    entity_dict = {}
    for line in raw_output.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            entity_dict[key.strip()] = value.strip()
    
    return {"entities": entity_dict}

def summary_node(state: State):
    """
    Краткое резюме сообщения пользователя на русском языке.
    """
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the customer's message in one sentence in Russian:\n\n{text}\n\nSummary:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    summary = llm.invoke([message]).content.strip()
    return {"summary": summary}

def retrieve_knowledge_node(state: State):
    """
    Поиск релевантной информации в базе знаний.
    """
    query = state["text"]
    docs = vectorstore.similarity_search_with_score(query, k=3)
    for doc, score in docs:
        logger.info(f"Query: {query} | Score: {score:.4f} | Content: {doc.page_content[:200]}...")
    relevant_docs = [doc for doc, score in docs if score < 0.7]  # подобрать порог
    if not relevant_docs:
        return {"knowledge": ""}
    context = "\n".join([doc.page_content for doc in relevant_docs])
    return {"knowledge": context}

def response_node(state: State):
    """
    Генерация финального ответа пользователю с учетом категории, тональности, памяти и знаний.
    """
    memory_summary = chat_memory.load_memory_variables({}).get("history", "")
    
    prompt = PromptTemplate(
        input_variables=["text", "classification", "sentiment", "knowledge", "history"],
        template=(
            "You are a helpful support agent. Use the following context to respond:\n\n"
            "History:\n{history}\n\n"
            "Context:\n{knowledge}\n\n"
            "Message: {text}\nCategory: {classification}\nSentiment: {sentiment}\n\nResponse:"
        )
    )
    message = HumanMessage(content=prompt.format(
        text=state["text"],
        classification=state.get("classification", "Other"),
        sentiment=state.get("sentiment", "Neutral"),
        knowledge=state.get("knowledge", ""),
        history=memory_summary
    ))
    
    try:
        response = llm.invoke([message]).content.strip()
        chat_memory.save_context({"input": state["text"]}, {"output": response})
    except Exception as e:
        error_type = type(e).__name__
        trace = traceback.format_exc()
        logger.error(f"[{error_type}] LLM error in response_node: {str(e)} | Input: {state}\nTraceback:\n{trace}")
        response = "Извините, произошла ошибка. Пожалуйста, повторите запрос позже."

    logger.info(json.dumps({
        "user_message": state["text"],
        "category": state.get("classification", "Unknown"),
        "sentiment": state.get("sentiment", "Unknown"),
        "entities": state.get("entities", {}),
        "summary": state.get("summary", ""),
        "knowledge": state.get("knowledge", ""),
        "response": response
    }, ensure_ascii=False, indent=2))


    return {"response": response}

def send_telegram_alert(text: str):
    """
    Перенаправление саммари пользовательского запроса оператору в Telegram.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.warning("Telegram token or chat_id not set.")
        return

    message = f"🚨 Перенаправление жалобы клиента:\n\n{text}"

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Ошибка отправки в Telegram: {str(e)}")

def escalate_node(state: State):
    """
    Уведомление пользователя и отправка жалобы оператору в Telegram.
    """
    message = "Ваш запрос передан оператору для дальнейшего рассмотрения."
    logger.info(f"Input: {state['text']} | Escalation triggered")
    send_telegram_alert(state["summary"])
    return {"response": message}

def review_request_node(state: State):
    """
    Просьба оставить отзыв, если отзыв пользователя положительный.
    """
    message = "Спасибо за положительный отзыв! Хотите оставить отзыв на сайте о товаре?"
    logger.info(f"Input: {state['text']} | Review requested")
    return {"response": message}

def route_request(state: State) -> str:
    """
    Логика маршрутизации в графе в зависимости от категории, тональности и наличия знаний.
    """
    if state["classification"] == "Feedback" and state["sentiment"] == "Positive":
        return "positive_feedback"
    if not state["knowledge"]:
        return "no_knowledge_or_negative"
    if state["classification"] == "Complaint" and state["sentiment"] == "Negative":
        return "no_knowledge_or_negative"
    return "respond"

# Граф обработки запроса
workflow = StateGraph(State)
workflow.add_node("pre_filter", pre_filter_node)
workflow.add_node("classify", classify_node)
workflow.add_node("analyze_sentiment", sentiment_node)
workflow.add_node("entity", entity_node)
workflow.add_node("summary_message", summary_node)
workflow.add_node("retrieve_knowledge", retrieve_knowledge_node)
workflow.add_node("get_response", response_node)
workflow.add_node("escalate_to_operator", escalate_node)
workflow.add_node("ask_for_review", review_request_node)

workflow.set_entry_point("pre_filter")
workflow.add_conditional_edges(
    "pre_filter",
    lambda state: "get_response" if "response" in state else "classify",
    {
        "get_response": "get_response",
        "classify": "classify"
    }
)
workflow.add_edge("classify", "analyze_sentiment")
workflow.add_edge("analyze_sentiment", "entity")
workflow.add_edge("entity", "summary_message")
workflow.add_edge("summary_message", "retrieve_knowledge")
workflow.add_conditional_edges(
    "retrieve_knowledge",
    route_request,
    {
        "respond": "get_response",
        "no_knowledge_or_negative": "escalate_to_operator",
        "positive_feedback": "ask_for_review"
    }
)
workflow.add_edge("get_response", END)
workflow.add_edge("escalate_to_operator", END)
workflow.add_edge("ask_for_review", END)

agent = workflow.compile()

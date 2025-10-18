import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import List, Optional, TypedDict, Union

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

load_dotenv()


class ConversationState(TypedDict, total=False):
    user_question: str
    faq_answer: str
    messages: List[str]
    route: str
    is_greeting: bool


DEFAULT_GREETING_MESSAGE = "Olá! Que bom ter você por aqui. Como posso ajudar hoje?"


_openai_api_key = os.getenv("OPENAI_API_KEY")

llm = None
_llm_executor: Optional[ThreadPoolExecutor] = None
if _openai_api_key:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.4,
        max_tokens=512,
        timeout=10,
    )
    _llm_executor = ThreadPoolExecutor(max_workers=1)


@tool("test_greeting")
def test_greeting(user_question: str) -> str:
    """
    Avalia se a mensagem do usuário é apenas uma saudação usando modelo da OpenAI.

    Returns:
        "greeting" para mensagens de saudação.
        "answer" caso contrário.
    """
    text = (user_question or "").strip()
    if not text:
        return "greeting"

    if llm is None:
        # Fallback simples quando o modelo não está disponível.
        if len(text.split()) <= 2:
            return "greeting"
        return "answer"

    prompt = [
        (
            "system",
            "Classifique a mensagem do usuário como 'greeting' quando for apenas uma saudação ou uma abertura cordial, "
            "caso contrário responda apenas 'answer'. Responda exatamente com uma dessas duas palavras.",
        ),
        ("user", f"Mensagem: {text}"),
    ]

    def _invoke_llm():
        return llm.invoke(prompt)

    try:
        if _llm_executor:
            future = _llm_executor.submit(_invoke_llm)
            resposta = future.result(timeout=30)
        else:
            resposta = llm.invoke(prompt)
        rotulo = _message_to_text(resposta).strip().lower()
        if rotulo not in {"greeting", "answer"}:
            return "answer"
        return rotulo
    except TimeoutError:
        print("[warn] Timeout ao classificar mensagem na OpenAI.")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Falha ao classificar mensagem na OpenAI: {exc}")

    return "answer"


@tool("generate_rag_answer")
def generate_rag_answer(user_question: str, faq_answer: str) -> str:
    """
    Gera uma resposta RAG, usando o LLM quando disponível.
    """
    fallback_answer = faq_answer or "Ainda não tenho detalhes sobre esse tópico."

    prompt = [
        (
            "system",
            "Você é um assistente da livraria Let's Read atendendo participantes do AKCIT Camp. "
            "Responda sempre em português do Brasil, com tom acolhedor e profissional. "
            "Use a resposta base do FAQ como referência e adapte para um formato natural. "
            "Se não tiver informação suficiente, seja transparente e convide a pessoa a reformular ou perguntar algo diferente.",
        ),
        (
            "user",
            f"Pergunta da pessoa: {user_question}\n\nResposta base do FAQ: {fallback_answer}\n\nMonte uma resposta completa e acolhedora.",
        ),
    ]

    if llm is None:
        return (
            f"{fallback_answer}\n"
            "Posso te ajudar em algo mais? (Não consegui acessar o modelo da OpenAI agora.)"
        )

    def _invoke_llm():
        return llm.invoke(prompt)

    try:
        if _llm_executor:
            future = _llm_executor.submit(_invoke_llm)
            resposta = future.result(timeout=90)
        else:
            resposta = llm.invoke(prompt)
        answer_text = _message_to_text(resposta).strip()
        return answer_text or fallback_answer
    except TimeoutError:
        print("[warn] Timeout ao consultar OpenAI.")
    except Exception as exc:  # noqa: BLE001 - mantemos log detalhado
        print(f"[warn] Falha ao consultar OpenAI: {exc}")

    return (
        f"{fallback_answer}\n"
        "Obs.: Não consegui reformular a resposta agora, mas posso ajudar com outra dúvida."
    )


def _greeting_node(state: ConversationState) -> ConversationState:
    print(f"[langgraph] Entrando no nó greet com estado: {state}")
    messages = list(state.get("messages", []))
    messages.append(DEFAULT_GREETING_MESSAGE)
    novo_estado = {**state, "messages": messages}
    print(f"[langgraph] Saindo do nó greet com estado: {novo_estado}")
    return novo_estado


def _message_to_text(message: Union[AIMessage, str]) -> str:
    """Extrai texto de respostas do LangChain, independentemente do formato."""
    if isinstance(message, AIMessage):
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Concatenar partes textuais quando o provedor retorna chunks
            parts = []
            for chunk in content:
                chunk_text = getattr(chunk, "text", None) or getattr(chunk, "content", None)
                if chunk_text:
                    parts.append(chunk_text)
            if parts:
                return "\n".join(parts)
        return message.pretty_repr()
    return str(message)


def _test_greeting_node(state: ConversationState) -> ConversationState:
    user_question = state.get("user_question", "")
    route = test_greeting.invoke({"user_question": user_question})
    novo_estado = {**state, "route": route, "is_greeting": route == "greeting"}
    print(f"[langgraph] Resultado do test_greeting: {route}")
    return novo_estado


def _rag_answer_node(state: ConversationState) -> ConversationState:
    print(f"[langgraph] Entrando no nó generate_rag_answer com estado: {state}")
    messages = list(state.get("messages", []))
    faq_answer = state.get("faq_answer", "")
    user_question = state.get("user_question", "")

    answer_text = generate_rag_answer.invoke(
        {"user_question": user_question, "faq_answer": faq_answer}
    )
    messages.append(answer_text)
    novo_estado = {**state, "messages": messages}
    print(f"[langgraph] Saindo do nó generate_rag_answer com estado: {novo_estado}")
    return novo_estado


_graph_builder = StateGraph(ConversationState)
_graph_builder.add_node("test_greeting", _test_greeting_node)
_graph_builder.add_node("greet", _greeting_node)
_graph_builder.add_node("generate_rag_answer", _rag_answer_node)
_graph_builder.set_entry_point("test_greeting")
_graph_builder.add_conditional_edges(
    "test_greeting",
    lambda state: state.get("route", "answer"),
    {
        "greeting": "greet",
        "answer": "generate_rag_answer",
    },
)
_graph_builder.add_edge("greet", END)
_graph_builder.add_edge("generate_rag_answer", END)

conversation_graph = _graph_builder.compile()


def generate_flow_response(user_question: str, faq_answer: str) -> List[str]:
    """Executa um fluxo simples no LangGraph para cumprimentar e responder ao usuário."""
    try:
        final_state = conversation_graph.invoke(
            {
                "user_question": user_question,
                "faq_answer": faq_answer,
                "messages": [],
            }
        )
    except Exception as exc:
        print(f"[error] Falha ao executar grafo: {exc}")
        return [faq_answer]

    if not isinstance(final_state, dict):
        print(f"[error] Estado final inesperado: {final_state}")
        return [faq_answer]

    messages = final_state.get("messages") or [faq_answer]
    print(f"[langgraph] Mensagens geradas: {messages}")
    return messages

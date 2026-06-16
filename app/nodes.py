from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.database import search_similar_tickets, settings
from app.state import (AnalyzeNodeState, ClassifyNodeState, RecommendNodeState,
                       TicketState)

llm = ChatOllama(
    model=settings.llm_model,
    base_url=settings.llm_base_url,
    keep_alive="24h",
    temperature=0,
)

# --- Classify Node ---

_classify_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert software project manager.
Analyze the given Jira ticket and classify it.

Return:
- labels: list of relevant technical tags (e.g. RAG, OCR, search, embeddings, bug, performance)
- work_type: one of: bug, feature, task, improvement
"""),
    ("human", "Title: {title}\n\nDescription: {description}"),
])

_classify_chain = _classify_prompt | llm.with_structured_output(ClassifyNodeState)


async def classify_node(state: TicketState) -> dict:
    result = await _classify_chain.ainvoke({
        "title": state.input_state.title,
        "description": state.input_state.description,
    })
    return {"classify_state": result}


# --- Analyze Node ---

_analyze_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior software engineer estimating ticket complexity.
Given the ticket details and similar past tickets, assess the priority.

Return:
- priority: one of: low, medium, high, very high
- similar_ticket_ids: list of similar ticket IDs provided to you (can be empty)
"""),
    ("human", """Title: {title}
Description: {description}
Labels: {labels}
Work type: {work_type}
Similar past tickets: {similar_tickets}
"""),
])

_analyze_chain = _analyze_prompt | llm.with_structured_output(AnalyzeNodeState)


async def analyze_node(state: TicketState) -> dict:
    similar = await search_similar_tickets(
        f"{state.input_state.title} {state.input_state.description}"
    )
    result = await _analyze_chain.ainvoke({
        "title": state.input_state.title,
        "description": state.input_state.description,
        "labels": state.classify_state.labels if state.classify_state else [],
        "work_type": state.classify_state.work_type if state.classify_state else "",
        "similar_tickets": similar,
    })
    return {"analyze_state": result}


# --- Recommend Node ---

_recommend_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior software engineer providing actionable recommendations.
Based on the ticket analysis, provide a concrete recommended first action.
Be specific and practical — what should the developer do first?
"""),
    ("human", """Title: {title}
Description: {description}
Labels: {labels}
Work type: {work_type}
Priority: {priority}
Similar past tickets: {similar_tickets}
"""),
])

_recommend_chain = _recommend_prompt | llm.with_structured_output(RecommendNodeState)


async def recommend_node(state: TicketState) -> dict:
    result = await _recommend_chain.ainvoke({
        "title": state.input_state.title,
        "description": state.input_state.description,
        "labels": state.classify_state.labels if state.classify_state else [],
        "work_type": state.classify_state.work_type if state.classify_state else "",
        "priority": state.analyze_state.priority if state.analyze_state else "",
        "similar_tickets": state.analyze_state.similar_ticket_ids if state.analyze_state else [],
    })
    return {"recommend_state": result}

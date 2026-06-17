from langgraph.graph import END, START, StateGraph

import app.database as database
from app.nodes import analyze_node, classify_node, recommend_node
from app.state import TicketState

graph = StateGraph(TicketState)

graph.add_node("classify", classify_node)
graph.add_node("analyze", analyze_node)
graph.add_node("recommend", recommend_node)

graph.add_edge(START, "classify")
graph.add_edge("classify", "analyze")
graph.add_edge("analyze", "recommend")
graph.add_edge("recommend", END)


def create_graph():
    compiled_graph = graph.compile(checkpointer=database.checkpointer)
    return compiled_graph

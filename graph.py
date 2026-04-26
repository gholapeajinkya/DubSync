from langgraph.graph import StateGraph, START, END
from nodes.clean_audio import clean_audio_node
from nodes.extract_audio import extract_audio_node
from nodes.separate_layers import separate_audio_layers_node
from nodes.transcribe import transcribe_audio_node
from nodes.translate import translation_node
from state import AgentState


def error_node(state: AgentState) -> AgentState:
    """Handles errors in the pipeline."""
    print(f"Pipeline error: {state['error']}")
    return state


def route_on_error(success_route: str):
    """Returns a routing function that goes to error on failure, or success_route on success."""
    def router(state: AgentState) -> str:
        if state.get("error"):
            return "error"
        return success_route
    return router


graph = StateGraph(AgentState)

# Nodes
graph.add_node("extract_audio_node", extract_audio_node)
graph.add_node("separate_audio_layers_node", separate_audio_layers_node)
graph.add_node("clean_audio_node", clean_audio_node)
graph.add_node("transcribe_audio_node", transcribe_audio_node)
# Placeholder for translation node
graph.add_node("translation_node", translation_node)
graph.add_node("error_node", error_node)

# Edges
graph.add_edge(START, "extract_audio_node")
graph.add_conditional_edges(
    "extract_audio_node",
    route_on_error("separate_audio_layers"),
    {
        "separate_audio_layers": "separate_audio_layers_node",
        "error": "error_node"
    })
graph.add_conditional_edges(
    "separate_audio_layers_node",
    route_on_error("clean_audio_node"),
    {
        "clean_audio": "clean_audio_node",
        "error": "error_node"
    })

graph.add_conditional_edges(
    "clean_audio_node",
    route_on_error("transcribe_audio_node"),
    {
        "transcribe_audio": "transcribe_audio_node",
        "error": "error_node"
    })

graph.add_conditional_edges(
    "transcribe_audio_node",
    route_on_error("translation_node"),
    {
        "translation": "translation_node",
        "error": "error_node"
    })

graph.add_conditional_edges(
    "translation_node",
    route_on_error("end"),
    {
        "end": END,
        "error": "error_node"
    }
)

graph.add_edge("error_node", END)

app = graph.compile()


with open("assets/graph.png", "wb") as f:
    f.write(app.get_graph().draw_mermaid_png())

print("Graph created successfully!")

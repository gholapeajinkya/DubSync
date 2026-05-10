from langgraph.graph import StateGraph, START, END
from nodes.clean_audio import clean_audio_node
from nodes.extract_audio import extract_audio_node
from nodes.separate_layers import separate_audio_layers_node
from nodes.transcribe import transcribe_audio_node
from nodes.translate import translation_node
from nodes.voice_cloning import (
    prepare_voice_cloning_node,
    fanout_voice_cloning,
    voice_cloning_worker,
    combine_cloned_audio_and_generate_video_node
)
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
# Parallel voice cloning nodes
graph.add_node("prepare_voice_cloning_node", prepare_voice_cloning_node)
graph.add_node("voice_cloning_worker", voice_cloning_worker)
graph.add_node("combine_cloned_audio_node", combine_cloned_audio_and_generate_video_node)
graph.add_node("error_node", error_node)

# Edges
graph.add_edge(START, "extract_audio_node")
graph.add_conditional_edges(
    "extract_audio_node",
    route_on_error("separate_audio_layers_node"),
    {
        "separate_audio_layers_node": "separate_audio_layers_node",
        "error": "error_node"
    })
graph.add_conditional_edges(
    "separate_audio_layers_node",
    route_on_error("clean_audio_node"),
    {
        "clean_audio_node": "clean_audio_node",
        "error": "error_node"
    })

graph.add_conditional_edges(
    "clean_audio_node",
    route_on_error("transcribe_audio_node"),
    {
        "transcribe_audio_node": "transcribe_audio_node",
        "error": "error_node"
    })

graph.add_conditional_edges(
    "transcribe_audio_node",
    route_on_error("translation_node"),
    {
        "translation_node": "translation_node",
        "error": "error_node"
    })

graph.add_conditional_edges(
    "translation_node",
    route_on_error("prepare_voice_cloning_node"),
    {
        "prepare_voice_cloning_node": "prepare_voice_cloning_node",
        "error": "error_node"
    }
)

# Fan-out: prepare_voice_cloning_node sends to parallel voice_cloning_workers
graph.add_conditional_edges(
    "prepare_voice_cloning_node",
    fanout_voice_cloning,  # Returns list of Send commands for parallel execution
)

# Fan-in: all voice_cloning_workers converge to combine_cloned_audio_node
graph.add_edge("voice_cloning_worker", "combine_cloned_audio_node")

graph.add_conditional_edges(
    "combine_cloned_audio_node",
    route_on_error("end"),
    {
        "end": END,
        "error": "error_node"
    }
)

graph.add_edge("error_node", END)

app = graph.compile()


# with open("assets/graph.png", "wb") as f:
#     f.write(app.get_graph().draw_mermaid_png())

# print("Graph created successfully!")

app.invoke({
    "input_video_path": "sample_inputs/Frieren_ Beyond Journey's End Season 2 _ OFFICIAL TRAILER.mp4",
    "temp_folder": "resources",
})
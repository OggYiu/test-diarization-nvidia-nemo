from langchain.chat_models import init_chat_model
from tools.audio_chopper_tool import chop_audio_by_rttm
from tools.diarize_tool import diarize_audio


model = init_chat_model(
    "qwen3:14b",
    model_provider="ollama",
    temperature=0
)


# Augment the LLM with tools
tools = [diarize_audio, chop_audio_by_rttm]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

from langchain.messages import SystemMessage


def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant with access to audio processing tools. "
                        "For audio analysis workflows:\n"
                        "1. Use diarize_audio tool first to identify speakers and when they spoke\n"
                        "2. Use chop_audio_by_rttm tool to split the audio into speaker segments based on diarization results\n"
                        "These tools work together: diarization provides RTTM data, which is then used to chop the audio."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

from langchain.messages import ToolMessage


def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

from typing import Literal
from langgraph.graph import StateGraph, START, END


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# Save the agent graph visualization
png_data = agent.get_graph(xray=True).draw_mermaid_png()
with open("agent_graph.png", "wb") as f:
    f.write(png_data)
print("Agent graph saved to agent_graph.png")

# Invoke
from langchain.messages import HumanMessage

# Example 1: Audio diarization and chopping
# First, diarize the audio to identify speakers
# Then, chop the audio into segments based on the diarization results
messages = [HumanMessage(content="Diarize the audio file 'assets/test_audio_files/[Dickson Lau 0489]_8330-96674941_20251013012751(880).wav' with 2 speakers, then chop it into speaker segments with 50ms padding.")]
result = agent.invoke({"messages": messages})
for m in result["messages"]:
    m.pretty_print()

print("\n" + "="*80)
print("Workflow completed!")
print("="*80)
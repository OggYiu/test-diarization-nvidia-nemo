from langchain.chat_models import init_chat_model
from tools.audio_chopper_tool import chop_audio_by_rttm
from tools.diarize_tool import diarize_audio
from tools.stt_tool import transcribe_audio_segments


model = init_chat_model(
    "qwen3:14b",
    model_provider="ollama",
    temperature=0
)


# Augment the LLM with tools
tools = [diarize_audio, chop_audio_by_rttm, transcribe_audio_segments]
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
                        "3. Use transcribe_audio_segments tool to transcribe the chopped audio segments using SenseVoiceSmall\n"
                        "These tools work together: diarization provides RTTM data, which is used to chop the audio, "
                        "and then the segments are transcribed to text."
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
import os

# Example 1: Audio diarization and chopping
# First, diarize the audio to identify speakers
# Then, chop the audio into segments based on the diarization results

# Use absolute path to avoid path resolution issues
current_dir = os.path.dirname(os.path.abspath(__file__))
audio_file = os.path.join(current_dir, "assets", "test_audio_files", "[Dickson Lau]_8330-96674941_20251013035051(3360).wav")

# Verify the file exists
if not os.path.exists(audio_file):
    print(f"❌ Audio file not found: {audio_file}")
    print("\nAvailable files in test_audio_files:")
    test_audio_dir = os.path.join(current_dir, "assets", "test_audio_files")
    if os.path.exists(test_audio_dir):
        for file in os.listdir(test_audio_dir):
            if file.endswith(('.wav', '.mp3', '.flac')):
                print(f"  - {file}")
    exit(1)

print(f"📁 Using audio file: {audio_file}")
print(f"✅ File exists: {os.path.exists(audio_file)}")
print(f"📊 File size: {os.path.getsize(audio_file) / (1024*1024):.2f} MB\n")

# Convert Windows backslashes to forward slashes for LLM message
# (LLMs can misinterpret backslashes, but Windows accepts forward slashes)
audio_file_for_llm = audio_file.replace('\\', '/')

messages = [HumanMessage(content=f"Diarize the audio file '{audio_file_for_llm}' with 2 speakers, then chop it into speaker segments with 50ms padding, and finally transcribe all the segments to text using SenseVoiceSmall.")]
result = agent.invoke({"messages": messages})
for m in result["messages"]:
    m.pretty_print()

print("\n" + "="*80)
print("Workflow completed!")
print("="*80)
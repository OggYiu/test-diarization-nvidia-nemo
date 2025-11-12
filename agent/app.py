import os

from dotenv import load_dotenv

load_dotenv(os.path.join("..", ".env"), override=True)

import sys
import io
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
# from langchain.agents import create_react_agent

# Fix console encoding for Windows to support emoji characters
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from langchain.chat_models import init_chat_model
from tools.audio_chopper_tool import chop_audio_by_rttm
from tools.diarize_tool import diarize_audio
from tools.stt_tool import transcribe_audio_segments
from tools.metadata_tool import extract_metadata_from_filename
from tools.cantonese_corrector_tool import correct_transcriptions
from tools.stock_identifier_tool import identify_stocks_in_conversation


model = ChatOpenAI(
    api_key="ollama",  # Not used, but required by ChatOpenAI
    model="qwen3:8b",
    # base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    base_url="http://localhost:11434/v1",
    temperature=0.0
)

# Augment the LLM with tools
tools = [
    extract_metadata_from_filename,
    diarize_audio,
    chop_audio_by_rttm,
    transcribe_audio_segments,
    correct_transcriptions,
    identify_stocks_in_conversation
]
# tools_by_name = {tool.name: tool for tool in tools}
# model_with_tools = model.bind_tools(tools)

from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator


# class MessagesState(TypedDict):
#     messages: Annotated[list[AnyMessage], operator.add]
#     llm_calls: int

from langchain.messages import SystemMessage


# def llm_call(state: dict):
#     """LLM decides whether to call a tool or not"""

#     return {
#         "messages": [
#             model_with_tools.invoke(
#                 [
#                     SystemMessage(
#                         content="""You are a helpful assistant with access to audio processing tools. For audio analysis workflows:
# 1. Use extract_metadata_from_filename tool to extract metadata from audio filenames (broker name, client info, timestamps, etc.)
# 2. Use diarize_audio tool to identify speakers and when they spoke. It returns a JSON response with 'audio_filepath' and 'rttm_content' fields.
# 3. Use chop_audio_by_rttm tool with the 'audio_filepath' and 'rttm_content' values from the diarize_audio response to split the audio into speaker segments. The response will contain an 'output_dir' field.
# 4. Use transcribe_audio_segments tool with the 'output_dir' value from chop_audio_by_rttm to transcribe the chopped audio segments using SenseVoiceSmall.
# 5. Use correct_transcriptions tool to apply Cantonese text corrections to the transcriptions.

# IMPORTANT: Each tool returns a JSON response. You must extract the specific fields from each response and pass them as arguments to the next tool. For example:
# - After diarize_audio returns {"success": true, "audio_filepath": "...", "rttm_content": "...", ...}, extract the audio_filepath and rttm_content values and pass them to chop_audio_by_rttm.
# - After chop_audio_by_rttm returns {"success": true, "output_dir": "...", ...}, extract the output_dir value and pass it to transcribe_audio_segments."""
#                     )
#                 ]
#                 + state["messages"]
#             )
#         ],
#         "llm_calls": state.get('llm_calls', 0) + 1
#     }

from langchain.messages import ToolMessage
import json


# def tool_node(state: dict):
#     """Performs the tool call"""

#     # print("="*80)
#     # print(f"🐛 DEBUG: Tool node called with state: {state}")
#     # print("="*80)

#     result = []
#     for tool_call in state["messages"][-1].tool_calls:
#         tool = tools_by_name[tool_call["name"]]
#         observation = tool.invoke(tool_call["args"])
#         # Convert dict/object results to JSON string for LLM to parse
#         if isinstance(observation, dict):
#             observation = json.dumps(observation, indent=2)
#         result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
#     return {"messages": result}

from typing import Literal
from langgraph.graph import StateGraph, START, END


# def should_continue(state: MessagesState) -> Literal["tool_node", END]:
#     """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

#     messages = state["messages"]
#     last_message = messages[-1]

#     # If the LLM makes a tool call, then perform an action
#     if last_message.tool_calls:
#         return "tool_node"

#     # Otherwise, we stop (reply to the user)
#     return END

# Build workflow
# agent_builder = StateGraph(MessagesState)

# Add nodes
# agent_builder.add_node("llm_call", llm_call)
# agent_builder.add_node("tool_node", tool_node)

# # Add edges to connect nodes
# agent_builder.add_edge(START, "llm_call")
# agent_builder.add_conditional_edges(
#     "llm_call",
#     should_continue,
#     ["tool_node", END]
# )
# agent_builder.add_edge("tool_node", "llm_call")

# # Compile the agent
# agent = agent_builder.compile()

# Save the agent graph visualization
# png_data = agent.get_graph(xray=True).draw_mermaid_png()
# with open("agent_graph.png", "wb") as f:
#     f.write(png_data)
# print("Agent graph saved to agent_graph.png")

# Invoke
from langchain.messages import HumanMessage
import os

# Example 1: Audio diarization and chopping
# First, diarize the audio to identify speakers
# Then, chop the audio into segments based on the diarization results

# Use absolute path to avoid path resolution issues
current_dir = os.path.dirname(os.path.abspath(__file__))
# audio_file = os.path.join(current_dir, "assets", "test_audio_files", "[Dickson Lau]_8330-96674941_20251013035051(3360).wav")
audio_file = os.path.join(current_dir, "assets", "test_audio_files", "[Dickson Lau 0489]_8330-96674941_20251013012751(880).wav")

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

# print(f"📁 Using audio file: {audio_file}")
# print(f"✅ File exists: {os.path.exists(audio_file)}")
# print(f"📊 File size: {os.path.getsize(audio_file) / (1024*1024):.2f} MB\n")

# Convert Windows backslashes to forward slashes for LLM message
# (LLMs can misinterpret backslashes, but Windows accepts forward slashes)
audio_file_for_llm = audio_file.replace('\\', '/')
print(f"📁 Using audio file for LLM: {audio_file_for_llm}")

# messages = [HumanMessage(content=f"First extract metadata from the audio file '{audio_file_for_llm}', then diarize it with 2 speakers, then chop it into speaker segments with 50ms padding, transcribe all the segments to text using SenseVoiceSmall, and finally apply Cantonese text corrections to the transcriptions.")]
# result = agent.invoke(
#     {"messages": messages},
#     config={"recursion_limit": 50}  # Increase from default 25 to handle complex workflows
# )
# for m in result["messages"]:
#     m.pretty_print()

# print("\n" + "="*80)
# print("Workflow completed!")
# print("="*80)

SYSTEM_PROMPT = """You are a helpful assistant with access to audio processing tools. For audio analysis workflows:
1. Use extract_metadata_from_filename tool to extract metadata from audio filenames (broker name, client info, timestamps, etc.)
2. Use diarize_audio tool to identify speakers and when they spoke. It returns a JSON response with 'audio_filepath' and 'rttm_content' fields.
3. Use chop_audio_by_rttm tool with the 'audio_filepath' and 'rttm_content' values from the diarize_audio response to split the audio into speaker segments. The response will contain an 'output_dir' field.
4. Use transcribe_audio_segments tool with the 'output_dir' value from chop_audio_by_rttm to transcribe the chopped audio segments using SenseVoiceSmall.
5. Use correct_transcriptions tool to apply Cantonese text corrections to the transcriptions.
6. Use identify_stocks_in_conversation tool to analyze conversation text and identify stocks that were discussed, including stock names, symbols, quantities, and prices mentioned.

IMPORTANT: Each tool returns a JSON response. You must extract the specific fields from each response and pass them as arguments to the next tool. For example:
- After diarize_audio returns {"success": true, "audio_filepath": "...", "rttm_content": "...", ...}, extract the audio_filepath and rttm_content values and pass them to chop_audio_by_rttm.
- After chop_audio_by_rttm returns {"success": true, "output_dir": "...", ...}, extract the output_dir value and pass it to transcribe_audio_segments."""

# Create agent
agent = create_react_agent(
    model,
    tools,
    prompt=SYSTEM_PROMPT,
    #state_schema=AgentState,  # default
).with_config({"recursion_limit": 20})  #recursion_limit limits the number of steps the agent will run

result1 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": f"First extract metadata from the audio file '{audio_file_for_llm}', then diarize it with 2 speakers, then chop it into speaker segments with 50ms padding, transcribe all the segments to text using SenseVoiceSmall, and finally apply Cantonese text corrections to the transcriptions.",
            }
        ],
    }
)

print(result1["messages"])
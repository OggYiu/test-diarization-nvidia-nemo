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

from langchain.messages import SystemMessage, ToolMessage
import json

from typing import Literal
from langgraph.graph import StateGraph, START, END

# Invoke
from langchain.messages import HumanMessage
import os

# Use absolute path to avoid path resolution issues
current_dir = os.path.dirname(os.path.abspath(__file__))

# Function to recursively find all audio files in a directory
def get_audio_files_from_directory(directory, audio_extensions=('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
    """
    Recursively scan a directory and return all audio files found.
    
    Args:
        directory: The directory path to scan
        audio_extensions: Tuple of audio file extensions to look for
        
    Returns:
        List of absolute paths to audio files, sorted alphabetically
    """
    audio_files = []
    
    if not os.path.exists(directory):
        print(f"⚠️  Warning: Directory not found: {directory}")
        return audio_files
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(audio_extensions):
                audio_files.append(os.path.join(root, file))
    
    # Sort files for consistent processing order
    audio_files.sort()
    return audio_files

# Define the directory to scan for audio files
audio_directory = os.path.join(current_dir, "assets", "phone-recordings")

# Get all audio files from the directory
audio_files = get_audio_files_from_directory(audio_directory)

# Check if any audio files were found
if not audio_files:
    print(f"❌ No audio files found in: {audio_directory}")
    print("   Please add audio files (.wav, .mp3, .flac, .m4a, .ogg) to the directory.")
    exit(1)

# Verify all files exist (sanity check)
for audio_file in audio_files:
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        exit(1)

print(f"📁 Found {len(audio_files)} audio file(s) to process:")
for idx, audio_file in enumerate(audio_files, 1):
    relative_path = os.path.relpath(audio_file, audio_directory)
    print(f"   {idx}. {relative_path}")

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

2. Use diarize_audio tool to identify speakers and when they spoke. 
   - Pass only: audio_filepath, num_speakers, domain_type
   - Returns: dict with 'audio_filepath' and 'rttm_filepath' fields

3. Use chop_audio_by_rttm tool to split the audio into speaker segments.
   - Pass only: audio_filepath and rttm_filepath (from step 2)
   - Do NOT pass output_dir - it's automatically determined
   - Returns: string path to the directory containing chopped segments

4. Use transcribe_audio_segments tool to transcribe the chopped audio segments.
   - Pass the segments_directory path returned from step 3
   - Returns: string path to the transcription output file

IMPORTANT: Never specify output directories manually. All tools automatically organize outputs into agent/output/ subdirectories.
"""


# SYSTEM_PROMPT = """You are a helpful assistant with access to audio processing tools. For audio analysis workflows:
# 1. Use extract_metadata_from_filename tool to extract metadata from audio filenames (broker name, client info, timestamps, etc.)
# 2. Use diarize_audio tool to identify speakers and when they spoke. It returns a JSON response with 'audio_filepath' and 'rttm_content' fields.
# 3. Use chop_audio_by_rttm tool with the 'audio_filepath' and 'rttm_content' values from the diarize_audio response to split the audio into speaker segments. The response will contain an 'output_dir' field.
# 4. Use transcribe_audio_segments tool with the 'output_dir' value from chop_audio_by_rttm to transcribe the chopped audio segments using SenseVoiceSmall.
# 5. Use correct_transcriptions tool to apply Cantonese text corrections to the transcriptions.
# 6. Use identify_stocks_in_conversation tool to analyze conversation text and identify stocks that were discussed, including stock names, symbols, quantities, and prices mentioned.

# IMPORTANT: Each tool returns a JSON response. You must extract the specific fields from each response and pass them as arguments to the next tool. For example:
# - After diarize_audio returns {"success": true, "audio_filepath": "...", "rttm_content": "...", ...}, extract the audio_filepath and rttm_content values and pass them to chop_audio_by_rttm.
# - After chop_audio_by_rttm returns {"success": true, "output_dir": "...", ...}, extract the output_dir value and pass it to transcribe_audio_segments."""

# Create agent
agent = create_react_agent(
    model,
    tools,
    prompt=SYSTEM_PROMPT,
    #state_schema=AgentState,  # default
).with_config({"recursion_limit": 20})  #recursion_limit limits the number of steps the agent will run

# Process each audio file one by one
corrected_transcription_files = []

for idx, audio_file in enumerate(audio_files, 1):
    print(f"\n{'='*80}")
    print(f"Processing audio file {idx}/{len(audio_files)}: {os.path.basename(audio_file)}")
    print(f"{'='*80}\n")
    
    # Convert Windows backslashes to forward slashes for LLM message
    audio_file_for_llm = audio_file.replace('\\', '/')
    
    # Process this audio file through the pipeline
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    # "content": f"""
                    # First extract metadata from the audio file '{audio_file_for_llm}',
                    # then diarize it with 2 speakers,
                    # then chop it into speaker segments with 0ms padding,
                    # transcribe all the segments to text using SenseVoiceSmall,
                    # and finally apply Cantonese text corrections to the transcriptions.""",
                    
                    "content": f"""
                    First extract metadata from the audio file '{audio_file_for_llm}',
                    then diarize it with 2 speakers,
                    then chop it into speaker segments with 0ms padding,
                    transcribe all the segments to text using SenseVoiceSmall""",
                }
            ],
        }
    )
    
    # Extract the path to the corrected transcription file from the agent's response
    # The correct_transcriptions tool saves to transcriptions_text_corrected.txt
    output_dir = os.path.join(current_dir, "output", "transcriptions", os.path.splitext(os.path.basename(audio_file))[0])
    corrected_file = os.path.join(output_dir, "transcriptions_text_corrected.txt")
    
    if os.path.exists(corrected_file):
        corrected_transcription_files.append(corrected_file)
        print(f"✅ Successfully processed audio file {idx}/{len(audio_files)}")
    else:
        print(f"⚠️  Warning: Could not find corrected transcription file for {os.path.basename(audio_file)}")
    
    print(f"\n{'='*80}")
    print(f"Completed audio file {idx}/{len(audio_files)}")
    print(f"{'='*80}\n")


do_combined_transcription = False
if do_combined_transcription and corrected_transcription_files:
# Now identify stocks in the conversation(s)
    print(f"\n{'='*80}")
    print(f"Identifying stocks in conversation(s)")
    print(f"{'='*80}\n")

    # Combine all transcriptions if there are multiple files
    combined_transcription = ""
    
    for idx, trans_file in enumerate(corrected_transcription_files, 1):
        with open(trans_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if len(corrected_transcription_files) > 1:
            # Add separator between conversations if multiple files
            combined_transcription += f"\n\n--- Conversation {idx} (from {os.path.basename(os.path.dirname(trans_file))}) ---\n\n"
        
        combined_transcription += content
    
    # Use the agent to identify stocks in the combined conversation
    stock_result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Please identify stocks in the following conversation transcription:\n\n{combined_transcription}",
                }
            ],
        }
    )
    
    print(f"\n{'='*80}")
    print("Stock Identification Results")
    print(f"{'='*80}\n")
    print(stock_result["messages"][-1].content)
    
print(f"\n{'='*80}")
print("All processing completed!")
print(f"{'='*80}\n")
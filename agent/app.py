import os
import time
import warnings
import logging

# Suppress harmless warnings from NeMo, PyTorch, and OneLogger
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Megatron num_microbatches_calculator.*')
warnings.filterwarnings('ignore', message='.*Redirects are currently not supported.*')
warnings.filterwarnings('ignore', message='.*No exporters were provided.*')
warnings.filterwarnings('ignore', message='.*OneLogger.*')

# Suppress NeMo logging warnings
logging.getLogger('nemo').setLevel(logging.ERROR)
logging.getLogger('nemo_logging').setLevel(logging.ERROR)

from dotenv import load_dotenv

load_dotenv(os.path.join("..", ".env"), override=True)

# Configure LangSmith (optional - only if LANGSMITH_API_KEY is set in .env)
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "default")

import sys
import io
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Fix console encoding for Windows to support emoji characters
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from langchain.chat_models import init_chat_model
from tools.audio_chopper_tool import chop_audio_by_rttm
from tools.diarize_tool import diarize_audio
from tools.stt_tool import transcribe_audio_segments
from tools.metadata_tool import identify_speakers_from_filename
from tools.cantonese_corrector_tool import correct_transcriptions
from tools.stock_identifier_tool import identify_stocks_in_conversation
from tools.stock_verifier_tool import verify_stocks
from tools.stock_review_tool import generate_transaction_report

# Configure dspy once at module level to avoid threading issues
import dspy
lm = dspy.LM("ollama_chat/qwen3:32b", api_base="http://localhost:11434", api_key="", temperature=0.0)
dspy.configure(lm=lm)

model = ChatOpenAI(
    api_key="ollama",  # Not used, but required by ChatOpenAI
    # model="qwen3:14b",
    model="qwen3:30b-instruct",
    # base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    base_url="http://localhost:11434/v1",
    temperature=0.0
)

# Augment the LLM with tools
tools = [
    identify_speakers_from_filename,
    diarize_audio,
    chop_audio_by_rttm,
    transcribe_audio_segments,
    correct_transcriptions,
    identify_stocks_in_conversation,
    verify_stocks,
    generate_transaction_report
]

from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator

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
    trace_start = time.time()
    print(f"[TRACE {time.strftime('%H:%M:%S')}] Starting directory scan: {directory}")
    
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
    
    trace_elapsed = time.time() - trace_start
    print(f"[TRACE {time.strftime('%H:%M:%S')}] Directory scan completed in {trace_elapsed:.2f}s - Found {len(audio_files)} files")
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

SYSTEM_PROMPT = """You are a helpful assistant that processes phone conversation audio files to generate comprehensive transaction reports.

YOUR GOAL: Analyze phone conversations between brokers and clients to generate a detailed report of stock transactions discussed.

To achieve this goal, you MUST execute ALL tools in this EXACT order:

1. identify_speakers_from_filename - Extract metadata (broker, client, datetime) from filename
2. diarize_audio - Identify who speaks when (speaker diarization)
3. chop_audio_by_rttm - Split audio into individual speaker segments
4. transcribe_audio_segments - Convert speech to text for each segment
5. correct_transcriptions - Apply Cantonese language corrections
6. identify_stocks_in_conversation - Extract stock mentions and transaction details
7. verify_stocks - Verify stock codes and names against database
8. generate_transaction_report - FINAL STEP: Generate the comprehensive transaction report

CRITICAL RULES:
- Execute EVERY tool in the pipeline in the order listed above
- Complete each tool fully before moving to the next tool
- Do NOT skip any tools, especially the final generate_transaction_report tool
- Do NOT stop until the transaction report has been generated
- Each tool will tell you to "Continue with the next step in the pipeline" - you MUST continue
- Pass the outputs from previous tools as inputs to subsequent tools as needed
- For generate_transaction_report, you need BOTH the corrected transcription file AND the verified stocks JSON file

IMPORTANT: The pipeline is NOT complete until generate_transaction_report has been executed and the final report is generated.

"""

# Create agent
agent = create_react_agent(
    model,
    tools,
    prompt=SYSTEM_PROMPT,
    #state_schema=AgentState,  # default
).with_config({"recursion_limit": 30})  #recursion_limit limits the number of steps the agent will run (increased to 30 to accommodate 8-step pipeline)

# Process each audio file one by one
corrected_transcription_files = []

for idx, audio_file in enumerate(audio_files, 1):
    print(f"\n{'='*80}")
    print(f"Processing audio file {idx}/{len(audio_files)}: {os.path.basename(audio_file)}")
    print(f"{'='*80}\n")
    
    # Convert Windows backslashes to forward slashes for LLM message
    audio_file_for_llm = audio_file.replace('\\', '/')
    
    # Process this audio file through the pipeline
    trace_start = time.time()
    print(f"[TRACE {time.strftime('%H:%M:%S')}] Starting agent.invoke() for file: {os.path.basename(audio_file)}")
    
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"""
Analyze the phone conversation in '{audio_file_for_llm}' and generate a comprehensive transaction report.

Execute ALL tools in sequence to achieve this goal:
1. identify_speakers_from_filename (extract metadata from the audio file)
2. diarize_audio (identify speakers, use 2 speakers)
3. chop_audio_by_rttm (split into speaker segments)
4. transcribe_audio_segments (convert speech to text)
5. correct_transcriptions (apply Cantonese corrections)
6. identify_stocks_in_conversation (extract stock transactions)
7. verify_stocks (verify against database)
8. generate_transaction_report (FINAL - create the comprehensive report)

IMPORTANT: Do not stop until generate_transaction_report has been executed and the final transaction report is generated.
Each tool will provide outputs needed by subsequent tools - pass these along as you proceed.
""",
                }
            ],
        }
    )
    
    trace_elapsed = time.time() - trace_start
    print(f"[TRACE {time.strftime('%H:%M:%S')}] agent.invoke() completed in {trace_elapsed:.2f}s ({trace_elapsed/60:.2f} minutes)")
    
    # Extract the path to the corrected transcription file from the agent's response
    # The correct_transcriptions tool saves to transcriptions_text_corrected.txt
    trace_start = time.time()
    print(f"[TRACE {time.strftime('%H:%M:%S')}] Checking for output files...")
    
    output_dir = os.path.join(current_dir, "output", "transcriptions", os.path.splitext(os.path.basename(audio_file))[0])
    corrected_file = os.path.join(output_dir, "transcriptions_text_corrected.txt")
    
    trace_elapsed = time.time() - trace_start
    print(f"[TRACE {time.strftime('%H:%M:%S')}] File check completed in {trace_elapsed:.4f}s")
    
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
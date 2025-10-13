import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from .diarization_service import diarize_audio, get_rttm_content

app = FastAPI(
    title="Speaker Diarization API",
    description="API for performing speaker diarization on audio files",
    version="1.0.0"
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Speaker Diarization API is running",
        "endpoints": {
            "POST /diarize": "Upload audio file for diarization",
            "GET /": "This health check endpoint"
        }
    }


@app.post("/diarize")
async def diarize(
    audio_file: UploadFile = File(..., description="Audio file to diarize (WAV format recommended)"),
    num_speakers: Optional[int] = Form(2, description="Number of speakers (default: 2)")
):
    """
    Perform speaker diarization on an uploaded audio file.
    
    Args:
        audio_file: Audio file to process (WAV, MP3, etc.)
        num_speakers: Expected number of speakers (default: 2)
    
    Returns:
        JSON response containing:
        - rttm_content: The content of the generated RTTM file
        - audio_filename: Original filename
        - num_speakers: Number of speakers used
    """
    
    # Validate file
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp(prefix="diarization_")
    
    try:
        # Save uploaded file to temporary location
        temp_audio_path = os.path.join(temp_dir, audio_file.filename)
        with open(temp_audio_path, "wb") as buffer:
            content = await audio_file.read()
            buffer.write(content)
        
        print(f"Processing audio file: {audio_file.filename}")
        print(f"Temporary audio path: {temp_audio_path}")
        
        # Create output directory for diarization results
        output_dir = os.path.join(temp_dir, "output")
        
        # Perform diarization
        try:
            diarize_audio(temp_audio_path, out_dir=output_dir, num_speakers=num_speakers)
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Diarization failed: {str(e)}"
            )
        
        # Get RTTM file content
        rttm_content = get_rttm_content(output_dir)
        
        if not rttm_content:
            raise HTTPException(
                status_code=500,
                detail="Diarization completed but no RTTM file was generated"
            )
        
        return JSONResponse(content={
            "status": "success",
            "audio_filename": audio_file.filename,
            "num_speakers": num_speakers,
            "rttm_content": rttm_content,
            "message": "Diarization completed successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"Warning: Could not clean up temporary directory {temp_dir}: {e}")


if __name__ == "__main__":
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)


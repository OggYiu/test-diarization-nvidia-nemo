# Diarization API Usage Guide

This document provides instructions on how to build, run, and use the Diarization API. The API allows you to upload an audio file, process it through a diarization and speech-to-text pipeline, and retrieve the results.

## 1. Building the Docker Image

The application is containerized using Docker. To build the image, navigate to the project's root directory and run:

```bash
docker build -t diarization-api:latest ./diarization-api
```

This command builds the image and tags it as `diarization-api:latest`.

## 2. Running the API

Once the image is built, you can run it as a container. The API server runs on port 8000 inside the container.

```bash
docker run -p 8000:8000 --rm --name diarization-api-container diarization-api:latest
```

- `-p 8000:8000`: Maps port 8000 on your host machine to port 8000 in the container.
- `--rm`: Automatically removes the container when it stops.
- `--name`: Assigns a convenient name to the running container.

The API will now be accessible at `http://localhost:8000`.

## 3. API Endpoints

The API provides three main endpoints.

### Health Check

- **Endpoint**: `GET /`
- **Description**: A simple endpoint to verify that the API is running.
- **Example Request**:
  ```bash
  curl http://localhost:8000/
  ```
- **Success Response**:
  ```json
  {
    "status": "ok"
  }
  ```

### Process Audio File

- **Endpoint**: `POST /process`
- **Description**: Submits an audio file for processing. The processing is done asynchronously in the background. The endpoint immediately returns a `job_id` which you can use to check the status and retrieve the results later.
- **Request Type**: `multipart/form-data`
- **Parameters**:
  | Name           | Type    | In   | Description                                                  |
  | -------------- | ------- | ---- | ------------------------------------------------------------ |
  | `file`         | file    | body | **Required**. The audio file to process (e.g., a `.wav` file). |
  | `num_speakers` | integer | body | *Optional*. The number of speakers to detect. Default: `2`.  |
  | `skip_llm`     | boolean | body | *Optional*. If `true`, skips the final LLM analysis step. Default: `false`. |

- **Example Request (`curl`)**:
  ```bash
  curl -X POST http://localhost:8000/process \
    -F "file=@/path/to/your/audio.wav" \
    -F "num_speakers=2"
  ```

- **Success Response**:
  ```json
  {
    "job_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef"
  }
  ```

### Check Job Status

- **Endpoint**: `GET /status/{job_id}`
- **Description**: Retrieves the status and results of a processing job.
- **Path Parameter**:
  - `job_id`: The unique ID returned by the `/process` endpoint.

- **Example Request (`curl`)**:
  ```bash
  curl http://localhost:8000/status/a1b2c3d4-e5f6-7890-1234-567890abcdef
  ```

- **Example Responses**:

  - **While processing**:
    ```json
    {
      "status": "processing",
      "stage": "Running Diarization"
    }
    ```
  - **On completion**:
    ```json
    {
      "status": "completed",
      "stage": "LLM Analysis Complete",
      "results": {
        "diarization_rttm": "/app/work/a1b2c3d4.../diarization.rttm",
        "transcription_json": "/app/work/a1b2c3d4.../transcriptions.json",
        "conversation_txt": "/app/work/a1b2c3d4.../conversation.txt",
        "llm_analysis": "Summary of the conversation..."
      }
    }
    ```
  - **If job fails**:
    ```json
    {
      "status": "failed",
      "stage": "Diarization Failed",
      "error": "Details about the error..."
    }
    ```
  - **If `job_id` not found**:
    ```json
    {
      "detail": "Job not found"
    }
    ```

## 4. Typical Workflow

1.  **Submit Audio**: Send a `POST` request to `/process` with your audio file.
2.  **Get Job ID**: Receive the `job_id` from the response.
3.  **Poll for Status**: Periodically send `GET` requests to `/status/{job_id}` until the `"status"` field is `completed` or `failed`.
4.  **Retrieve Results**: Once completed, the results will be available in the response body.
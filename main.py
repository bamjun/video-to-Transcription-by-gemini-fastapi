import fastapi
import os
import uvicorn
import google.generativeai as genai
from fastapi import File, UploadFile, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Union
import mimetypes  # To guess file type
from dotenv import load_dotenv
import tempfile  # 임시 파일 사용을 위해 추가
import asyncio  # 파일 상태 확인 대기를 위해 추가
import time  # 타임아웃 로직을 위해 추가
from google.api_core import exceptions as google_exceptions

load_dotenv()

# --- Configuration ---
# Best practice: Store your API key securely (e.g., environment variable)
# Make sure to set this environment variable before running the script:
# export GOOGLE_API_KEY="YOUR_API_KEY"
try:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except KeyError:
    print("ERROR: GOOGLE_API_KEY environment variable not set.")
    exit()
except Exception as e:
    print(f"Error configuring Generative AI SDK: {e}")
    exit()


# --- Pydantic Models ---
# Since Gemini might not return structured diarization reliably,
# we'll return the generated text as a single string for simplicity.
class GeminiTranscriptionResponse(BaseModel):
    transcribed_text: str
    model_used: str = "gemini-2.5-pro-exp-03-25"  # Model name will be set dynamically

# 새로운 모델: 파일 정보를 담기 위한 모델
class FileInfo(BaseModel):
    name: str
    display_name: Optional[str] = None
    state: str
    create_time: str # ISO 8601 format string
    size_bytes: Optional[int] = None
    uri: Optional[str] = None
    
class FileStatusResponse(BaseModel):
    active: bool
    status: str

# --- Constants ---
FILE_PROCESSING_TIMEOUT_SECONDS = 3000  # 50 minutes timeout for file processing
FILE_PROCESSING_CHECK_INTERVAL_SECONDS = 30  # Check every 30 seconds

# --- FastAPI App ---
app = fastapi.FastAPI()

# 새로운 엔드포인트: 업로드된 파일 목록 및 상태 확인
@app.get("/list_uploaded_files/", response_model=list[FileInfo])
async def list_uploaded_files():
    """
    Lists all files currently uploaded to the Gemini File API for this API key
    and shows their status (e.g., ACTIVE, PROCESSING).
    """
    try:
        print("Fetching list of uploaded files from Gemini File API...")
        listed_files = genai.list_files()
        file_infos = []
        for f in listed_files:
            # Convert datetime object to ISO string for JSON serialization
            create_time_str = f.create_time.isoformat() if f.create_time else "Unknown"
            file_infos.append(FileInfo(
                name=f.name,
                display_name=f.display_name,
                state=f.state.name,
                create_time=create_time_str,
                size_bytes=f.size_bytes,
                uri=f.uri
            ))
        print(f"Found {len(file_infos)} files.")
        return file_infos
    except Exception as e:
        print(f"Error listing files from Gemini API: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list uploaded files: {e}")

# 새로운 엔드포인트: 기존 파일 ID로 상태 확인 및 트랜스크립션 요청
@app.get("/transcribe_file/{file_id}/", response_model=Union[GeminiTranscriptionResponse, FileStatusResponse])
async def transcribe_existing_file(file_id: str, model: str = "gemini-2.5-pro-exp-03-25"):
    """
    Checks the status of an existing file on Gemini File API using its ID.
    If the file is ACTIVE, performs transcription with speaker diarization.
    If not ACTIVE, returns the current status.
    
    models: gemini-2.5-pro-exp-03-25 gemini-1.5-pro-latest
    """
    full_file_name = f"files/{file_id}"
    print(f"Checking status and potentially transcribing file: {full_file_name}")

    try:
        # 1. 파일 상태 확인
        print(f"Getting file info for {full_file_name}...")
        current_file = genai.get_file(full_file_name)
        print(f"File state for {full_file_name} is {current_file.state.name}")

        # 2. 상태에 따른 분기 처리
        if current_file.state.name == "ACTIVE":
            print(f"File {full_file_name} is ACTIVE. Proceeding with transcription...")

            # --- Prompt Engineering (재사용) ---
            prompt = """Please transcribe the audio content of the provided video/audio file.
Identify different speakers in the conversation to the best of your ability. You can label them as "Speaker 1", "Speaker 2", etc., or use contextual clues if possible.
Present the transcription clearly, indicating who is speaking for each part of the conversation.
Also, include approximate timestamps (e.g., [00:01], [02:15]) for each speaker’s section to provide temporal context.
Note: Precise, timestamped speaker diarization like dedicated speech APIs might not be possible, but please provide the most informative and time-referenced speaker-differentiated transcript you can generate.
"""

            # --- Select Gemini Model (대용량 파일 처리 가능한 모델 선택) ---
            model = genai.GenerativeModel(model)
            print(f"Using model: {model._model_name}")

            # --- Call Gemini API --- #
            try:
                print(f"Sending prompt and file reference ({current_file.name}) to Gemini model...")
                response = await model.generate_content_async([prompt, current_file])
                print("Received response from Gemini.")

                # --- Extract Text (재사용) ---
                if not response.parts:
                    print("Warning: Gemini response has no parts.")
                    generated_text = "Model did not return any content."
                elif hasattr(response, 'text'):
                    generated_text = response.text
                else:
                    try:
                        generated_text = response.parts[0].text
                    except (IndexError, AttributeError):
                        print(f"Warning: Could not extract text from response parts. Response: {response}")
                        block_reason = "Unknown"
                        try:
                            if response.prompt_feedback and response.prompt_feedback.block_reason:
                                block_reason = response.prompt_feedback.block_reason.name
                        except AttributeError:
                            pass
                        generated_text = f"Could not extract text. Reason: {block_reason}"

                # 성공 응답 반환 (기존과 동일한 형식)
                return GeminiTranscriptionResponse(transcribed_text=generated_text, model_used=model._model_name)

            except Exception as e:
                print(f"Error calling Gemini API for transcription on {current_file.name}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to transcribe file {full_file_name}: {e}")

        else: # ACTIVE 상태가 아닐 경우
            print(f"File {full_file_name} is not ACTIVE (state: {current_file.state.name}). Returning status.")
            # 상태 정보 응답 반환
            return FileStatusResponse(active=False, status=current_file.state.name)

    except google_exceptions.NotFound:
        print(f"Error: File {full_file_name} not found.")
        raise HTTPException(status_code=404, detail=f"File with ID {file_id} not found.")
    except Exception as e:
        print(f"Error checking or processing file {full_file_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check or process file {full_file_name}: {e}")

@app.post("/transcribe_gemini/", response_model=GeminiTranscriptionResponse)
async def transcribe_with_gemini(file: UploadFile = File(...)):
    """
    Uploads a video/audio file and uses Gemini
    to generate a transcript, attempting speaker identification.
    Uses the Gemini File API for large files and waits for the file to be ACTIVE.
    """
    # --- Basic File Validation (Optional but recommended) ---
    content_type = file.content_type
    if not content_type or not (
        content_type.startswith("video/") or content_type.startswith("audio/")
    ):
        # Try guessing if content_type is generic
        guessed_type, _ = mimetypes.guess_type(file.filename or "unknown")
        if not guessed_type or not (
            guessed_type.startswith("video/") or guessed_type.startswith("audio/")
        ):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {content_type or guessed_type or 'unknown'}. Please upload video or audio.",
            )
        content_type = guessed_type  # Use guessed type if valid
    print(f"Processing file: {file.filename}, Content-Type: {content_type}")

    # --- Read File Content and Upload using File API ---
    temp_file_path = None
    uploaded_file = None
    start_time = time.time()
    try:
        # Create a temporary file to store the upload
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.filename or "")[1]
        ) as temp_file:
            temp_file_path = temp_file.name
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            temp_file.write(content)
            print(f"Saved uploaded file temporarily to: {temp_file_path}")

        # Upload the file using the File API
        print(f"Uploading temporary file {temp_file_path} to Gemini File API...")
        uploaded_file = genai.upload_file(
            path=temp_file_path, display_name=file.filename
        )
        print(
            f"Successfully initiated upload for file: {uploaded_file.name} ({uploaded_file.display_name})"
        )

        # --- Wait for File Processing --- #
        print(f"Waiting for file {uploaded_file.name} to become ACTIVE...")
        while True:
            current_file = genai.get_file(uploaded_file.name)
            if current_file.state.name == "ACTIVE":
                print(f"File {uploaded_file.name} is now ACTIVE.")
                break
            elif current_file.state.name == "PROCESSING":
                elapsed_time = time.time() - start_time
                if elapsed_time > FILE_PROCESSING_TIMEOUT_SECONDS:
                    raise HTTPException(
                        status_code=504,
                        detail=f"File processing timed out after {FILE_PROCESSING_TIMEOUT_SECONDS} seconds.",
                    )
                print(
                    f"File state is {current_file.state.name}, waiting {FILE_PROCESSING_CHECK_INTERVAL_SECONDS}s... (Elapsed: {elapsed_time:.0f}s)"
                )
                await asyncio.sleep(FILE_PROCESSING_CHECK_INTERVAL_SECONDS)
            elif current_file.state.name == "FAILED":
                raise HTTPException(
                    status_code=500,
                    detail=f"File processing failed on Gemini side. State: {current_file.state.name}",
                )
            else:
                # Handle unexpected states if necessary
                raise HTTPException(
                    status_code=500,
                    detail=f"Unexpected file state: {current_file.state.name}",
                )

    except HTTPException:  # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle file reading, temp file creation, upload, or status check errors
        raise HTTPException(
            status_code=500, detail=f"Failed during file preparation or upload: {e}"
        )
    finally:
        # Clean up the local temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Cleaned up temporary file: {temp_file_path}")

    # Ensure uploaded_file is valid before proceeding
    if not uploaded_file or genai.get_file(uploaded_file.name).state.name != "ACTIVE":
        raise HTTPException(
            status_code=500,
            detail="File upload did not complete successfully or file is not active.",
        )

    # --- Prompt Engineering ---
    # Ask Gemini to transcribe and identify speakers.
    # Be explicit that perfect diarization might not happen.
    prompt = """Please transcribe the audio content of the provided video/audio file.
Identify different speakers in the conversation to the best of your ability. You can label them as "Speaker 1", "Speaker 2", etc., or use contextual clues if possible.
Present the transcription clearly, indicating who is speaking for each part of the conversation.
Also, include approximate timestamps (e.g., [00:01], [02:15]) for each speaker’s section to provide temporal context.
Note: Precise, timestamped speaker diarization like dedicated speech APIs might not be possible, but please provide the most informative and time-referenced speaker-differentiated transcript you can generate.
"""

    # --- Select Gemini Model ---
    # Use the latest available model supporting video/audio input
    model = genai.GenerativeModel(
        "gemini-2.5-pro-exp-03-25"
    )  # Or gemini-1.5-flash-latest for faster/cheaper option if sufficient

    # --- Call Gemini API --- #
    print(
        f"Sending prompt and file reference ({uploaded_file.name}) to Gemini model: {model._model_name}..."
    )
    try:
        # Combine the prompt and the ACTIVE uploaded file object
        response = await model.generate_content_async([prompt, uploaded_file])

        print("Received response from Gemini.")

        # --- Extract Text ---
        # Handle potential errors or safety blocks in the response
        if not response.parts:
            print("Warning: Gemini response has no parts.")
            generated_text = "Model did not return any content."
        elif hasattr(response, "text"):
            generated_text = response.text
        else:
            # Fallback or handle cases where .text might not be directly available
            # (e.g., if blocked due to safety settings)
            try:
                # Try accessing the first part's text if available
                generated_text = response.parts[0].text
            except (IndexError, AttributeError):
                print(
                    f"Warning: Could not extract text from response parts. Response: {response}"
                )
                # Check for safety feedback specifically
                block_reason = "Unknown"
                try:
                    if (
                        response.prompt_feedback
                        and response.prompt_feedback.block_reason
                    ):
                        block_reason = response.prompt_feedback.block_reason.name
                except AttributeError:
                    pass  # feedback might not exist
                generated_text = f"Could not extract text. Check model response/safety settings. Reason: {block_reason}"

    except Exception as e:
        print(f"Error calling Gemini API with file {uploaded_file.name}: {e}")
        # You might want to check the type of exception for more specific handling
        # e.g., google.api_core.exceptions.ResourceExhausted for quota issues
        raise HTTPException(
            status_code=500, detail=f"Failed to process file with Gemini: {e}"
        )
    finally:
        # Optionally delete the file from Gemini server now that processing is done
        print(f"Deleting uploaded file {uploaded_file.name} from Gemini...")
        try:
            genai.delete_file(uploaded_file.name)
            print(f"Successfully deleted file {uploaded_file.name}.")
        except Exception as delete_err:
            print(f"Warning: Failed to delete file {uploaded_file.name} from Gemini: {delete_err}")
        pass

    # --- Return Response ---
    return GeminiTranscriptionResponse(
        transcribed_text=generated_text, model_used=model._model_name
    )

# --- Run the app ---
if __name__ == "__main__":
    print("Starting FastAPI server for Gemini Transcription...")
    uvicorn.run(
        "main:app", host="0.0.0.0", port=8002, reload=True
    )  # Use a different port (e.g., 8002)

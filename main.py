import fastapi
import os
import uvicorn
import google.generativeai as genai
from fastapi import File, UploadFile, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
import mimetypes # To guess file type

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
    model_used: str = "gemini-2.5-pro-exp-03-25" # Or the specific model you target

# --- FastAPI App ---
app = fastapi.FastAPI()

@app.post("/transcribe_gemini/", response_model=GeminiTranscriptionResponse)
async def transcribe_with_gemini(file: UploadFile = File(...)):
    """
    Uploads a video/audio file and uses Gemini
    to generate a transcript, attempting speaker identification.
    """
    # --- Basic File Validation (Optional but recommended) ---
    content_type = file.content_type
    if not content_type or not (content_type.startswith("video/") or content_type.startswith("audio/")):
         # Try guessing if content_type is generic
         guessed_type, _ = mimetypes.guess_type(file.filename or "unknown")
         if not guessed_type or not (guessed_type.startswith("video/") or guessed_type.startswith("audio/")):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {content_type or guessed_type or 'unknown'}. Please upload video or audio.",
            )
         content_type = guessed_type # Use guessed type if valid
    print(f"Processing file: {file.filename}, Content-Type: {content_type}")


    # --- Read File Content ---
    try:
        file_bytes = await file.read()
        if not file_bytes:
             raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        print(f"Read {len(file_bytes)} bytes from uploaded file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read uploaded file: {e}")

    # --- Prepare for Gemini API ---
    # Note: Large files might exceed API limits for direct upload.
    # For larger files (> limits), upload to a service like Google Cloud Storage
    # and provide the URI to Gemini instead of raw bytes.
    video_file_part = {
        "mime_type": content_type,
        "data": file_bytes,
    }

    # --- Prompt Engineering ---
    # Ask Gemini to transcribe and identify speakers.
    # Be explicit that perfect diarization might not happen.
    prompt = """Please transcribe the audio content of the provided video/audio file.
Identify different speakers in the conversation to the best of your ability. You can label them as "Speaker 1", "Speaker 2", etc., or use contextual clues if possible.
Present the transcription clearly, indicating who is speaking for each part of the conversation.
Note: Precise, timestamped speaker diarization like dedicated speech APIs might not be possible, but please provide the most informative speaker-differentiated transcript you can generate.
"""

    # --- Select Gemini Model ---
    # Use the latest available model supporting video/audio input
    model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25") # Or gemini-1.5-flash-latest for faster/cheaper option if sufficient

    # --- Call Gemini API ---
    print(f"Sending data to Gemini model: {model._model_name}...")
    try:
        # Combine the prompt and the file data
        # response = model.generate_content([prompt, video_file_part], stream=False) # stream=True for chunked response

        # Use generate_content asynchronously if preferred within FastAPI
        response = await model.generate_content_async([prompt, video_file_part])

        print("Received response from Gemini.")

        # --- Extract Text ---
        # Handle potential errors or safety blocks in the response
        if not response.parts:
             print("Warning: Gemini response has no parts.")
             generated_text = "Model did not return any content."
        elif hasattr(response, 'text'):
             generated_text = response.text
        else:
             # Fallback or handle cases where .text might not be directly available
             # (e.g., if blocked due to safety settings)
             try:
                 # Try accessing the first part's text if available
                 generated_text = response.parts[0].text
             except (IndexError, AttributeError):
                 print(f"Warning: Could not extract text from response parts. Response: {response}")
                 generated_text = f"Could not extract text. Check model response/safety settings. Reason: {response.prompt_feedback.block_reason if response.prompt_feedback else 'Unknown'}"


    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        # You might want to check the type of exception for more specific handling
        # e.g., google.api_core.exceptions.ResourceExhausted for quota issues
        raise HTTPException(status_code=500, detail=f"Failed to process file with Gemini: {e}")

    # --- Return Response ---
    return GeminiTranscriptionResponse(transcribed_text=generated_text)


@app.get("/")
async def read_root_gemini():
    return {"message": "Gemini Transcription API. POST to /transcribe_gemini/"}

# --- Run the app ---
if __name__ == "__main__":
    print("Starting FastAPI server for Gemini Transcription...")
    uvicorn.run("main_gemini:app", host="0.0.0.0", port=8002, reload=True) # Use a different port (e.g., 8002)
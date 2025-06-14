import os
import string
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the output folders exist
os.makedirs("success", exist_ok=True)
os.makedirs("revision", exist_ok=True)

# Valid letters: A–Z plus Ñ
VALID_LETTERS = set(string.ascii_lowercase) | {"ñ"}

@app.post("/process/")
async def process_image(
    label: str = Form(...),
    image: UploadFile = File(...)
):
    # Normalize and validate the label
    letter = label.strip().lower()
    print(f"[PROCESS] Received label: {letter}")
    if letter not in VALID_LETTERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid label: '{label}'. Must be one letter A–Z or Ñ."
        )

    # Read the uploaded image content
    content = await image.read()

    # Perform your external validation (placeholder here)
    try:
        is_valid = True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    # Save the image in the appropriate folder
    folder = "success" if is_valid else "revision"
    file_path = os.path.join(folder, image.filename)
    with open(file_path, "wb") as out_file:
        out_file.write(content)

    # Return a simple JSON response
    return JSONResponse(content={"success": is_valid})

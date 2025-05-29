# server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle

# 1. Load your trained pipeline
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# 2. Define the input schema
class LandmarksRequest(BaseModel):
    # expecting a list of 21 [x,y,z] points
    landmarks: list[list[float]]

# 3. Create the FastAPI app
app = FastAPI()

# 4. (Optional) Enable CORS so your frontend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # or your specific frontend URL
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# 5. Prediction endpoint
@app.post("/predict")
def predict(req: LandmarksRequest):
    # Flatten or preprocess as your pipeline expects
    # For example, flatten 21×3 → 63-element vector:
    X = [coord for point in req.landmarks for coord in point]
    # Run through your pipeline
    pred = model.predict([X])[0]
    return {"letter": pred}

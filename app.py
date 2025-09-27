from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from utils.recommend import get_recommendations
from dotenv import load_dotenv
import requests
import os
import pandas as pd

load_dotenv()

app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

colleges_df = pd.read_csv("data/colleges.csv")
colleges_locations_df = pd.read_csv("data/colleges_locations.csv")

class PredictRequest(BaseModel):
    location: str
    interests: str
    field: str

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_API_TOKEN"),
)
class CareerChatRequest(BaseModel):
    question: str
@app.post("/api/career-chat")
async def career_chat(data: CareerChatRequest):
    try:
        completion = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
            messages=[
                {
                    "role": "user",
                    "content": data.question
                }
            ],
        )
        answer = completion.choices[0].message.content
        return {"answer": answer}
    except Exception as e:
        return {"answer": f"API request failed: {str(e)}"}
    
@app.post("/api/predict")
async def predict_college(data: PredictRequest):
    recommendations = get_recommendations(
        colleges_df,
        data.location,
        data.interests,
        data.field
    )
    return recommendations

# Optional health check
@app.get("/api/health")
async def health():
    return {"status": "ok"}

# Add to d:\Damodarone\SIH\college\CareerCompass\CareerCompassAi\app.py

@app.get("/api/colleges")
async def get_colleges():
    # Merge colleges and locations on college_name
    merged = pd.merge(colleges_df, colleges_locations_df, on="college_name", how="left")
    colleges = merged.to_dict(orient="records")
    return colleges

@app.get("/api/college-location")
async def college_location(name: str):
    # Find college in locations file
    college = colleges_locations_df[colleges_locations_df['college_name'].str.lower() == name.lower()]
    if not college.empty:
        row = college.iloc[0]
        return {
            "college_name": row["college_name"],
            "latitude": row["latitude"],
            "longitude": row["longitude"]
        }
    return {"error": "College not found"}
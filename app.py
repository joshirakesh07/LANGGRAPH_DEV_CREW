import os
import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="LangGraph Dev Crew",
    version="1.0",
    description="AI Development Crew Agent"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# GOOGLE API KEY
# =========================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")


# =========================================================
# INPUT / OUTPUT
# =========================================================

class DevCrewInput(BaseModel):
    input: str


class DevCrewOutput(BaseModel):
    output: str


# =========================================================
# GEMINI MODEL
# =========================================================

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)


# =========================================================
# DEV CREW FUNCTION
# =========================================================

def dev_crew(data):

    user_input = data["input"]

    prompt = f"""
You are an AI Development Crew Agent.

Analyze the user's request and provide a useful software-development
plan.

User request:
{user_input}

Your response should contain:

1. Requirement Analysis
2. Recommended Technology Stack
3. Development Steps
4. Suggested Project Structure
5. Important Implementation Notes
6. Testing Plan
7. Deployment Suggestions

Keep the answer clear and practical.
"""

    response = model.invoke(prompt)

    return {
        "output": response.content
    }


# =========================================================
# RUNNABLE
# =========================================================

devcrew = RunnableLambda(dev_crew).with_types(
    input_type=DevCrewInput,
    output_type=DevCrewOutput
)


# =========================================================
# LANGSERVE ROUTES
# =========================================================

add_routes(
    app,
    devcrew,
    path="/devcrew"
)


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():
    return {
        "message": "LangGraph Dev Crew is running"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )

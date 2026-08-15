import os
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="LangGraph Dev Crew",
    version="1.0"
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
# API KEY
# =========================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")


# =========================================================
# GEMINI
# =========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)


# =========================================================
# REQUEST MODEL
# =========================================================

class DevCrewRequest(BaseModel):
    input: str


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():
    return {
        "message": "LangGraph Dev Crew is running"
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# =========================================================
# AI API
# =========================================================

@app.post("/devcrew/invoke")
async def invoke_devcrew(data: DevCrewRequest):

    prompt = f"""
You are an AI Development Crew Agent.

Analyze the user's request and provide a practical software
development plan.

User Request:
{data.input}

Give the response in this format:

1. Requirement Analysis
2. Recommended Technology Stack
3. Development Steps
4. Project Structure
5. Implementation Notes
6. Testing Plan
7. Deployment Suggestions

Keep the explanation clear and useful for a B.Tech student.
"""

    try:

        response = llm.invoke(prompt)

        return {
            "output": response.content
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# =========================================================
# PLAYGROUND
# =========================================================

@app.get("/devcrew/playground/", response_class=HTMLResponse)
async def playground():

    html = """
<!DOCTYPE html>

<html>

<head>

<title>LangGraph Dev Crew</title>

<meta name="viewport"
content="width=device-width, initial-scale=1">

<style>

body {
    margin: 0;
    padding: 0;
    font-family: Arial, sans-serif;
    background: #f4f6f8;
}

.container {
    width: 700px;
    max-width: 90%;
    margin: 60px auto;
    background: white;
    padding: 35px;
    border-radius: 15px;
    box-shadow: 0 5px 25px rgba(0,0,0,0.12);
}

h1 {
    text-align: center;
    color: #222;
}

.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 30px;
}

label {
    display: block;
    font-weight: bold;
    margin-bottom: 10px;
}

textarea {
    width: 100%;
    height: 150px;
    padding: 15px;
    box-sizing: border-box;
    border: 1px solid #ccc;
    border-radius: 10px;
    font-size: 16px;
    resize: vertical;
}

button {
    width: 100%;
    margin-top: 20px;
    padding: 15px;
    border: none;
    border-radius: 10px;
    background: #2563eb;
    color: white;
    font-size: 17px;
    cursor: pointer;
}

button:hover {
    background: #1d4ed8;
}

button:disabled {
    background: #999;
    cursor: not-allowed;
}

#result {
    margin-top: 25px;
    padding: 20px;
    background: #f1f5f9;
    border-radius: 10px;
    white-space: pre-wrap;
    line-height: 1.6;
    display: none;
}

.loading {
    text-align: center;
    color: #666;
}

</style>

</head>


<body>

<div class="container">

<h1>🤖 LangGraph Dev Crew</h1>

<p class="subtitle">
AI Development Planning Agent
</p>


<label>
Enter your request
</label>


<textarea
id="input"
placeholder="Example: Create a development plan for an AI/ML Engineer..."
></textarea>


<button
id="startButton"
onclick="runAgent()">

▶ Start

</button>


<div id="result"></div>


</div>


<script>

async function runAgent() {

    const input =
        document.getElementById("input").value.trim();

    const button =
        document.getElementById("startButton");

    const result =
        document.getElementById("result");


    if (!input) {

        alert("Please enter your request.");

        return;

    }


    button.disabled = true;

    button.innerText = "⏳ Processing...";


    result.style.display = "block";

    result.innerHTML =
        '<div class="loading">AI Agent is working...</div>';


    try {

        const response = await fetch(
            "/devcrew/invoke",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    input: input
                })
            }
        );


        const data = await response.json();


        if (data.output) {

            result.innerText = data.output;

        }

        else if (data.error) {

            result.innerText =
                "Error: " + data.error;

        }

        else {

            result.innerText =
                JSON.stringify(data, null, 2);

        }


    }

    catch (error) {

        result.innerText =
            "Connection error: " + error.message;

    }


    button.disabled = false;

    button.innerText = "▶ Start";

}

</script>


</body>

</html>
"""

    return HTMLResponse(content=html)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 8000)
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )

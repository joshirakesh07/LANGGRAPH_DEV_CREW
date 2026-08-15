import os
from typing import TypedDict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, START, END


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(title="LangGraph Dev Crew")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# GEMINI API KEY
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")


# =========================================================
# GEMINI MODEL
# =========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=GEMINI_API_KEY
)


# =========================================================
# LANGGRAPH STATE
# =========================================================

class AgentState(TypedDict):
    input: str
    output: str


# =========================================================
# DEV CREW AGENT
# =========================================================

def developer_agent(state: AgentState):

    user_input = state["input"]

    prompt = f"""
You are LangGraph Dev Crew, an AI Development Planning Agent.

Analyze the user's request and provide a clear, practical,
well-structured answer.

If the user asks for a development plan, include:

1. Project Overview
2. Required Skills
3. Technologies and Tools
4. Development Steps
5. Project Structure
6. Three Portfolio Projects
7. Learning Roadmap

If the user asks a technical question, explain it clearly
and give examples when useful.

User Request:

{user_input}
"""

    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    return {
        "input": user_input,
        "output": response.content
    }


# =========================================================
# CREATE LANGGRAPH
# =========================================================

graph = StateGraph(AgentState)

graph.add_node(
    "developer_agent",
    developer_agent
)

graph.add_edge(
    START,
    "developer_agent"
)

graph.add_edge(
    "developer_agent",
    END
)

dev_crew = graph.compile()


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message": "LangGraph Dev Crew is running"
    }


# =========================================================
# API ENDPOINT
# =========================================================

@app.post("/devcrew/invoke")
async def invoke_agent(data: dict):

    user_input = data.get("input", "").strip()

    if not user_input:

        return {
            "error": "Please enter a request."
        }

    try:

        result = dev_crew.invoke(
            {
                "input": user_input,
                "output": ""
            }
        )

        return {
            "output": result["output"]
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# =========================================================
# PLAYGROUND
# =========================================================

@app.get(
    "/devcrew/playground/",
    response_class=HTMLResponse
)
async def playground():

    return """
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>LangGraph Dev Crew</title>

<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background: #f4f6f8;
}

.container {

    width: 820px;

    max-width: 92%;

    margin: 70px auto;

    background: white;

    padding: 40px;

    border-radius: 18px;

    box-shadow:
        0 10px 35px
        rgba(0, 0, 0, 0.12);
}

h1 {

    text-align: center;

    margin: 0;

    font-size: 34px;

    color: #111827;
}

.subtitle {

    text-align: center;

    color: #6b7280;

    margin-top: 12px;

    margin-bottom: 35px;

    font-size: 18px;
}

label {

    display: block;

    font-size: 18px;

    font-weight: bold;

    margin-bottom: 10px;
}

textarea {

    width: 100%;

    height: 160px;

    padding: 16px;

    border: 1px solid #cbd5e1;

    border-radius: 10px;

    font-size: 16px;

    resize: vertical;

    outline: none;
}

textarea:focus {

    border-color: #2563eb;
}

button {

    width: 100%;

    margin-top: 20px;

    padding: 16px;

    border: none;

    border-radius: 9px;

    background: #2563eb;

    color: white;

    font-size: 18px;

    font-weight: bold;

    cursor: pointer;
}

button:hover {

    background: #1d4ed8;
}

button:disabled {

    background: #9ca3af;

    cursor: not-allowed;
}

#result {

    display: none;

    margin-top: 25px;

    padding: 22px;

    background: #f8fafc;

    border: 1px solid #e2e8f0;

    border-radius: 10px;

    white-space: pre-wrap;

    line-height: 1.6;

    font-size: 16px;
}

.loading {

    color: #555;
}

.error {

    color: #dc2626;

    background: #fef2f2 !important;

    border-color: #fecaca !important;
}

</style>

</head>


<body>


<div class="container">


<h1>🤖 LangGraph Dev Crew</h1>


<div class="subtitle">

AI Development Planning Agent

</div>


<label for="userInput">

Enter your request

</label>


<textarea
    id="userInput"
    placeholder="Example: Create a development plan for an AI/ML Engineer..."
></textarea>


<button
    id="startButton"
    onclick="runAgent()"
>

▶ Start

</button>


<div id="result"></div>


</div>


<script>

async function runAgent() {

    const input =
        document
        .getElementById("userInput")
        .value
        .trim();

    const button =
        document
        .getElementById("startButton");

    const result =
        document
        .getElementById("result");


    if (!input) {

        alert(
            "Please enter your request."
        );

        return;
    }


    button.disabled = true;

    button.innerText =
        "⏳ Processing...";


    result.style.display = "block";

    result.className =
        "loading";

    result.innerText =
        "AI agent is working...";


    try {

        const response =
            await fetch(
                "/devcrew/invoke",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        input: input
                    })
                }
            );


        const data =
            await response.json();


        if (data.output) {

            result.className = "";

            result.innerText =
                data.output;

        } else {

            result.className =
                "error";

            result.innerText =
                "Error: " +
                (
                    data.error ||
                    "Unknown error"
                );
        }


    } catch (error) {

        result.className =
            "error";

        result.innerText =
            "Error connecting to server: "
            + error.message;

    }


    button.disabled = false;

    button.innerText =
        "▶ Start";

}

</script>


</body>

</html>
"""


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )

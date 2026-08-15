import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# =========================================================
# APP
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
# GEMINI
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")


llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=GEMINI_API_KEY,
)


# =========================================================
# LANGGRAPH STATE
# =========================================================

class AgentState(TypedDict):
    input: str
    output: str


# =========================================================
# AGENT NODE
# =========================================================

def developer_agent(state: AgentState):

    user_input = state["input"]

    prompt = f"""
You are LangGraph Dev Crew, an AI Development Planning Agent.

Analyze the user's request carefully.

Provide a useful and practical response.

If the user asks for a development plan, include:

1. Project overview
2. Required skills
3. Technologies and tools
4. Development steps
5. Suggested project structure
6. Three portfolio projects
7. Recommended learning roadmap

If the user asks a technical question, explain it clearly
with examples where useful.

User request:

{user_input}
"""

    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    return {
        "output": response.content
    }


# =========================================================
# BUILD LANGGRAPH
# =========================================================

graph = StateGraph(AgentState)

graph.add_node("developer_agent", developer_agent)

graph.add_edge(START, "developer_agent")
graph.add_edge("developer_agent", END)

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
# DIRECT API
# =========================================================

@app.post("/devcrew/invoke")
async def invoke_agent(data: dict):

    user_input = data.get("input", "").strip()

    if not user_input:
        return {
            "error": "Please provide an input"
        }

    try:

        result = dev_crew.invoke({
            "input": user_input,
            "output": ""
        })

        return {
            "output": result["output"]
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# =========================================================
# PLAYGROUND UI
# =========================================================

@app.get("/devcrew/playground/", response_class=HTMLResponse)
async def playground():

    return """
<!DOCTYPE html>

<html>

<head>

<title>LangGraph Dev Crew</title>

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<style>

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #f4f6f8;
}

.container {

    width: 700px;
    max-width: 90%;

    margin: 70px auto;

    background: white;

    padding: 35px;

    border-radius: 18px;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.12);
}

h1 {

    text-align: center;

    margin-bottom: 10px;

}

.subtitle {

    text-align: center;

    color: #666;

    margin-bottom: 35px;

}

label {

    font-size: 18px;

    font-weight: bold;

}

textarea {

    width: 100%;

    height: 150px;

    margin-top: 12px;

    padding: 15px;

    box-sizing: border-box;

    border: 1px solid #bbb;

    border-radius: 10px;

    font-size: 16px;

    resize: vertical;

}

button {

    width: 100%;

    margin-top: 20px;

    padding: 16px;

    background: #2563eb;

    color: white;

    border: none;

    border-radius: 8px;

    font-size: 18px;

    cursor: pointer;

}

button:hover {

    background: #1d4ed8;

}

button:disabled {

    background: #888;

    cursor: not-allowed;

}

#result {

    margin-top: 25px;

    padding: 20px;

    background: #f8fafc;

    border-radius: 10px;

    white-space: pre-wrap;

    line-height: 1.6;

    display: none;

}

.error {

    color: #dc2626;

}

.loading {

    color: #555;

}

</style>

</head>


<body>


<div class="container">

<h1>🤖 LangGraph Dev Crew</h1>

<div class="subtitle">

AI Development Planning Agent

</div>


<label>

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
        document.getElementById("userInput").value.trim();

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

    result.className = "loading";

    result.innerText =
        "AI agent is working...";


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

            result.className = "";

            result.innerText = data.output;

        }

        else {

            result.className = "error";

            result.innerText =
                "Error: " +
                (data.error || "Unknown error");

        }


    }

    catch (error) {

        result.className = "error";

        result.innerText =
            "Error connecting to the server: "
            + error.message;

    }


    button.disabled = false;

    button.innerText = "▶ Start";

}

</script>


</body>

</html>
"""


# =========================================================
# RENDER START
# =========================================================

if __name__ == "__main__":

    import uvicorn

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )

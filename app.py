import os
import uvicorn

from fastapi import FastAPI
from langserve import add_routes

from pydantic import BaseModel, Field
from typing import TypedDict

from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. GOOGLE API KEY
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set")


# ============================================================
# 2. GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# 3. LANGGRAPH STATE
# ============================================================

class DevCrewState(TypedDict):

    task: str
    plan: str
    code: str
    review: str
    tests: str
    final_answer: str


# ============================================================
# 4. PLANNER AGENT
# ============================================================

def planner_agent(state: DevCrewState):

    prompt = f"""
You are the Planner Agent of a software development team.

USER TASK:
{state["task"]}

Create a clear development plan for this task.

Include:
1. Problem understanding
2. Approach
3. Required technologies or libraries
4. Implementation steps

Do NOT write the complete code.
"""

    response = llm.invoke(prompt)

    return {
        "plan": response.content
    }


# ============================================================
# 5. CODER AGENT
# ============================================================

def coder_agent(state: DevCrewState):

    prompt = f"""
You are the Coder Agent of a software development team.

USER TASK:
{state["task"]}

PLANNER'S PLAN:
{state["plan"]}

Write a complete working solution.

Requirements:
- Use clean code
- Include all required imports
- Keep the code easy to understand
- Do not skip important parts
- Include comments where useful
"""

    response = llm.invoke(prompt)

    return {
        "code": response.content
    }


# ============================================================
# 6. REVIEWER AGENT
# ============================================================

def reviewer_agent(state: DevCrewState):

    prompt = f"""
You are the Code Reviewer Agent.

USER TASK:
{state["task"]}

GENERATED CODE:
{state["code"]}

Review the generated code carefully.

Check for:

1. Syntax errors
2. Logic errors
3. Missing imports
4. Incorrect functions
5. Runtime errors
6. Whether the solution actually solves the task

Give specific corrections if required.
"""

    response = llm.invoke(prompt)

    return {
        "review": response.content
    }


# ============================================================
# 7. TESTER AGENT
# ============================================================

def tester_agent(state: DevCrewState):

    prompt = f"""
You are the Tester Agent.

USER TASK:
{state["task"]}

GENERATED CODE:
{state["code"]}

CODE REVIEW:
{state["review"]}

Create test cases for the solution.

For every test case provide:

- Input
- Expected output
- Purpose of the test

Also identify any possible edge cases.
"""

    response = llm.invoke(prompt)

    return {
        "tests": response.content
    }


# ============================================================
# 8. FINAL AGENT
# ============================================================

def final_agent(state: DevCrewState):

    prompt = f"""
You are the Lead Developer.

Prepare the final answer for the user.

USER TASK:
{state["task"]}

DEVELOPMENT PLAN:
{state["plan"]}

GENERATED CODE:
{state["code"]}

CODE REVIEW:
{state["review"]}

TEST CASES:
{state["tests"]}

Create the final response using this format:

## Plan

Give the development plan.

## Code

Give the corrected complete code.

## Review

Summarize the important review points.

## Test Cases

Give useful test cases with expected outputs.

## How to Run

Explain how to run the code.

IMPORTANT:
If the reviewer found errors, fix them before showing
the final code.

Do not mention internal agents or LangGraph processing.
"""

    response = llm.invoke(prompt)

    return {
        "final_answer": response.content
    }


# ============================================================
# 9. CREATE LANGGRAPH WORKFLOW
# ============================================================

workflow = StateGraph(DevCrewState)


# Add agents

workflow.add_node(
    "planner",
    planner_agent
)

workflow.add_node(
    "coder",
    coder_agent
)

workflow.add_node(
    "reviewer",
    reviewer_agent
)

workflow.add_node(
    "tester",
    tester_agent
)

workflow.add_node(
    "final",
    final_agent
)


# ============================================================
# 10. CONNECT AGENTS
# ============================================================

workflow.add_edge(
    START,
    "planner"
)

workflow.add_edge(
    "planner",
    "coder"
)

workflow.add_edge(
    "coder",
    "reviewer"
)

workflow.add_edge(
    "reviewer",
    "tester"
)

workflow.add_edge(
    "tester",
    "final"
)

workflow.add_edge(
    "final",
    END
)


# Compile graph

dev_crew = workflow.compile()


# ============================================================
# 11. INPUT MODEL
# ============================================================

class AgentInput(BaseModel):

    input: str = Field(
        description="Describe your software development task"
    )


# ============================================================
# 12. FORMAT INPUT
# ============================================================

def format_for_agent(x):

    if isinstance(x, dict):

        user_input = x.get(
            "input",
            ""
        )

    else:

        user_input = x.input


    return {

        "task": user_input,

        "plan": "",

        "code": "",

        "review": "",

        "tests": "",

        "final_answer": ""

    }


# ============================================================
# 13. EXTRACT FINAL RESPONSE
# ============================================================

def extract_response(output):

    if isinstance(output, dict):

        return output.get(
            "final_answer",
            str(output)
        )

    return str(output)


# ============================================================
# 14. CREATE LANGSERVE CHAIN
# ============================================================

formatted_agent_chain = (

    RunnableLambda(
        format_for_agent
    )

    | dev_crew

    | RunnableLambda(
        extract_response
    )

).with_types(

    input_type=AgentInput,

    output_type=str

)


# ============================================================
# 15. FASTAPI
# ============================================================

app = FastAPI(

    title="LangGraph Agent - Dev Crew",

    description=(
        "Multi-agent software development "
        "crew using LangGraph and Gemini"
    ),

    version="1.0"

)


# ============================================================
# 16. LANGSERVE ROUTE
# ============================================================

add_routes(

    app,

    formatted_agent_chain,

    path="/devcrew"

)


# ============================================================
# 17. HOME ROUTE
# ============================================================

@app.get("/")
def home():

    return {

        "message": "LangGraph Dev Crew is running!",

        "endpoint": "/devcrew",

        "playground": "/devcrew/playground/"

    }


# ============================================================
# 18. START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(

        os.environ.get(
            "PORT",
            8000
        )

    )

    uvicorn.run(

        app,

        host="0.0.0.0",

        port=port

    )

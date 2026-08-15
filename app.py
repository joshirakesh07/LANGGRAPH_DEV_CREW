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
# 4. PLANNER
# ============================================================

def planner_agent(state):

    prompt = f"""
You are the Planning Agent.

User task:
{state["task"]}

Create a clear step-by-step plan for solving this
software development problem.

Do not write code.
"""

    response = llm.invoke(prompt)

    return {
        "plan": str(response.content)
    }


# ============================================================
# 5. CODER
# ============================================================

def coder_agent(state):

    prompt = f"""
You are the Coding Agent.

User task:
{state["task"]}

Plan:
{state["plan"]}

Write a complete working solution.

Include:
- Required imports
- Complete code
- Comments where useful
- Simple implementation
"""

    response = llm.invoke(prompt)

    return {
        "code": str(response.content)
    }


# ============================================================
# 6. REVIEWER
# ============================================================

def reviewer_agent(state):

    prompt = f"""
You are the Code Review Agent.

User task:
{state["task"]}

Code:
{state["code"]}

Check the code for:

- Syntax errors
- Logic errors
- Missing imports
- Runtime problems
- Incorrect output
- Edge cases

If there are problems, explain exactly how to fix them.
"""

    response = llm.invoke(prompt)

    return {
        "review": str(response.content)
    }


# ============================================================
# 7. TESTER
# ============================================================

def tester_agent(state):

    prompt = f"""
You are the Testing Agent.

User task:
{state["task"]}

Code:
{state["code"]}

Review:
{state["review"]}

Create useful test cases.

For each test case give:
- Input
- Expected output
- Purpose

Also include important edge cases.
"""

    response = llm.invoke(prompt)

    return {
        "tests": str(response.content)
    }


# ============================================================
# 8. FINAL AGENT
# ============================================================

def final_agent(state):

    prompt = f"""
You are the Lead Developer.

Prepare the final solution for the user.

User task:
{state["task"]}

Plan:
{state["plan"]}

Code:
{state["code"]}

Review:
{state["review"]}

Tests:
{state["tests"]}

If the reviewer found errors, fix them.

Return the final answer using:

PLAN:
<plan>

CODE:
<complete corrected code>

REVIEW:
<review summary>

TEST CASES:
<test cases>

HOW TO RUN:
<instructions>

Do not mention internal agents or LangGraph.
"""

    response = llm.invoke(prompt)

    return {
        "final_answer": str(response.content)
    }


# ============================================================
# 9. LANGGRAPH
# ============================================================

workflow = StateGraph(DevCrewState)


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


dev_crew = workflow.compile()


# ============================================================
# 10. INPUT MODEL
# ============================================================

class AgentInput(BaseModel):

    input: str = Field(
        description="Your software development task"
    )


# ============================================================
# 11. RUN DEV CREW
# ============================================================

def run_dev_crew(data):

    # LangServe sends {"input": "..."}
    if isinstance(data, dict):

        task = data.get("input", "")

    else:

        task = str(data)


    task = str(task).strip()


    if not task:

        return "Please enter a software development task."


    initial_state = {

        "task": task,

        "plan": "",

        "code": "",

        "review": "",

        "tests": "",

        "final_answer": ""

    }


    result = dev_crew.invoke(
        initial_state
    )


    return result["final_answer"]


# ============================================================
# 12. CREATE LANGSERVE RUNNABLE
# ============================================================

dev_crew_chain = RunnableLambda(
    run_dev_crew
).with_types(
    input_type=AgentInput,
    output_type=str
)


# ============================================================
# 13. FASTAPI
# ============================================================

app = FastAPI(
    title="LangGraph Agent - Dev Crew",
    version="1.0"
)


# ============================================================
# 14. LANGSERVE
# ============================================================

add_routes(
    app,
    dev_crew_chain,
    path="/devcrew"
)


# ============================================================
# 15. HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "LangGraph Dev Crew is running"
    }


# ============================================================
# 16. START
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

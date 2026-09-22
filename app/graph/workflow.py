import os
from typing import TypedDict
import json
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

from app.agents.weather_agent import WeatherAgent
from app.agents.research_agent import ResearchAgent
from app.agents.travel_agent import TravelAgent
from app.agents.reviewer_agent import ReviewerAgent


# Load environment variables
load_dotenv()


# Create LLM client
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


# Shared State
class WeatherState(TypedDict):
    user_message: str
    city: str
    result: str
    next_agent: str
    agent_results: dict
    agent_plan: list


# --------------------------------------------------
# Supervisor / Orchestrator
# --------------------------------------------------

def supervisor_node(state: WeatherState):

    user_message = state["user_message"]

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the supervisor/orchestrator of a multi-agent system.\n\n"

                    "Available agents:\n"
                    "- weather_agent: weather, temperature, rain, humidity, wind, forecast\n"
                    "- research_agent: facts, history, culture, general information\n"
                    "- travel_agent: places to visit, attractions, things to do, trip planning\n\n"

                    "Analyze the user's request and create an execution plan.\n"
                    "Use one or more agents when necessary.\n\n"

                    "Return ONLY a JSON array containing agent names.\n\n"

                    "Valid agent names:\n"
                    "weather_agent\n"
                    "research_agent\n"
                    "travel_agent\n\n"

                    "Example:\n"
                    "[\"weather_agent\", \"research_agent\", \"travel_agent\"]"
                )
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )

    raw_plan = (
        response.choices[0].message.content or ""
    ).strip()

    print(f"🧠 Supervisor raw plan: {raw_plan}")

    try:
        agent_plan = json.loads(raw_plan)
    except json.JSONDecodeError:
        agent_plan = ["research_agent"]

    valid_agents = {
        "weather_agent",
        "research_agent",
        "travel_agent"
    }

    agent_plan = [
        agent
        for agent in agent_plan
        if agent in valid_agents
    ]

    if not agent_plan:
        agent_plan = ["research_agent"]

    next_agent = agent_plan[0]

    print(f"🔥 Supervisor plan: {agent_plan}")
    print(f"➡️ Next Agent: {next_agent}")

    return {
        "agent_plan": agent_plan,
        "next_agent": next_agent
    }

# --------------------------------------------------
# Router
# --------------------------------------------------

def route_agent(state: WeatherState):
    return state["next_agent"]

def route_next_agent(state: WeatherState):

    plan = state["agent_plan"]
    completed = state["agent_results"]

    for agent in plan:

        agent_key = agent.replace("_agent", "")

        if agent_key not in completed:
            return agent

    return "reviewer"

# --------------------------------------------------
# Weather Agent
# --------------------------------------------------

def weather_node(state: WeatherState):

    weather_agent = WeatherAgent(client)

    answer = weather_agent.run(
        state["user_message"]
    )

    print("🌦️ Weather Agent called!")

    return {
        "result": answer,
        "agent_results": {
            **state["agent_results"],
            "weather": answer
        }
    }


# --------------------------------------------------
# Research Agent
# --------------------------------------------------

def research_node(state: WeatherState):

    research_agent = ResearchAgent(client)

    answer = research_agent.run(
        state["user_message"]
    )

    print("🔎 Research Agent called!")

    return {
        "result": answer,
        "agent_results": {
            **state["agent_results"],
            "research": answer
        }
    }


# --------------------------------------------------
# Travel Agent
# --------------------------------------------------

def travel_node(state: WeatherState):

    travel_agent = TravelAgent(client)

    answer = travel_agent.run(
        state["user_message"]
    )

    print("✈️ Travel Agent called!")

    return {
        "result": answer,
        "agent_results": {
            **state["agent_results"],
            "travel": answer
        }
    }


# --------------------------------------------------
# Reviewer Agent
# --------------------------------------------------
def reviewer_node(state: WeatherState):

    reviewer_agent = ReviewerAgent(client)

    agent_results = state["agent_results"]

    combined_results = "\n\n".join(
        f"--- {agent.upper()} AGENT RESULT ---\n{answer[:2500]}"
        for agent, answer in agent_results.items()
    )

    answer = reviewer_agent.run(
        state["user_message"],
        combined_results
    )

    print("📝 Reviewer Agent called!")

    return {
        "result": answer
    }


# --------------------------------------------------
# Create Graph
# --------------------------------------------------

graph_builder = StateGraph(WeatherState)


# Add Nodes
graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("weather_agent", weather_node)
graph_builder.add_node("research_agent", research_node)
graph_builder.add_node("travel_agent", travel_node)
graph_builder.add_node("reviewer", reviewer_node)


# --------------------------------------------------
# Define Flow
# --------------------------------------------------

graph_builder.add_edge(
    START,
    "supervisor"
)


graph_builder.add_conditional_edges(
    "supervisor",
    route_agent,
    {
        "weather_agent": "weather_agent",
        "research_agent": "research_agent",
        "travel_agent": "travel_agent"
    }
)


# Agents → Reviewer
graph_builder.add_conditional_edges(
    "weather_agent",
    route_next_agent,
    {
        "weather_agent": "weather_agent",
        "research_agent": "research_agent",
        "travel_agent": "travel_agent",
        "reviewer": "reviewer"
    }
)

graph_builder.add_conditional_edges(
    "research_agent",
    route_next_agent,
    {
        "weather_agent": "weather_agent",
        "research_agent": "research_agent",
        "travel_agent": "travel_agent",
        "reviewer": "reviewer"
    }
)

graph_builder.add_conditional_edges(
    "travel_agent",
    route_next_agent,
    {
        "weather_agent": "weather_agent",
        "research_agent": "research_agent",
        "travel_agent": "travel_agent",
        "reviewer": "reviewer"
    }
)


graph_builder.add_edge(
    "reviewer",
    END
)

# Compile Graph
graph = graph_builder.compile()


# --------------------------------------------------
# Run Graph
# --------------------------------------------------

if __name__ == "__main__":

    user_message = input(
        "What do you want to know? "
    )

    result = graph.invoke({
        "user_message": user_message,
        "city": "",
        "result": "",
        "next_agent": "",
        "agent_results": {},
        "agent_plan": []
    })

    print("\n🌦️ LangGraph Result:")
    print(result["result"])
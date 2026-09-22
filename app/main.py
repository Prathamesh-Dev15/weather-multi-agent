import os

from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI
from pydantic import BaseModel



from app.agents.weather_agent import WeatherAgent
from app.agents.travel_agent import TravelAgent
from app.agents.research_agent import ResearchAgent
from app.agents.reviewer_agent import ReviewerAgent
from app.agents.supervisor_agent import SupervisorAgent


load_dotenv()


client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


app = FastAPI(
    title="AI Weather & Travel Multi-Agent",
    version="1.0.0"
)


class AskRequest(BaseModel):
    question: str


# ---------------------------------
# Create Agents
# ---------------------------------

weather_agent = WeatherAgent(client)

travel_agent = TravelAgent(client)

research_agent = ResearchAgent(client)

reviewer_agent = ReviewerAgent(client)


# ---------------------------------
# Create Supervisor
# ---------------------------------

supervisor = SupervisorAgent(
    client,
    weather_agent,
    travel_agent,
    research_agent,
    reviewer_agent
)


# ---------------------------------
# Root API
# ---------------------------------

@app.get("/")
def root():

    return {
        "message": "AI Weather & Travel Multi-Agent API is running"
    }


# ---------------------------------
# Main Multi-Agent API
# ---------------------------------

@app.post("/ask")
def ask(request: AskRequest):

    answer = supervisor.run(request.question)

    return {
        "question": request.question,
        "answer": answer
    }
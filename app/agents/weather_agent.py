import json
import requests
from openai import OpenAI

from app.tools.weather_tool import get_weather

class WeatherAgent:

    def __init__(self, client: OpenAI):
        self.client = client

    def run(self, user_message: str) -> str:

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather information for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The name of the city."
                            }
                        },
                        "required": ["city"]
                    }
                }
            }
        ]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful weather assistant. "
                    "Always use the get_weather tool to get "
                    "current weather information."
                )
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

        # Step 1: Ask LLM to call weather tool
        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {
                    "name": "get_weather"
                }
            }
        )

        assistant_message = response.choices[0].message

        print("🔧 Tool calls:", assistant_message.tool_calls)

        # Step 2: Check tool call
        if assistant_message.tool_calls:

            messages.append(assistant_message)

            for tool_call in assistant_message.tool_calls:

                if tool_call.function.name == "get_weather":

                    arguments = json.loads(
                        tool_call.function.arguments
                    )

                    city = arguments["city"]

                    print(f"🌍 Calling weather tool for: {city}")

                    # Step 3: Python executes tool
                    weather_data = get_weather(city)

                    print("📡 Weather data:", weather_data)

                    # Step 4: Send tool result to LLM
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(weather_data)
                        }
                    )

            # Step 5: Generate final answer
            final_response = self.client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages
            )

            return final_response.choices[0].message.content

        return assistant_message.content
from openai import OpenAI


class TravelAgent:

    def __init__(self, client: OpenAI):
        self.client = client

    def run(self, user_message: str) -> str:

        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful travel assistant. "
                        "Help users with travel recommendations, "
                        "places to visit, things to do, travel tips, "
                        "and simple travel planning."
                    )
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        )

        return response.choices[0].message.content
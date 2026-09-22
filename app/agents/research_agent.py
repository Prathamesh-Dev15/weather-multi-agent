from openai import OpenAI


class ResearchAgent:

    def __init__(self, client: OpenAI):
        self.client = client

    def run(self, user_message: str) -> str:

        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant. "
                        "Provide useful factual information "
                        "based on the user's request."
                    )
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        )

        return response.choices[0].message.content
from openai import OpenAI


class ReviewerAgent:

    def __init__(self, client: OpenAI):
        self.client = client

    def run(self, user_message: str, agent_results: str) -> str:

        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the final reviewer in a multi-agent system.\n\n"

                        "Your job is to combine the results from the "
                        "specialized agents into one useful final answer.\n\n"

                        "STRICT RULES:\n"
                        "1. Use only information provided by the agents.\n"
                        "2. Do NOT invent facts, numbers, weather values, "
                        "locations, dates, or recommendations.\n"
                        "3. Do NOT add information from your own knowledge.\n"
                        "4. If agents provide conflicting information, "
                        "prefer the most specific or tool-generated result.\n"
                        "5. Keep the final answer concise and well organized.\n"
                        "6. Preserve exact weather values returned by the "
                        "Weather Agent.\n"
                        "7. Do not claim live/current information unless "
                        "the agent result explicitly provides it.\n"
                    )
                },
                {
                    "role": "user",
                    "content": f"""
Original user request:
{user_message}

Results from specialized agents:

{agent_results}

Create the final answer using ONLY the information above.
"""
                }
            ]
        )

        return response.choices[0].message.content
class SupervisorAgent:

    def __init__(
        self,
        client,
        weather_agent,
        travel_agent,
        research_agent,
        reviewer_agent
    ):
        self.client = client
        self.weather_agent = weather_agent
        self.travel_agent = travel_agent
        self.research_agent = research_agent
        self.reviewer_agent = reviewer_agent

    def run(self, question: str):

        print("\n🧠 Supervisor Agent")
        print(f"Question: {question}")

        responses = []

        question_lower = question.lower()

        # -----------------------------
        # Weather Agent
        # -----------------------------
        if any(word in question_lower for word in [
            "weather",
            "temperature",
            "rain",
            "rainy",
            "forecast",
            "humidity"
        ]):
            print("🌦️ Calling Weather Agent...")

            weather_response = self.weather_agent.run(question)

            responses.append(
                f"Weather Agent:\n{weather_response}"
            )

        # -----------------------------
        # Travel Agent
        # -----------------------------
        if any(word in question_lower for word in [
            "travel",
            "trip",
            "tour",
            "visit",
            "places",
            "tourist",
            "carry",
            "packing"
        ]):
            print("✈️ Calling Travel Agent...")

            travel_response = self.travel_agent.run(question)

            responses.append(
                f"Travel Agent:\n{travel_response}"
            )

        # -----------------------------
        # Research Agent
        # -----------------------------
        if any(word in question_lower for word in [
            "research",
            "information",
            "details",
            "tell me about",
            "information about"
        ]):
            print("🔎 Calling Research Agent...")

            research_response = self.research_agent.run(question)

            responses.append(
                f"Research Agent:\n{research_response}"
            )

        # -----------------------------
        # No Agent Selected
        # -----------------------------
        if not responses:

            print("🤖 No specific agent selected.")

            research_response = self.research_agent.run(question)

            responses.append(
                f"Research Agent:\n{research_response}"
            )

        # -----------------------------
        # Combine Agent Responses
        # -----------------------------
        combined_response = "\n\n".join(responses)

        print("\n📋 Combined Agent Responses:")
        print(combined_response)

        # -----------------------------
        # Reviewer Agent
        # -----------------------------
        print("\n📝 Calling Reviewer Agent...")

        final_response = self.reviewer_agent.run(
            question,
            combined_response
        )

        return final_response
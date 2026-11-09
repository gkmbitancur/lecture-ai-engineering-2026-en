"""Advanced option (a): same instructions and read-only tools, ADK harness."""

from google.adk.agents import Agent
from google.genai import types

from app import tools
from app.agent import INSTRUCTIONS
from app.config import MODEL

root_agent = Agent(
    model=MODEL,
    name="glossary_agent",
    instruction=INSTRUCTIONS,
    generate_content_config=types.GenerateContentConfig(temperature=0),
    tools=[tools.search_glossary, tools.get_lecture_summary, tools.make_quiz],
)

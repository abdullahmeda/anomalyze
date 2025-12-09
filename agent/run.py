"""
Anomaly Analysis Agent Runner

Analyzes detected anomalies and generates structured incident reports.

Usage:
    python3 -m agent.run --date 2023-10-05
"""

import argparse
import asyncio
import logging
import os

from dotenv import load_dotenv

# Google ADK Imports
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Tracing Imports (Phoenix/OpenInference)
from phoenix.otel import register
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from agent.prompts import SYSTEM_PROMPT, get_user_prompt
from agent.tools import fetch_ticket_stats, fetch_ticket_samples

load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)

# Setup Tracing (Logs to a local Phoenix server)
tracer_provider = register(project_name="anomalyze-agent")
GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


openrouter_model = LiteLlm(model="openrouter/anthropic/claude-sonnet-4.5")

anomaly_agent = LlmAgent(
    name="anomaly_analyst",
    model=openrouter_model,
    instruction=SYSTEM_PROMPT,
    tools=[fetch_ticket_stats, fetch_ticket_samples],
)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


async def analyze_anomaly(date: str) -> str:
    """Run the anomaly analysis agent for a specific date."""
    
    # Session service handles conversation memory
    session_service = InMemorySessionService()
    
    app_name = "anomalyze"
    user_id = "analyst"
    session_id = f"analysis_{date}"
    
    # Initialize the session
    await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    
    runner = Runner(
        agent=anomaly_agent,
        app_name=app_name,
        session_service=session_service,
    )
    
    prompt = get_user_prompt(date)
    
    # Run the agent and collect final response
    final_response = None
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=prompt)])
    ):
        if event.is_final_response():
            final_response = event.content.parts[0].text
    
    return final_response


def main():
    parser = argparse.ArgumentParser(description="Analyze anomalies and generate incident reports")
    parser.add_argument("--date", "-d", required=True, help="Date to analyze (YYYY-MM-DD)")
    parser.add_argument("--output", "-o", help="Output file path (optional, prints to stdout if not specified)")
    args = parser.parse_args()

    if "OPENROUTER_API_KEY" not in os.environ:
        logger.error("OPENROUTER_API_KEY not set. Create a .env file or export the variable.")
        return

    logger.info(f"Analyzing anomaly on {args.date}...")
    logger.info("=" * 60)

    report = asyncio.run(analyze_anomaly(args.date))

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        logger.info(f"Report saved to {args.output}")
    else:
        logger.info("\nIncident Report:")
        logger.info("-" * 60)
        print(report)


if __name__ == "__main__":
    main()

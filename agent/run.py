"""
Anomaly Analysis Agent Runner

Analyzes detected anomalies and generates structured incident reports.

Usage:
    python3 -m agent.run --date 2023-10-05
"""

import argparse
import asyncio
import json
import os

from dotenv import load_dotenv
from agents import Agent, Runner

from agent.prompts import SYSTEM_PROMPT, IncidentReport
from agent.tools import fetch_ticket_stats, fetch_ticket_samples

load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


anomaly_agent = Agent(
    name="Anomaly Analyst",
    instructions=SYSTEM_PROMPT,
    tools=[fetch_ticket_stats, fetch_ticket_samples],
    model="gpt-4o",
    output_type=IncidentReport,
)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


async def analyze_anomaly(date: str) -> IncidentReport:
    """Run the anomaly analysis agent for a specific date."""
    prompt = f"""An automated monitor has detected a high volume of support tickets on {date}.

Your task is to conduct a full analysis and generate a final incident report.

Use the available tools to:
1. First, fetch the ticket statistics to understand the volume and distribution changes
2. Then, fetch ticket samples to understand what customers are actually reporting
3. Finally, synthesize your findings into a comprehensive incident report
"""

    result = await Runner.run(anomaly_agent, prompt)
    return result.final_output


def main():
    parser = argparse.ArgumentParser(description="Analyze anomalies and generate incident reports")
    parser.add_argument("--date", "-d", required=True, help="Date to analyze (YYYY-MM-DD)")
    parser.add_argument("--output", "-o", help="Output file path (optional, prints to stdout if not specified)")
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY not set. Create a .env file or export the variable.")
        return

    print(f"Analyzing anomaly on {args.date}...")
    print("=" * 60)

    report = asyncio.run(analyze_anomaly(args.date))

    # Output as formatted JSON
    report_json = report.model_dump_json(indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report_json)
        print(f"Report saved to {args.output}")
    else:
        print("\nIncident Report:")
        print("-" * 60)
        print(report_json)


if __name__ == "__main__":
    main()


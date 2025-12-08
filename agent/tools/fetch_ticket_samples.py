"""Tool to fetch raw ticket samples for a given date."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

DATASET_PATH = Path(__file__).parent.parent.parent / "dataset" / "data" / "full_dataset.csv"


def _load_dataset() -> pd.DataFrame:
    """Load the full dataset with parsed timestamps."""
    return pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])


def fetch_ticket_samples(date: str, limit: int = 20) -> str:
    """
    Retrieve random ticket samples from a specific date.

    Use this tool to understand the actual content of customer complaints
    and identify patterns in what customers are reporting.

    Args:
        date: The date to sample from in YYYY-MM-DD format (e.g., "2023-10-05")
        limit: Maximum number of tickets to return (default: 20)

    Returns:
        JSON string containing a list of tickets, each with:
        - ticket_id: Unique identifier
        - subject: Ticket subject line (may be null)
        - body: Full ticket body text
        - priority: Ticket priority level
        - type: Ticket type (Incident, Request, Problem, Change)
        - queue: Department queue
        - language: Language code (en/de)
    """
    df = _load_dataset()

    target_date = datetime.strptime(date, "%Y-%m-%d").date()
    df["date"] = df["timestamp"].dt.date

    # Filter to target date
    day_tickets = df[df["date"] == target_date]

    # Sample tickets
    sample_size = min(limit, len(day_tickets))
    samples = day_tickets.sample(n=sample_size, random_state=42)

    # Build response
    tickets = []
    for _, row in samples.iterrows():
        ticket = {
            "ticket_id": int(row["ticket_id"]),
            "subject": row["subject"] if pd.notna(row["subject"]) else None,
            "body": row["body"] if pd.notna(row["body"]) else None,
            "priority": row["priority"],
            "type": row["type"] if pd.notna(row["type"]) else None,
            "queue": row["queue"],
            "language": row["language"],
        }
        tickets.append(ticket)

    result = {
        "date": date,
        "total_tickets_on_date": len(day_tickets),
        "samples_returned": len(tickets),
        "tickets": tickets,
    }

    return json.dumps(result, indent=2, ensure_ascii=False)


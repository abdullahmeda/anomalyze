import asyncio
import os
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, OpenAIConversationsSession, trace

load_dotenv()

# --- 1. Tools (Same as before) ---
@function_tool
def get_stock_price(ticker: str) -> str:
    """Gets the current stock price for a given ticker symbol."""
    return f"{ticker} is currently trading at $150.00"

@function_tool
def calculate_position_value(shares: int, price_per_share: float) -> float:
    """Calculates total value of a stock position."""
    return shares * price_per_share

# --- 2. Agent (Same as before) ---
finance_agent = Agent(
    name="Portfolio Assistant",
    instructions="You are a helpful financial assistant.",
    tools=[get_stock_price, calculate_position_value],
    model="gpt-4.1"
)

async def main():
    # --- 3. Initialize OpenAI Session ---
    # Instead of a local file, this creates a conversation on OpenAI's servers.
    # To resume a previous conversation, you would pass:
    # session = OpenAIConversationsSession(conversation_id="conv_123")
    session = OpenAIConversationsSession()

    print("--- Turn 1 (New Query) ---")
    
    with trace(workflow_name="Finance Query Workflow"):
        result_1 = await Runner.run(
            finance_agent,
            "What is the price of AAPL? If I have 10 shares, how much is that worth?",
            session=session
        )
        print(f"Agent: {result_1.final_output}")
        
        # Important: The session automatically gets a session_id assigned by OpenAI.
        # You should save this ID if you want to resume this chat in a different script run.
        session_id = await session._get_session_id()
        print(f"\n[System] Session ID: {session_id}\n")
        print(f"[System] To resume this conversation later, use: OpenAIConversationsSession(session_id='{session_id}')\n")

    print("--- Turn 2 (Testing Memory) ---")
    
    result_2 = await Runner.run(
        finance_agent,
        "What if I sell half of them?", 
        session=session
    )
    print(f"Agent: {result_2.final_output}")

if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = input("Enter OpenAI API Key: ")
    asyncio.run(main())
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

WsearchAgent = Agent(
    name="WsearchAgent",
    role="Search the web for information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include the source of the information you find."],
    show_tool_calls=True,
    markdown=True,
)


FinAgent = Agent(
    name="FinAgent",
    role="Financial Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            key_financial_ratios=True,
            company_news=True,
        )
    ],
    instructions=["Use tables to display the data."],
    show_tool_calls=True,
    markdown=True,
)

MultiAgent = Agent(
    team=[WsearchAgent, FinAgent],
    model=Groq(id="llama-3.1-70b-versatile"),
    instructions=["Always include source", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

MultiAgent.print_response("Summarize analyst recommendations and share the latest news.", stream=True)
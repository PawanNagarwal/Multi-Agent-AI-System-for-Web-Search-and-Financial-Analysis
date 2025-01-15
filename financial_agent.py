from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv 
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


# client = openai.Client()

# web search agent
web_search_agent = Agent(
    name = "web search agent",
    role = "seach the web for the information",
    model = Groq(id = "llama-3.3-70b-versatile"),
    tools = [DuckDuckGo()],
    instructions = ["Always include the sources"],
    show_tools_calls = True, 
    markdown = True,
)


# financial agent
financial_agent = Agent(
    name = "financial agent",
    role = "search the web for financial information",
    model = Groq(id = "llama-3.3-70b-versatile"),
    tools = [YFinanceTools(stock_price = True, analyst_recommendations = True, stock_fundamentals = True, company_news = True)],
    instructions = ["use tables to display the data"],
    show_tools_calls = True, 
    markdown = True,
)

multi_ai_agent = Agent(
    team = [web_search_agent, financial_agent],
    instructions = ["always include the sources","use tables to display the data"],
    show_tools_calls = True,
    markdown = True
)

multi_ai_agent.print_response("summarize analyst recommendation and share the latest news for honda cars", stream = True)
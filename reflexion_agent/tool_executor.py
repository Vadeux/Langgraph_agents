from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode
from schemas import AnswerQuestion, ReviseAnswer

search_tool = TavilySearch(max_results=5)


def run_queries(_: str, reflection: dict, **kwargs) -> list[str]:
    """Run the generated search_queries."""
    return search_tool.batch(
        [{"query": query} for query in reflection["search_queries"]]
    )


execute_tools = ToolNode(  # examine last elem of messages list to search for tool_call
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)  # here we want to use the same function in multiple tools

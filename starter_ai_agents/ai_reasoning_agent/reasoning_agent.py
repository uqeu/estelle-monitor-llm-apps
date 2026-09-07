import os

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.reasoning import ReasoningTools
from rich.console import Console

regular_model = os.environ.get("REGULAR_MODEL", "gpt-4o-mini")
reasoning_model = os.environ.get("REASONING_MODEL", "gpt-4o")
regular_agent = Agent(model=OpenAIChat(id=regular_model, timeout=90, max_retries=0), markdown=True)
console = Console()
reasoning_agent = Agent(
    model=OpenAIChat(id=reasoning_model, timeout=90, max_retries=0),
    tools=[ReasoningTools(add_instructions=True)],
    markdown=True,
    structured_outputs=True,
)

task = "How many 'r' are in the word 'supercalifragilisticexpialidocious'?"

if __name__ == "__main__":
    console.rule("[bold green]Regular Agent[/bold green]")
    regular_agent.print_response(task, stream=True)
    console.rule("[bold yellow]Reasoning Agent[/bold yellow]")
    reasoning_agent.print_response(task, stream=True, show_full_reasoning=True)

"""
CrewAI integration example.

This example shows how to use LitForge tools with CrewAI agents
for automated literature research.

Requires: pip install crewai crewai-tools
"""

try:
    from crewai import Agent, Task, Crew
except ImportError:
    print("CrewAI not installed. Install with: pip install crewai crewai-tools")
    exit(1)

from litforge.integrations.crewai import LitForgeTools


def main():
    # Initialize LitForge tools for CrewAI
    tools = LitForgeTools()
    
    # Create a literature researcher agent
    researcher = Agent(
        role="Literature Researcher",
        goal="Find and synthesize relevant scientific papers on a given topic",
        backstory="""You are an expert research scientist with years of experience
        in literature review. You excel at finding relevant papers, understanding
        their key contributions, and synthesizing information across multiple sources.""",
        tools=tools.all(),  # search, lookup, retrieve, citations, references, ask
        verbose=True,
    )
    
    # Create a research task
    task = Task(
        description="""Research the current state of mRNA vaccine technology.
        
        1. Search for recent papers on mRNA vaccines
        2. Identify the most influential papers in the field
        3. Summarize the key advances and remaining challenges
        4. List the top 5 papers that anyone studying this topic should read
        """,
        expected_output="""A comprehensive summary including:
        - Overview of mRNA vaccine technology
        - Key recent advances (with citations)
        - Current challenges and limitations
        - Top 5 recommended papers with brief descriptions
        """,
        agent=researcher,
    )
    
    # Create and run the crew
    crew = Crew(
        agents=[researcher],
        tasks=[task],
        verbose=True,
    )
    
    print("🚀 Starting literature research...\n")
    result = crew.kickoff()
    
    print("\n" + "=" * 60)
    print("RESEARCH RESULTS")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()

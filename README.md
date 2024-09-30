# AI Agent Documents

## AI Agent Research Paper
[GitHub Pages](https://pages.github.com/).

## AI Agent Framework

## LangChain

### Description
LangChain is a framework designed to simplify the development of applications that leverage large language models (LLMs). It provides tools and abstractions to manage the complexities of integrating LLMs into various applications.

### Applications
- Chatbots and conversational agents
- Text summarization and generation
- Language translation
- Sentiment analysis

### Strengths
- Simplifies the integration of LLMs
- Provides a modular architecture
- Supports multiple LLMs and APIs
- Extensive documentation and community support

### Limitations
- May require significant computational resources
- Can be complex for beginners
- Limited to the capabilities of the underlying LLMs

### Best Use Case For
Developers looking to build sophisticated NLP applications with minimal overhead.

```python
from langchain_anthropic import ChatAnthropic
#other imports

# Create the agent
memory = MemorySaver()
model = ChatAnthropic(model_name="claude-3-sonnet-20240229")
search = TavilySearchResults(max_results=2)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob! and i live in sf")]}, config
):
    print(chunk)
    print("----")

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather where I live?")]}, config
):
    print(chunk)
    print("----")
```
---

## CrewAI

### Description
CrewAI is a collaborative AI framework designed to enhance team productivity by integrating AI-driven insights and automation into workflows. It focuses on improving communication, task management, and decision-making within teams.

### Applications
- Project management
- Team collaboration tools
- Automated reporting and analytics
- Workflow automation

### Strengths
- Enhances team productivity
- Integrates seamlessly with existing tools
- Provides real-time insights and analytics
- Customizable to fit various team needs

### Limitations
- May require training for effective use
- Dependent on data quality and integration
- Potential privacy concerns with sensitive data

### Best Use Case For
Teams looking to streamline their workflows and enhance collaboration through AI-driven insights.
```python
from crewai import Agent, Task, Crew
from custom_agent import CustomAgent 

from langchain.agents import load_tools

langchain_tools = load_tools(["google-serper"], llm=llm)

agent1 = CustomAgent(
    role="agent role",
    goal="who is {input}?",
    backstory="agent backstory",
    verbose=True,
)

task1 = Task(
    expected_output="a short biography of {input}",
    description="a short biography of {input}",
    agent=agent1,
)

agent2 = Agent(
    role="agent role",
    goal="summarize the short bio for {input} and if needed do more research",
    backstory="agent backstory",
    verbose=True,
)

task2 = Task(
    description="a tldr summary of the short biography",
    expected_output="5 bullet point summary of the biography",
    agent=agent2,
    context=[task1],
)

my_crew = Crew(agents=[agent1, agent2], tasks=[task1, task2])
crew = my_crew.kickoff(inputs={"input": "Mark Twain"})
```
---

## Microsoft Autogen

### Description
Microsoft Autogen is a framework for automating the generation of content using AI. It leverages advanced machine learning models to create high-quality text, images, and other media.

### Applications
- Content creation for marketing
- Automated report generation
- Creative writing assistance
- Image and video generation

### Strengths
- High-quality content generation
- Supports multiple media types
- Integrates with Microsoft’s ecosystem
- Scalable and efficient

### Limitations
- May produce generic or repetitive content
- Requires fine-tuning for specific use cases
- Dependent on the quality of training data

### Best Use Case For
Businesses looking to automate content creation and enhance their marketing efforts.

```python
import os
from autogen import AssistantAgent, UserProxyAgent

llm_config = {"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}
assistant = AssistantAgent("assistant", llm_config=llm_config)
user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)

# Start the chat
user_proxy.initiate_chat(
    assistant,
    message="Tell me a joke about NVDA and TESLA stock prices.",
)
```
---

## Microsoft Semantic Kernel

### Description
Microsoft Semantic Kernel is a framework designed to enhance the semantic understanding of text. It uses advanced NLP techniques to extract meaningful insights from large volumes of text data.

### Applications
- Text analysis and summarization
- Knowledge management
- Semantic search
- Sentiment analysis

### Strengths
- Advanced semantic understanding
- Integrates with Microsoft’s data tools
- Scalable for large datasets
- Provides actionable insights

### Limitations
- Requires significant computational resources
- May need customization for specific domains
- Dependent on the quality of input data

### Best Use Case For
Organizations needing to extract and analyze insights from large text datasets.

```python
import asyncio
import logging
from semantic_kernel import Kernel
#other imports


async def main():
    # Initialize the kernel
    kernel = Kernel()

    # Add Azure OpenAI chat completion
    chat_completion = AzureChatCompletion(
        deployment_name="your_models_deployment_name",
        api_key="your_api_key",
        base_url="your_base_url",
    )
    kernel.add_service(chat_completion)
    # Enable planning
    execution_settings = AzureChatPromptExecutionSettings()
    execution_settings.function_call_behavior = FunctionCallBehavior.EnableFunctions(auto_invoke=True, filters={})

    # Create a history of the conversation
    history = ChatHistory()
    history.add_user_message("Can you help me write an email for my boss?")

    result = await chat_completion.get_chat_message_content(
        chat_history=history,
        settings=execution_settings,
        kernel=kernel
    )
    print(result)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
```

---

## Agentverse

### Description
Agentverse is a framework for developing intelligent agents that can interact with users and perform tasks autonomously. It focuses on creating agents that can understand and respond to natural language inputs.

### Applications
- Virtual assistants
- Customer support bots
- Autonomous task management
- Interactive learning systems

### Strengths
- Supports natural language understanding
- Can automate a wide range of tasks
- Customizable and extensible
- Integrates with various platforms

### Limitations
- May require extensive training data
- Can be complex to develop and maintain
- Dependent on the accuracy of NLP models

### Best Use Case For
Developers looking to create intelligent agents for customer support or task automation.

```python
from uagents import Agent, Context
 
# Create an agent named Alice
alice = Agent(name="alice", seed="YOUR NEW PHRASE")
 
# Define a periodic task for Alice
@alice.on_interval(period=2.0)
async def say_hello(ctx: Context):
    ctx.logger.info(f'hello, my name is {alice.name}')
 
# Run the agent
if __name__ == "__main__":
    alice.run()

```
    

## AI Agent Tutorials

## AI Agent 

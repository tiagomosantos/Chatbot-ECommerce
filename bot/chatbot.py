# Import necessary classes and modules
from chains.product_info import ProductInfoReasoningChain, ProductInfoResponseChain
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from bot.memory import MemoryManager


class CustomerServiceBot:
    """A bot that handles customer service interactions by processing user inputs and
    routing them through configured reasoning and response chains.
    """

    def __init__(self, user_id: str, conversation_id: str):
        """Initialize the bot with session and language model configurations.

        Args:
            user_id: Identifier for the user.
            conversation_id: Identifier for the conversation.
        """
        # Initialize session manager for handling session history and configuration
        self.memory = MemoryManager(user_id, conversation_id)

        # Set up the language model with specified parameters
        self.llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")

        # Define chain map with reasoning and response chains for different intents
        self.chain_map = {
            "product_information": {
                "reasoning": ProductInfoReasoningChain(llm=self.llm),
                "response": self.add_memory_to_chain(
                    ProductInfoResponseChain(llm=self.llm)
                ),
            }
        }

    def add_memory_to_chain(self, original_chain):
        """Wrap a chain with session history functionality.

        Args:
            original_chain: The chain instance to which session history will be added.

        Returns:
            An instance of RunnableWithMessageHistory that incorporates session history.
        """
        return RunnableWithMessageHistory(
            original_chain,
            self.memory.get_session_history,
            input_messages_key="customer_input",
            history_messages_key="chat_history",
            history_factory_config=self.memory.get_history_factory_config(),
        ).with_config(
            {"run_name": original_chain.__class__.__name__}
        )  # Add a run name to the chain on LangSmith

    def get_chain(self, intent: str):
        """Retrieve the reasoning and response chains based on user intent.

        Args:
            intent: The identified intent of the user input.

        Returns:
            A tuple containing the reasoning and response chain instances for the intent.
        """

        # Use a get method to retrieve the chains based on the intent
        return self.chain_map[intent]["reasoning"], self.chain_map[intent]["response"]

    def process_user_input(self, user_input: str):
        """Process user input by routing through the appropriate reasoning and response chains.

        Args:
            user_input: The input text from the user.

        Returns:
            The content of the response after processing through the chains.
        """
        # Step 1: Classify intent (using placeholder intent classification)
        intent = "product_information"  # Static placeholder for intent classification
        memory_config = self.memory.get_memory_config()
        memory_config["configurable"]["intention"] = intent

        # Step 2: Retrieve reasoning and response chains based on intent
        reasoning_chain, response_chain = self.get_chain(intent)

        # Step 3: Process the input with the ReasoningChain
        reasoning_output = reasoning_chain.invoke(user_input)

        # Step 4: Use the output from ReasoningChain to get the response from ResponseChain
        response = response_chain.invoke(reasoning_output, config=memory_config)

        return response.content

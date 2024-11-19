# Import necessary classes and modules
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from cobuy.chatbot.memory import MemoryManager
from cobuy.chatbot.router.loader import load_intention_classifier
from cobuy.chatbot.chains.product_info import (
    ProductInfoReasoningChain,
    ProductInfoResponseChain,
)


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

        self.intention_classifier = load_intention_classifier()

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

    def handle_product_information(self, user_input: Dict):
        """Handle the product information intent by processing user input and providing a response.

        Args:
            user_input: The input text from the user.

        Returns:
            The content of the response after processing through the chains.
        """
        # Step 1: Retrieve reasoning and response chains based on intent
        reasoning_chain, response_chain = self.get_chain("product_information")

        # Step 2: Process the input with the ReasoningChain
        reasoning_output = reasoning_chain.invoke(user_input)

        # Step 3: Use the output from ReasoningChain to get the response from ResponseChain
        response = response_chain.invoke(
            reasoning_output, config=self.memory.get_memory_config()
        )

        return response.content

    def get_user_intent(self, user_input: Dict):
        """Classify the user intent based on the input text.

        Args:
            user_input: The input text from the user.

        Returns:
            The classified intent of the user input.
        """
        intent_routes = self.intention_classifier.retrieve_multiple_routes(
            user_input["customer_input"]
        )

        if len(intent_routes) == 0:
            return None
        else:
            intention = intent_routes[0].name

        # Check if the intention is None
        if intention is None:
            return None
        else:
            # Check if the intention is a string
            if isinstance(intention, str):
                return intention
            else:
                intention_type = type(intention).__name__
                print(
                    f"I'm sorry, I didn't understand that. The intention type is {intention_type}."
                )
                return None

    def process_user_input(self, user_input: Dict):
        """Process user input by routing through the appropriate reasoning and response chains.

        Args:
            user_input: The input text from the user.

        Returns:
            The content of the response after processing through the chains.
        """
        # Step 1: Classify the intent using the intention classifier
        intention = self.get_user_intent(user_input)

        # Step 2: Process the input based on the identified intention
        if intention == "product_information":
            response = self.handle_product_information(user_input)
            return response
        else:
            return "I'm sorry, I didn't understand that."

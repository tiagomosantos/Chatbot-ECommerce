# Import necessary classes and modules for chatbot functionality
from typing import Any, Callable, Dict, Optional, Tuple

from langchain.schema.runnable.base import Runnable
from langchain_core.messages.ai import AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from cobuy.chatbot.agents.order_agent import OrderAgent
from cobuy.chatbot.chains.chitchat import ChitChatClassifierChain, ChitChatResponseChain
from cobuy.chatbot.chains.product_info import (
    ProductInfoReasoningChain,
    ProductInfoResponseChain,
)
from cobuy.chatbot.chains.router import RouterChain
from cobuy.chatbot.memory import MemoryManager
from cobuy.chatbot.rag.rag import RAGPipeline
from cobuy.chatbot.router.loader import load_intention_classifier


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
        # Initialize the memory manager to manage session history
        self.memory = MemoryManager()
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.memory_config = {
            "configurable": {
                "user_id": self.user_id,
                "conversation_id": self.conversation_id,
            }
        }
        # Configure the language model with specific parameters for response generation
        self.llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")

        # Map intent names to their corresponding reasoning and response chains
        self.chain_map = {
            "product_information": {
                "reasoning": ProductInfoReasoningChain(llm=self.llm),  # Reasoning chain
                "response": self.add_memory_to_runnable(
                    ProductInfoResponseChain(llm=self.llm)  # Response chain with memory
                ),
            },
            "chitchat": {
                "reasoning": ChitChatClassifierChain(llm=self.llm),
                "response": self.add_memory_to_runnable(
                    ChitChatResponseChain(llm=self.llm)
                ),
            },
            "router": {
                "reasoning": RouterChain(llm=self.llm),
            },
        }

        self.agent_map = {
            "order": self.add_memory_to_runnable(
                OrderAgent(llm=self.llm).agent_executor
            )
        }

        self.rag = self.add_memory_to_runnable(
            RAGPipeline(
                index_name="rag",
                embeddings_model="text-embedding-3-small",
                llm=self.llm,
                memory=True,
            ).rag_chain
        )

        # Load the intention classifier to determine user intents
        self.intention_classifier = load_intention_classifier()

        # Map of intentions to their corresponding handlers
        self.intent_handlers: Dict[Optional[str], Callable[[Dict[str, str]], str]] = {
            "product_information": self.handle_product_information,
            "create_order": self.handle_order_intent,
            "order_status": self.handle_order_intent,
            "support_information": self.handle_support_information,
        }

    def add_memory_to_runnable(
        self, original_runnable: Runnable[Any, Any]
    ) -> RunnableWithMessageHistory:
        """Wrap a runnable with session history functionality.

        Args:
            original_runnable: The runnable instance to which session history will be added.

        Returns:
            An instance of RunnableWithMessageHistory that incorporates session history.
        """
        return RunnableWithMessageHistory(
            runnable=original_runnable,
            get_session_history=self.memory.get_session_history,  # Retrieve session history
            input_messages_key="customer_input",  # Key for user inputs
            history_messages_key="chat_history",  # Key for chat history
            history_factory_config=self.memory.get_history_factory_config(),  # Config for history factory
        ).with_config(
            {
                "run_name": original_runnable.__class__.__name__
            }  # Add runnable name for tracking
        )

    def get_chain(self, intent: str) -> Tuple[Optional[Runnable], Optional[Runnable]]:
        """Retrieve the reasoning and response chains based on user intent.

        Args:
            intent: The identified intent of the user input.

        Returns:
            A tuple containing the reasoning and response chain instances for the intent.
        """
        reasoning_chain: Optional[Runnable] = self.chain_map[intent].get(
            "reasoning", None
        )
        response_chain: Optional[Runnable] = self.chain_map[intent].get(
            "response", None
        )

        return reasoning_chain, response_chain

    def get_agent(self, intent: str):
        """Retrieve the agent based on user intent.

        Args:
            intent: The identified intent of the user input.

        Returns:
            The agent instance for the intent.
        """
        return self.agent_map[intent]

    def get_user_intent(self, user_input: Dict[str, str]):
        """Classify the user intent based on the input text.

        Args:
            user_input: The input text from the user.

        Returns:
            The classified intent of the user input.
        """
        # Retrieve possible routes for the user's input using the classifier
        intent_routes = self.intention_classifier.retrieve_multiple_routes(
            user_input["customer_input"]
        )

        print("Intent Routes:", intent_routes)

        # Handle cases where no intent is identified
        if len(intent_routes) == 0:
            return None
        else:
            intention = intent_routes[0].name  # Use the first matched intent

        # Validate the retrieved intention and handle unexpected types
        if intention is None:
            return "None"
        else:
            return intention

    def handle_product_information(self, user_input: Dict[str, str]) -> str:
        """Handle the product information intent by processing user input and providing a response.

        Args:
            user_input: The input text from the user.

        Returns:
            The content of the response after processing through the chains.
        """
        # Retrieve reasoning and response chains for the product information intent
        reasoning_chain, response_chain = self.get_chain("product_information")

        # Process user input through the reasoning chain
        reasoning_output: AIMessage = reasoning_chain.invoke(user_input)

        # Generate a response using the output of the reasoning chain
        response: AIMessage = response_chain.invoke(
            reasoning_output, config=self.memory_config
        )

        return response.content

    def handle_order_intent(self, user_input: Dict[str, str]) -> str:
        """Handle the order intent by processing user input and providing a response.

        Args:
            user_input: The input text from the user.

        Returns:
            The content of the response after processing through the chains.
        """
        # Retrieve the agent for the order intent
        agent = self.get_agent("order")

        # Process user input through the agent
        response: Dict[str, str] = agent.invoke(
            {
                "customer_id": self.user_id,
                "customer_input": user_input["customer_input"],
            },
            config=self.memory_config,
        )

        return response["output"]

    def handle_support_information(self, user_input: Dict[str, str]) -> str:

        response = self.rag.invoke(user_input, config=self.memory_config)

        return response

    def handle_chitchat_intent(self, user_input: Dict[str, str]) -> str:
        """Handle the chitchat intent by providing a response.

        Args:
            user_input: The input text from the user.

        Returns:
            The content of the response after processing through the chitchat chain.
        """
        _, chitchat_response_chain = self.get_chain("chitchat")

        response = chitchat_response_chain.invoke(user_input, config=self.memory_config)
        return response

    def handle_unknown_intent(self, user_input: Dict[str, str]) -> str:
        """Handle unknown intents by providing a chitchat response.

        Args:
            user_input: The input text from the user.

        Returns:
            The content of the response after processing through the new chain.
        """
        possible_intention = [
            "Product Information",
            "Create Order",
            "Order Status",
            "Support Information",
            "Chitchat",
        ]

        chitchat_reasoning_chain, _ = self.get_chain("chitchat")

        input_message = {}

        input_message["customer_input"] = user_input["customer_input"]
        input_message["possible_intentions"] = possible_intention
        input_message["chat_history"] = self.memory.get_session_history(
            self.user_id, self.conversation_id
        )

        reasoning_output1 = chitchat_reasoning_chain.invoke(input_message)

        if reasoning_output1.chitchat:
            print("Chitchat")
            return self.handle_chitchat_intent(user_input)
        else:
            router_reasoning_chain2, _ = self.get_chain("router")
            reasoning_output2 = router_reasoning_chain2.invoke(input_message)
            new_intention = reasoning_output2.intent
            print("New Intention:", new_intention)
            new_handler = self.intent_handlers.get(new_intention)
            return new_handler(user_input)

    def save_memory(self) -> None:
        """Save the current memory state of the bot."""
        self.memory.save_session_history(self.user_id, self.conversation_id)

    def process_user_input(self, user_input: Dict[str, str]) -> str:
        """Process user input by routing through the appropriate intention pipeline.

        Args:
            user_input: The input text from the user.

        Returns:
            The content of the response after processing through the chains.
        """
        # Classify the user's intent based on their input
        intention = self.get_user_intent(user_input)

        print("Intent:", intention)

        # Route the input based on the identified intention
        handler = self.intent_handlers.get(intention, self.handle_unknown_intent)
        return handler(user_input)

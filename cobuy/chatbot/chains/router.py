from typing import Literal

from langchain import callbacks
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.runnable.base import Runnable
from pydantic import BaseModel, Field

from cobuy.chatbot.chains.base import PromptTemplate, generate_prompt_templates


class IntentClassification(BaseModel):

    intent: Literal[
        "product_information",
        "create_order",
        "order_status",
        "support_information",
    ] = Field(
        ...,
        description="The classified intent of the user query",
    )


class RouterChain(Runnable):
    def __init__(self, llm, memory=False):
        super().__init__()

        self.llm = llm
        prompt_template = PromptTemplate(
            system_template="""
            You are an expert classifier of user intentions for the Cobuy e-commerce platform,
            which specializes in electronics. Your role is to accurately identify the user's 
            intent based on their query and the context provided by the conversation history.
            Analyze the user's query in the conversation history context and classify 
            it into one of the intents: Product Information, Create Order, Order Status, 
            or Support Information. You'll use the following detailed descriptions to classify the user's intent:

            1. **Product Information:**  
            The user is seeking details about a specific product on Cobuy. They might inquire 
            about product features, specifications, price, warranty, brand, model number, 
            or general category information. The user typically refers to the product by its 
            name or as 'it' or 'this product'.

            2. **Create Order:**  
            The user intends to place an order for a product they've already selected on Cobuy.
            They may not specify the exact quantity, and will refer to the product by its 
            name or by using terms like 'it' or 'this product'.

            3. **Order Status:**  
            The user wants to know about the status of an existing order. 
            They provide their order number and may ask about delivery date, expected delivery
            time, or the current location of the order.

            4. **Support Information:**  
            The user is interested in details about Cobuy's electronic product platform, including:
            - **Pricing:** Discounts, promotions, and transparency.
            - **Availability:** Stock status, pre-orders, restock notifications.
            - **Delivery:** Estimated times, shipping regions, restrictions.
            - **Returns:** Policies, time frames, refund processes, exchange options.
            - **Customer Support:** Contact methods, technical assistance, setup guidance.
            - **Additional Services:** Payment options, user manuals, repair services, and 
            warranties.

            **Input:**

            - Customer Input: {customer_input}  
            - Conversation History: {chat_history}

            **Output Format:**

            - Follow the specified output format and use these detailed descriptions:
            {format_instructions}
            """,
            human_template="User Query: {customer_input}",
        )

        self.prompt = generate_prompt_templates(prompt_template, memory=memory)

        self.output_parser = PydanticOutputParser(pydantic_object=IntentClassification)
        self.format_instructions = self.output_parser.get_format_instructions()
        self.chain = (self.prompt | self.llm | self.output_parser).with_config(
            {"run_name": self.__class__.__name__}
        )  # Add a run name to the chain on LangSmith

    def invoke(self, input, config=None, **kwargs):
        """Invoke the product information response chain."""
        with callbacks.collect_runs() as cb:
            return self.chain.invoke(
                {
                    "customer_input": input["customer_input"],
                    "chat_history": input["chat_history"],
                    "format_instructions": self.format_instructions,
                },
            )

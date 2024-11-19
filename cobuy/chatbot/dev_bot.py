# Import necessary classes and modules
from typing import Dict
from cobuy.chatbot.router.loader import load_intention_classifier
from cobuy.chatbot.router.auxiliar import add_message


class DevCustomerServiceBot:
    """A bot that handles customer service interactions by processing user inputs and
    routing them through configured reasoning and response chains.
    """

    def __init__(self, user_id: str, conversation_id: str, intentions: list):
        """Initialize the bot with session and language model configurations.

        Args:
            user_id: Identifier for the user.
            conversation_id: Identifier for the conversation.
        """
        self.intention_classifier = load_intention_classifier()
        self.intentions = intentions

    def get_choice_from_list(self):
        print()
        print("Available intentions:")
        for i, choice in enumerate(self.intentions, 1):
            print(f"{i}. {choice}")

        while True:
            try:
                user_input = int(input("Select a new intention: "))
                if 1 <= user_input <= len(self.intentions):
                    return self.intentions[user_input - 1]
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def process_user_input(self, user_input: Dict):
        """Process user input by routing through the appropriate reasoning and response chains.

        Args:
            user_input: The input text from the user.

        Returns:
            The content of the response after processing through the chains.
        """
        # Step 1: Classify intent (using placeholder intent classification)
        intent_route = self.intention_classifier(user_input["customer_input"])

        intention = intent_route.name

        print(f"Cobuy: {intention}")

        is_correct = input("It's correct? [Y/N]: ").strip()

        if is_correct.lower() == "y":
            pass
        elif is_correct.lower() == "n":
            new_intention = self.get_choice_from_list()
            new_item = {
                "Intention": new_intention,
                "Message": user_input["customer_input"],
            }
            add_message(new_item, "new_intentions.json")
            print("New intention added successfully")
        else:
            print("Cobuy: I'm sorry, I didn't understand that. Let's try again.")

        print("")

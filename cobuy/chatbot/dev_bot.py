# Import necessary classes and modules
from typing import Dict
from cobuy.chatbot.router.auxiliar import add_message
from cobuy.chatbot.bot import CustomerServiceBot


class DevCustomerServiceBot(CustomerServiceBot):
    """A bot that handles customer service interactions by processing user inputs and
    routing them through configured reasoning and response chains.
    """

    def __init__(self, user_id: str, conversation_id: str, intentions: list):

        super().__init__(user_id, conversation_id)
        self.intentions = intentions

    def get_choice_from_list(self):
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

    def validate_intention(self, intention: str):

        print("")
        print(f"Predicted Intention: {intention}")
        is_correct = input("It's correct? [Y/N]: ").strip()
        print("")

        if is_correct.lower() == "y":
            return True

        elif is_correct.lower() == "n":
            return False
        else:
            return None

    def create_new_user_messages(self, user_input: Dict):

        new_intention = self.get_choice_from_list()
        new_item = {
            "Intention": new_intention,
            "Message": user_input["customer_input"],
        }
        add_message(new_item, "new_intentions.json")

    def process_user_input(self, user_input: Dict):
        """Process user input by routing through the appropriate reasoning and response chains.

        Args:
            user_input: The input text from the user.

        Returns:
            The content of the response after processing through the chains.
        """
        # Step 1: Classify the intent using the intention classifier
        intention = self.get_user_intent(user_input)

        # Step 2: Validate if the intention is correct
        is_correct = self.validate_intention(intention)

        # Step 3: Check if the intention is correct
        if is_correct:
            # Step 4: Process the user input based on the intention
            if intention == "product_information":
                response = self.handle_product_information(user_input)
                return response
            else:
                return "I'm sorry, I don't understand that intention."
        else:
            if is_correct is None:
                print("You should enter 'Y' or 'N'. Please try again.")
            # Step 5: Create a new intention
            else:
                self.create_new_user_messages(user_input)
                return "New intention added successfully."

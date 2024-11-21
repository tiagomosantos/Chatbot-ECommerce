from dotenv import load_dotenv  # Import dotenv to load environment variables

from cobuy import CustomerServiceBot  # Import the chatbot class


def main(bot: CustomerServiceBot):
    """Main interaction loop for the chatbot.

    Args:
        bot: An instance of the CustomerServiceBot.
    """
    while True:
        # Prompt the user for input
        user_input = input("You: ").strip()

        # Allow the user to exit the conversation
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            # Process the user's input using the bot and display the response
            response = bot.process_user_input({"customer_input": user_input})
            print(f"Cobuy: {response}")
        except Exception as e:
            # Handle any exceptions and prompt the user to try again
            print(f"Error: {str(e)}")
            print("Please try again with a different query.")


if __name__ == "__main__":
    # Load environment variables from a .env file
    load_dotenv()

    # Notify the user that the bot is starting
    print("Starting the bot...")

    # Initialize the CustomerServiceBot with dummy user and conversation IDs
    bot = CustomerServiceBot(user_id="user_123", conversation_id="conversation_123")

    # Display instructions for ending the conversation
    print(
        "Customer Service Bot initialized. Type 'exit' or 'quit' to end the conversation."
    )

    # Start the main interaction loop
    main(bot)

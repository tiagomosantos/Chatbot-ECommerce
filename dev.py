from dotenv import load_dotenv  # Import dotenv to load environment variables

from cobuy import DevCustomerServiceBot  # Import the development version of the chatbot


def main(bot: DevCustomerServiceBot):
    """Main interaction loop for the development version of the chatbot.

    Args:
        bot: An instance of the DevCustomerServiceBot.
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

    # Notify the user that the bot is running in development mode
    print("RUNNING IN DEV MODE.")
    print("Starting the bot...")

    # Define a list of intentions for the development bot
    intentions = ["order_status", "create_order", "product_information"]

    # Initialize the development version of the CustomerServiceBot
    bot = DevCustomerServiceBot(
        user_id="user_123", conversation_id="conversation_123", intentions=intentions
    )

    # Display instructions for ending the conversation
    print(
        "Customer Service Bot initialized. Type 'exit' or 'quit' to end the development process."
    )

    # Start the main interaction loop
    main(bot)

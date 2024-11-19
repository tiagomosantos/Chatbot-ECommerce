from cobuy import CustomerServiceBot
from dotenv import load_dotenv


def main(bot: CustomerServiceBot):

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            response = bot.process_user_input({"customer_input": user_input})
            print(f"Cobuy: {response}")
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try again with a different query.")


if __name__ == "__main__":
    # Load environment variables from a .env file
    load_dotenv()

    print("Starting the bot...")

    bot = CustomerServiceBot(user_id="user_123", conversation_id="conversation_123")

    print(
        "Customer Service Bot initialized. Type 'exit' or 'quit' to end the conversation."
    )

    main(bot)

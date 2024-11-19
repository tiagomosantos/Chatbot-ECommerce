from cobuy import DevCustomerServiceBot
from dotenv import load_dotenv


def main(bot: DevCustomerServiceBot):

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            # You can specify different intentions here
            bot.process_user_input({"customer_input": user_input})
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try again with a different query.")


if __name__ == "__main__":
    # Load environment variables from a .env file
    load_dotenv()

    print("RUNNING IN DEV MODE.")
    print("Starting the bot...")

    intentions = ["order_status", "create_order", "product_information"]
    bot = DevCustomerServiceBot(
        user_id="user_123", conversation_id="conversation_123", intentions=intentions
    )

    print(
        "Customer Service Bot initialized. Type 'exit' or 'quit' to end the developement process."
    )

    main(bot)

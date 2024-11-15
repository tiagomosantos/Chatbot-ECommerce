from bot.chatbot import CustomerServiceBot
from dotenv import load_dotenv

load_dotenv()

bot = CustomerServiceBot(user_id="user_123", conversation_id="conversation_123")

print(
    "Customer Service Bot initialized. Type 'exit' or 'quit' to end the conversation."
)

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    try:
        # You can specify different intentions here
        response = bot.process_user_input({"customer_input": user_input})
        print(f"Bot: {response}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please try again with a different query.")

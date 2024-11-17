# E-commerce Chatbot with LangChain 🤖🛒

Welcome to the E-commerce Chatbot project! This chatbot leverages the power of LangChain to create a dynamic, interactive shopping assistant capable of managing products, clients, and orders. By integrating Tools, Chains, and Agents, this chatbot can perform complex tasks like order creation and query handling, all through natural language interactions.

# Project Structure

```plaintext
root/
├── app.py                # Main Streamlit application script.
├── synthetic_data.ipynb  # Notebook for generating synthetic data.
├── requirements.txt      # Python dependencies.
├── .gitignore            # Standard .gitignore file.
├── README.md             # Comprehensive project documentation.
├── tests/                # Testing modules.
│   ├── intentions.ipynb  # Tests for chatbot intentions.
│   └── README.md         # Test examples documentation.
├── company_name/         # Replace "company_name" with your company name.
│   ├── chatbot/          # Chatbot modules and assets.
│   │   ├── __init__.py   # Package marker.
│   │   ├── bot.py        # Core chatbot logic.
│   │   ├── memory.py     # Chatbot memory.
│   │   ├── chains/       # Custom LangChain chains.
│   │   │   └── *.py      # Chain modules.
│   │   ├── tools/        # Utility scripts.
│   │   │   └── *.py      # Utility modules.
│   │   ├── agents/       # Custom agents.
│   │   │   └── *.py      # Agent modules.
│   │   ├── rag/          # RAG-related modules for retrieval-augmented generation.
│   │   │   └── *.py       # Scripts for retrieval, embedding, ranking, QA pipelines, and utilities.
│   ├── pages/            # Streamlit app pages.
│   │   └── *.py          # Page modules.
│   ├── data/             # Data and scripts.
│   │   │── loader.py     # Functions to load data.
│   │   ├── database/     # Database files/scripts.
│   │   │   ├── *.db      # SQLite databases.
│   │   │   ├── *.ipynb   # Scripts for database creation.
│   │   ├── pdfs/         # PDFs and embedding scripts.
│   │   │   ├── *.pdf     # PDF files.
│   │   │   └── *.ipynb   # PDF processing scripts.

from operator import itemgetter
from typing import Dict, List

from dotenv import load_dotenv
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Index, Pinecone

from cobuy.chatbot.chains.base import PromptTemplate, generate_prompt_templates


class RAGPipeline:
    """
    A class to encapsulate a Retrieval-Augmented Generation (RAG) pipeline.
    This class sets up a Pinecone vector store for document retrieval and a language model for question answering.
    """

    def __init__(
        self,
        index_name: str,
        embeddings_model: str,
        llm: ChatOpenAI,
        memory: bool = False,
    ):
        """
        Initializes the RAGPipeline with Pinecone, vector store, and LLM components.

        Args:
            index_name (str): The name of the Pinecone index.
            embeddings_model (str): The OpenAI model to use for embeddings.
            llm (ChatOpenAI): The language model for question answering.
        """
        # Load environment variables from a .env file
        load_dotenv()

        # Initialize Pinecone and set up the index
        self.pc = Pinecone()
        self.index: Index = self.pc.Index(index_name)

        # Create a vector store with the given index and embedding model
        self.vector_store = PineconeVectorStore(
            index=self.index,
            embedding=OpenAIEmbeddings(model=embeddings_model),
        )

        # Configure the retriever with similarity search and score threshold
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.5},
        )

        # Define the custom RAG prompt template
        self.prompt_template = PromptTemplate(
            system_template="""Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Use three sentences maximum and keep the answer as concise as possible.
            You have acess to the previous conversation history to personalize the conversation.

            {context}

            Question: {customer_input}

            Helpful Answer:""",
            human_template="Customer Query: {customer_input}",
        )

        self.prompt = generate_prompt_templates(self.prompt_template, memory=memory)

        # Initialize the language model
        self.llm = llm

        # Combine components into a RAG chain

        context = itemgetter("customer_input") | self.retriever | self._format_docs
        first_step = RunnablePassthrough.assign(context=context)
        self._rag_chain = first_step | self.prompt | self.llm | StrOutputParser()

    @staticmethod
    def _format_docs(documents: List[Document]):
        """
        Formats retrieved documents into a single string for context input to the model.

        Args:
            documents (list): List of documents retrieved by the retriever.

        Returns:
            str: Concatenated document content separated by double newlines.
        """
        return "\n\n".join(doc.page_content for doc in documents)

    @property
    def rag_chain(self):
        """
        Getter for the RAG chain.

        Returns:
            Runnable: The RAG chain.
        """
        return self._rag_chain

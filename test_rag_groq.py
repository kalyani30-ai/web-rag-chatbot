
from embeddings import get_embedding_model  # Loads the shared question embedding model.
from vector_store import load_vector_store, search_vector_store  # Imports functions to load and search FAISS.
from prompt_builder import build_rag_prompt  # Imports the function that prepares the RAG prompt.
from groq_client import generate_answer # Imports the function that sends the prompt to Groq.


def retrieve_chunks(question):  # Retrieves relevant information from the saved database.

    index, chunks = load_vector_store()  # Loads the saved FAISS index and original text chunks.

    query_embedding = get_embedding_model().encode([question])[0]  # Converts the user's question into a 384-dimensional vector.

    distances, indices = search_vector_store(index, query_embedding, top_k=3)  # Finds the three closest matching vectors.

    retrieved_chunks = []  # Creates an empty list to store retrieved information.

    for position, chunk_index in enumerate(indices[0]):  # Goes through each matching vector position.

        if chunk_index == -1:  # Checks whether FAISS returned an invalid position.
            continue  # Skips invalid results.

        retrieved_chunks.append({  # Stores the retrieved text and its source URL.
            "text": chunks[chunk_index]["text"],  # Retrieves the original text chunk.
            "url": chunks[chunk_index]["url"]  # Retrieves the original website URL.
        })

    return retrieved_chunks  # Returns all retrieved chunks.


if __name__ == "__main__":  # Runs this section when the file is executed directly.

    print("WELCOME TO YOUR RAG CHATBOT")  # Displays the chatbot heading.

    question = input("\nEnter your question: ")  # Accepts a question from the user.

    if question.strip():  # Checks whether the user entered a non-empty question.

        retrieved_chunks = retrieve_chunks(question)  # Retrieves relevant information from FAISS.

        print("\nRelevant information retrieved successfully!")  # Confirms successful retrieval.

        prompt = build_rag_prompt(question, retrieved_chunks)  # Combines the question and retrieved chunks into a prompt.

        print("\nGenerating your answer using Groq...") # Displays a message before answer generation.

        try:  # Starts error handling for the Groq API request.

            answer = generate_answer(prompt)  # Sends the RAG prompt to Groq and receives its answer.

            print("\nCHATBOT ANSWER")  # Displays the answer heading.

            print(answer)  # Displays Groq's generated answer.

        except Exception as error:  # Handles errors returned by Groq.

            print("\nGROQ API ERROR") # Displays an error heading.

            print(error)  # Displays the error details without exposing your API key.

    else:  # Runs when the user enters an empty question.

        print("Please enter a valid question.")  # Asks the user to enter a question.

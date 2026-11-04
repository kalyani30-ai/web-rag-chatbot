
from embeddings import get_embedding_model  # Loads the shared question embedding model.
from vector_store import load_vector_store, search_vector_store  # Imports functions to load and search the saved FAISS database.
from prompt_builder import build_rag_prompt  # Imports the function that prepares the Gemini prompt.


def retrieve_chunks(question):  # Retrieves relevant information from the saved database.

    index, chunks = load_vector_store()  # Loads the saved FAISS index and text chunks.

    query_embedding = get_embedding_model().encode([question])[0]  # Converts the question into a numerical vector.

    distances, indices = search_vector_store(index, query_embedding, top_k=3)  # Finds the three closest matching vectors.

    retrieved_chunks = []  # Creates a list to store the retrieved text.

    for position, chunk_index in enumerate(indices[0]):  # Goes through the matching vector positions.

        if chunk_index == -1:  # Checks whether the returned position is invalid.
            continue  # Skips invalid positions.

        retrieved_chunks.append({  # Stores the matching text and its source URL.
            "text": chunks[chunk_index]["text"],  # Retrieves the original text chunk.
            "url": chunks[chunk_index]["url"]  # Retrieves the original website URL.
        })

    return retrieved_chunks  # Returns the retrieved information.


if __name__ == "__main__":  # Runs this section when the file is executed directly.

    question = input("Enter your question: ")  # Accepts a question from the user.

    if question.strip():  # Checks whether the question is not empty.

        retrieved_chunks = retrieve_chunks(question)  # Retrieves relevant information from FAISS.

        prompt = build_rag_prompt(question, retrieved_chunks)  # Combines the question and retrieved information into a prompt.

        print("\nPROMPT PREPARED FOR GEMINI")  # Displays the prompt heading.

        print(prompt)  # Displays the complete prompt.

    else:  # Runs if the user enters an empty question.
        print("Please enter a valid question.")  # Displays an input validation message.

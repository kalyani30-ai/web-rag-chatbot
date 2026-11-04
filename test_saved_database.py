
from embeddings import get_embedding_model  # Loads the shared question embedding model.
from vector_store import load_vector_store, search_vector_store  # Imports functions to load the saved FAISS database and search it.


def search_saved_database(question):  # Defines a function that searches the previously saved database.

    index, chunks = load_vector_store()  # Loads the FAISS index and its corresponding text chunks from disk.

    print("\nConverting your question into an embedding...")  # Displays a message before converting the question.

    query_embedding = get_embedding_model().encode([question])[0]  # Converts the question into a 384-dimensional numerical vector.

    print("Searching the saved database...")  # Displays a message before searching FAISS.

    distances, indices = search_vector_store(index, query_embedding, top_k=3)  # Retrieves the three closest matching vectors.

    print("\nSEARCH RESULTS")  # Displays the search results heading.

    for position, chunk_index in enumerate(indices[0]):  # Goes through the positions of the retrieved vectors.

        if chunk_index == -1:  # Checks whether FAISS returned an invalid result.
            continue  # Skips invalid results.

        print(f"\nResult {position + 1}")  # Displays the result number.

        print("Distance:", round(float(distances[0][position]), 4))  # Displays the L2 distance between the question and retrieved vector.

        print("Source:", chunks[chunk_index]["url"])  # Displays the original website URL.

        print("Text:", chunks[chunk_index]["text"][:500])  # Displays the first 500 characters of the matching text chunk.


if __name__ == "__main__":  # Runs this section when the file is executed directly.

    print("SAVED RAG DATABASE TEST")  # Displays the program heading.

    user_question = input("\nEnter your question: ")  # Accepts a question from the user through the terminal.

    if user_question.strip():  # Checks whether the user entered a non-empty question.

        search_saved_database(user_question)  # Searches the saved FAISS database using the entered question.

    else:  # Runs when the user enters nothing.
        print("Please enter a valid question.")  # Asks the user to provide a question.

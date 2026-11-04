from embeddings import get_embedding_model  # Loads the shared sentence embedding model.
from vector_store import create_vector_store, search_vector_store  # Imports the FAISS creation and search functions.

sentences = [  # Creates three sample documents for testing.
    "Artificial Intelligence helps computers perform intelligent tasks.",
    "Machine learning is a branch of Artificial Intelligence.",
    "I enjoy eating pizza."
]

model = get_embedding_model()  # Reuses the one initialized model.
embeddings = model.encode(sentences)  # Converts each sentence into a 384-dimensional vector.

index = create_vector_store(embeddings)  # Stores all three vectors inside the FAISS index.

question = "What is machine learning?"  # Defines the question we want FAISS to search for.

question_embedding = model.encode(question)  # Converts the question into a numerical vector.

distances, indices = search_vector_store(index, question_embedding, top_k=2)  # Retrieves the two closest document vectors.

print("\nUser Question:", question)  # Displays the question.

print("\nMost Relevant Results:")  # Displays a heading for the search results.

for i in range(len(indices[0])):  # Loops through each of the two retrieved results.
    position = indices[0][i]  # Gets the original position of the matching sentence.
    distance = distances[0][i]  # Gets the distance between the question and matching sentence.

    print("\nSentence:", sentences[position])  # Displays the matching sentence.
    print("Distance:", distance)  # Displays its distance from the question vector.

import re  # Imports Python's regular-expression library for cleaning text.


def clean_text(text: str) -> str:
    """Collapse layout whitespace so scraped pages can be split consistently."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    text = re.sub(r'\s+', ' ', text)  # Replaces multiple spaces, tabs and newlines with one space.

    text = text.strip()  # Removes unnecessary spaces from the beginning and end.

    return text  # Returns the cleaned text.


def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping word windows without losing the source order."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least 0 and smaller than chunk_size.")

    words = text.split()  # Divides the complete text into a list of individual words.

    chunks = []  # Creates an empty list to store the generated text chunks.

    start = 0  # Sets the starting word position for the first chunk.

    while start < len(words):  # Continues while there are still words left to process.

        end = start + chunk_size  # Calculates the ending position of the current chunk.

        chunk_words = words[start:end]  # Selects the words belonging to the current chunk.

        chunk = " ".join(chunk_words)  # Combines the selected words into a readable sentence string.

        chunks.append(chunk)  # Adds the completed chunk to the chunks list.

        if end >= len(words):  # Checks whether the current chunk has reached the end of the text.
            break  # Stops the loop because there are no more words to process.

        start = end - overlap  # Starts the next chunk 50 words before the previous chunk ended.

    return chunks  # Returns the complete list of text chunks.


# This block runs only when processor.py is executed directly.
if __name__ == "__main__":

    sample_text = """
    Artificial Intelligence enables computers to perform tasks that normally
    require human intelligence. Machine learning is a subset of Artificial
    Intelligence. Deep learning uses neural networks to learn complex patterns.
    AI is used in healthcare, education, transportation and many other fields.
    """  # Creates sample text so we can test our functions without using an API.

    cleaned_text = clean_text(sample_text)  # Cleans the sample text using our first function.

    print("Cleaned Text:\n")  # Displays a heading before printing the cleaned text.

    print(cleaned_text)  # Prints the cleaned sample text.

    chunks = split_into_chunks(cleaned_text, chunk_size=15, overlap=3)  # Splits sample text into small chunks for easy testing.

    print("\nGenerated Chunks:\n")  # Displays a heading before printing the chunks.

    for index, chunk in enumerate(chunks, start=1):  # Loops through chunks and numbers them starting from 1.

        print(f"Chunk {index}: {chunk}")  # Displays each chunk with its number.

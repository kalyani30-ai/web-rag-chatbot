"""Make one short, independent Groq API request for connection diagnosis."""

import time

from groq_client import (
    CONNECTION_TEST_PROMPT,
    MODEL_NAME,
    GroqClientError,
    generate_answer,
)


if __name__ == "__main__":
    print(f"Testing Groq model {MODEL_NAME} with a short prompt...")
    started = time.perf_counter()
    try:
        answer = generate_answer(CONNECTION_TEST_PROMPT)
    except GroqClientError as error:
        print(f"Groq test failed: {error}")
        print(f"Elapsed time: {time.perf_counter() - started:.2f} seconds")
        raise SystemExit(1) from error

    safe_answer = answer.encode("ascii", "backslashreplace").decode("ascii")
    print(f"Groq response: {safe_answer}")
    print(f"Elapsed time: {time.perf_counter() - started:.2f} seconds")

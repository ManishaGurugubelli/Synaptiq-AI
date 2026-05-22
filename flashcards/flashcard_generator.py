from groq import Groq
from dotenv import load_dotenv
import os
import json

from rag.rag_pipeline import retrieve_context

# Load environment variables
load_dotenv()

# Initialize Groq
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Generate Flashcards
def generate_flashcards(topic):

    context = retrieve_context(topic)

    prompt = f"""
    Generate 5 flashcards
    based ONLY on the context.

    Return ONLY valid JSON.

    Format:
    [
      {{
        "question": "Question",
        "answer": "Answer"
      }}
    ]

    Context:
    {context}
    """

    completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama-3.1-8b-instant",
    )

    response = (
        completion
        .choices[0]
        .message.content
    )

    flashcards = json.loads(response)

    return flashcards
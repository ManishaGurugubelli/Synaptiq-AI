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

# Generate Quiz
def generate_quiz(topic):

    context = retrieve_context(topic)

    prompt = f"""
    Generate 5 multiple-choice questions
    based ONLY on the context.

    Return ONLY valid JSON.

    IMPORTANT:
    - correct_answer must contain
      ONLY the correct option index.
    - Use numbers:
      0,1,2,3

    Format:
    [
      {{
        "question": "Question text",

        "options": [
          "Option 1",
          "Option 2",
          "Option 3",
          "Option 4"
        ],

        "correct_answer": 0
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

    response = response.replace(
        "```json",
        ""
    ).replace(
        "```",
        ""
    )

    quiz = json.loads(response)

    return quiz
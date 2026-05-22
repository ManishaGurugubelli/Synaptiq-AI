from groq import Groq
from dotenv import load_dotenv
import os

from rag.rag_pipeline import retrieve_context

# Load environment variables
load_dotenv()

# Initialize Groq
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Generate Summary
def generate_summary(topic):

    context = retrieve_context(topic)

    prompt = f"""
    You are Synaptiq AI.

    Generate concise study notes
    based ONLY on the context.

    Context:
    {context}

    Format:
    - Headings
    - Bullet points
    - Easy explanation
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

    return response
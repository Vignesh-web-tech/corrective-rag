from groq import Groq

from app.config import settings


client = Groq(
    api_key=settings.GROQ_API_KEY
)


MODEL = "llama-3.3-70b-versatile"


SYSTEM_PROMPT = """
You are a helpful RAG assistant.

Answer the user's question ONLY using the
provided document context.

Rules:
1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not present in the context,
   say:
   "I couldn't find that information in the uploaded document."
4. Give a clear and concise answer.
"""


def generate_answer(
    query: str,
    context: str
):

    prompt = f"""
Document Context:

{context}


User Question:

{query}


Answer the question using ONLY the document context.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content
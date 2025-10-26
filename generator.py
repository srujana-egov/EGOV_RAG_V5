import os
from dotenv import load_dotenv
load_dotenv()
import openai
import json

def chat_with_assistant(query, docs, model="gpt-4"):
    """
    Calls OpenAI chat completion model with query and supporting docs.
    Returns a structured JSON array as Python list.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in environment variables.")
    
    client = openai.OpenAI(api_key=api_key)

    # Combine docs into a single context string
    context = "\n\n".join([
        f"--- {doc.get('title', 'No Title')} ---\n{doc['content']}" if isinstance(doc, dict) else doc
        for doc in docs
    ])

    # Strong prompt for structured JSON output
    prompt = f"""
You are a precise assistant that answers questions using only the provided context.

Instructions:
1. Use only information in the context.
2. If asked for a **JSON array**, return a **complete JSON array** with **one object per field**, using exactly this schema:

{{
    "Field Name": "",
    "Description": "",
    "Mandatory": true/false,
    "Input": "User" or "System",
    "Data Type": "",
    "Min Length": int or null,
    "Max Length": int or null,
    "Minimum Value": number or null,
    "Maximum Value": number or null,
    "Need Data from Program or State": true/false
}}

3. If information is missing, use `null` for string, number, or boolean fields.
4. Merge complementary information from multiple chunks if needed.
5. Add a separate field at the end for URLs used: `"Reference URLs": []`
6. Return only **valid JSON**, no extra text, no markdown, no comments.

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful, precise assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    answer_text = response.choices[0].message.content

    try:
        # Convert JSON string to Python list/dict
        answer_json = json.loads(answer_text)
        return answer_json
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}\nResponse was:\n{answer_text}")


if __name__ == "__main__":
    # Example usage
    docs = [
        {"title": "Campaign Setup Spec", "content": "Your context text goes here..."}
    ]
    query = "Generate a JSON array of all fields for Campaign Setup."
    result = chat_with_assistant(query, docs)
    print(json.dumps(result, indent=2))

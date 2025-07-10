import logging
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the LLM (e.g., Mistral or LLaMA2)
try:
    llm = pipeline("text-generation", model="mistralai/Mixtral-8x7B-Instruct-v0.1")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {str(e)}", exc_info=True)
    llm = None

def generate_answer_with_llm(documents, query, history=None, model="mistral", domain="general"):
    """
    Generate an answer using the specified LLM model, incorporating documents, query, history, and domain.
    
    Args:
        documents (list): List of dictionaries with 'text', 'url', and 'score' keys.
        query (str): The user's query.
        history (list): List of previous conversation messages.
        model (str): The LLM model to use (e.g., 'mistral', 'llama2').
        domain (str): The domain to focus the response on (e.g., 'sport').
    
    Returns:
        str: The generated answer.
    """
    if llm is None:
        logger.error("LLM model failed to initialize.")
        return "Erreur : impossible de générer une réponse en raison d'un problème avec le modèle."

    # Prepare context from documents
    context = ""
    for doc in documents:
        text = doc.get("text", "")
        if text:
            context += f"- {text}\n"

    # Prepare conversation history
    history_text = ""
    if history:
        for msg in history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", msg.get("user", msg.get("bot", "")))
            history_text += f"{role}: {content}\n"

    # Construct the prompt, incorporating the domain
    domain_prompt = f"Focus on the '{domain}' domain when answering." if domain != "general" else ""
    prompt = f"""
{domain_prompt}
Based on the following context and conversation history, provide a concise and accurate answer to the query: "{query}"

Context:
{context}

Conversation History:
{history_text}

Answer:
"""
    logger.info(f"Generating answer with prompt: {prompt[:200]}...")  # Log first 200 chars for brevity

    try:
        # Generate response using the LLM
        response = llm(prompt, max_length=200, num_return_sequences=1, do_sample=True, temperature=0.7)
        answer = response[0]["generated_text"].strip()
        # Extract the answer part after "Answer:" if present
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        logger.info(f"Generated answer: {answer[:100]}...")  # Log first 100 chars
        return answer
    except Exception as e:
        logger.error(f"Error generating answer with LLM: {str(e)}", exc_info=True)
        return f"Erreur : impossible de générer une réponse ({str(e)})."
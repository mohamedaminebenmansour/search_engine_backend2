from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .web_scraper import scrape_web
from .llm_answer import generate_answer_with_llm
from utils.message_filter import analyze_message
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Explicitly initialize SentenceTransformer to avoid conflicts
try:
    transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    logger.error(f"Failed to initialize SentenceTransformer: {str(e)}", exc_info=True)
    transformer_model = None

chat_memory = {}

def hybrid_search(query, user_id=None, top_k=5, model="llama2", domain="general"):
    """
    Perform a hybrid search and generate a response using the specified LLM model.
    
    Args:
        query: User query string.
        user_id: User ID for context (optional).
        top_k: Number of top documents to consider.
        model: Name of the LLM model to use.
        domain: Domain to focus the search on.
    
    Returns:
        Dictionary with answer and sources.
    """
    if not isinstance(query, str):
        logger.error(f"Invalid query type: {type(query)}. Expected string.")
        return {"answer": "Erreur : la requête doit être une chaîne de caractères.", "sources": []}

    if transformer_model is None:
        logger.error("SentenceTransformer model failed to initialize.")
        history = chat_memory.get(user_id, []) if user_id else []
        answer = generate_answer_with_llm([], query, history=history, model=model, domain=domain)
        return {
            "answer": f"Erreur : impossible de générer des embeddings, mais voici une réponse : {answer}",
            "sources": []
        }

    analysis = analyze_message(query)

    response_prefix = ""
    if analysis["has_greeting"]:
        response_prefix += "Salut ! "
    if analysis["has_thanks"]:
        response_prefix += "Merci à toi ! "

    if not analysis["is_technical"]:
        if response_prefix:
            return {"answer": response_prefix + "Comment puis-je t'aider ? 😊", "sources": []}
        else:
            return {"answer": "Je suis là pour t'aider ! Pose-moi ta question 😊", "sources": []}

    logger.info(f"Scraping web for query: {query} in domain: {domain}")
    web_results = scrape_web(query, domain=domain)
    logger.info(f"Web results: {len(web_results)} documents retrieved")

    if not web_results:
        logger.warning("No web results found, falling back to LLM without context.")
        history = chat_memory.get(user_id, []) if user_id else []
        answer = generate_answer_with_llm([], query, history=history, model=model, domain=domain)
        return {
            "answer": response_prefix + answer,
            "sources": []
        }

    texts = [doc["text"] for doc in web_results if isinstance(doc.get("text"), str) and doc.get("text")]
    logger.info(f"Valid texts for embedding: {len(texts)}")
    if not texts:
        logger.warning("No valid texts found in web results.")
        history = chat_memory.get(user_id, []) if user_id else []
        answer = generate_answer_with_llm([], query, history=history, model=model, domain=domain)
        return {
            "answer": response_prefix + answer,
            "sources": []
        }

    try:
        query_embedding = transformer_model.encode(query)
        doc_embeddings = transformer_model.encode(texts)
        query_embedding = np.array(query_embedding).reshape(1, -1)
        doc_embeddings = np.array(doc_embeddings)
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    except Exception as e:
        logger.error(f"Error in embedding generation: {str(e)}", exc_info=True)
        history = chat_memory.get(user_id, []) if user_id else []
        answer = generate_answer_with_llm([], query, history=history, model=model, domain=domain)
        return {
            "answer": response_prefix + f"Erreur lors de la génération des embeddings, mais voici une réponse : {answer}",
            "sources": []
        }

    for i, doc in enumerate(web_results):
        doc["score"] = float(similarities[i]) if i < len(similarities) else 0.0
        doc["source"] = "web"

    sorted_docs = sorted(web_results, key=lambda x: x["score"], reverse=True)[:top_k]
    logger.info(f"Top {top_k} documents selected with scores: {[doc['score'] for doc in sorted_docs]}")

    history = chat_memory.get(user_id, []) if user_id else []
    answer = generate_answer_with_llm(sorted_docs, query, history=history, model=model, domain=domain)

    if user_id:
        chat_memory.setdefault(user_id, []).append({
            "user": query,
            "bot": answer
        })

    sources = []
    for doc in sorted_docs:
        url = doc.get("url")
        if url and isinstance(url, str) and url not in sources:
            sources.append(url)
        if len(sources) >= 4:
            break
    logger.info(f"Sources extracted: {sources}")

    return {
        "answer": response_prefix + answer,
        "sources": sources
    }
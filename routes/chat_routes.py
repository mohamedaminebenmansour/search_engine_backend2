from flask import Blueprint, request, jsonify, current_app
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from transformers import pipeline
from extensions import db
from models.history_model import History, Conversation
from models.user_model import User
from utils.auth_utils import decode_jwt
from utils.message_filter import analyze_message
from services.web_scraper import scrape_web
from services.llm_answer import generate_answer_with_llm
import logging
import numpy as np
import re
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize SentenceTransformer
try:
    transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    logger.error(f"Failed to initialize SentenceTransformer: {str(e)}", exc_info=True)
    transformer_model = None

# Initialize zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define available domains
DOMAINS = [
    "sport", "music", "movies", "technology",
    "science", "business", "health", "politics",
    "education", "food", "travel", "general"
]

chat_memory = {}
chat_bp = Blueprint("chat", __name__)

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
    if analysis.get("has_greeting"):
        response_prefix += "Salut ! "
    if analysis.get("has_thanks"):
        response_prefix += "Merci à toi ! "

    if not analysis.get("is_technical"):
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

def format_answer_for_readability(text):
    text = text.replace("\\n", "\n").replace("\r", "").strip()
    text = re.sub(r"(Je comprends mieux.*?exemples :) *\n*", r"\1\n\n", text, flags=re.IGNORECASE)
    lines = text.splitlines()
    numbered = []
    count = 1
    for line in lines:
        line = line.strip()
        if line.startswith("*"):
            content = line.lstrip("*").strip()
            numbered.append(f"{count}. {content}")
            count += 1
        else:
            numbered.append(line)
    text = "\n".join(numbered)
    text = re.sub(
        r"^Je comprends mieux.*?Voici quelques-uns des exemples :",
        "📊 **Taux de chômage les plus bas dans certains pays :**",
        text,
        flags=re.IGNORECASE
    )
    return text.strip()

def detect_domain(query, candidate_domains):
    try:
        result = classifier(query, candidate_domains, multi_label=False)
        return result['labels'][0]  # Return the most likely domain
    except Exception as e:
        current_app.logger.error(f"Domain classification failed: {str(e)}")
        return "general"

@chat_bp.route('/domains', methods=['GET'])
def get_domains():
    return jsonify({"domains": DOMAINS}), 200

@chat_bp.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Requête JSON manquante"}), 400

        query = data.get("query", "").strip()
        user_id = data.get("user_id")
        messages = data.get("messages", [])
        model = data.get("model", "mistral")
        history_id = data.get("history_id")
        domain = data.get("domain")  # Get domain from request

        auth_header = request.headers.get('Authorization')
        current_user = None

        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            try:
                user_id_from_token = decode_jwt(token)
                current_user = User.query.filter_by(id=user_id_from_token).first()
                if not current_user:
                    return jsonify({"error": "Token invalide ou utilisateur non trouvé"}), 401
            except Exception as e:
                current_app.logger.error(f"Token decoding failed: {str(e)}")
                return jsonify({"error": "Token expiré ou invalide"}), 401

        allowed_models = (
            ["llama2", "gemma", "llama3", "mistral"]
            if current_user else
            ["llama3", "mistral"]
        )

        if model not in allowed_models:
            return jsonify({"error": f"Modèle non supporté. Choisissez parmi : {', '.join(allowed_models)}"}), 400

        if not query:
            return jsonify({"error": "Le champ 'query' est requis"}), 400

        if current_user and user_id:
            if not isinstance(user_id, int) or user_id != current_user.id:
                return jsonify({"error": "Le champ 'user_id' doit être un entier et correspondre à l'utilisateur connecté"}), 400

        # Detect domain if not provided
        if not domain:
            domain = detect_domain(query, DOMAINS)
            current_app.logger.info(f"Detected domain: {domain}")

        try:
            analysis = analyze_message(query)
            greeting = analysis.get("greeting")
        except Exception as e:
            current_app.logger.error(f"Message analysis failed: {str(e)}")
            return jsonify({"error": "Erreur lors de l'analyse du message."}), 500

        try:
            result = hybrid_search(
                query=query,
                user_id=user_id if current_user else None,
                model=model,
                domain=domain
            )
            formatted_answer = format_answer_for_readability(result["answer"])
            if greeting:
                formatted_answer = f"{greeting} ! 😊\n\n{formatted_answer}"
            response = {
                "answer": formatted_answer,
                "sources": result.get("sources", []),
                "domain": domain
            }
        except Exception as e:
            current_app.logger.error(f"Hybrid search failed: {str(e)}", exc_info=True)
            return jsonify({"error": "Erreur dans la recherche hybride."}), 500

        if current_user and user_id:
            try:
                latest_history = db.session.query(History).filter_by(user_id=user_id).order_by(History.created_at.desc()).first()
                is_new_chat = (
                    not latest_history or
                    datetime.utcnow() - latest_history.created_at > timedelta(minutes=5) or
                    history_id is None
                )

                new_message = {"content": query, "role": "user", "id": str(datetime.utcnow().timestamp())}
                assistant_message = {"content": formatted_answer, "role": "assistant", "id": str(datetime.utcnow().timestamp() + 0.1)}
                updated_messages = messages + [new_message, assistant_message]

                if is_new_chat:
                    new_history = History(user_id=user_id, search_query=query)
                    db.session.add(new_history)
                    db.session.flush()
                    new_conversation = Conversation(
                        history_id=new_history.id,
                        messages=json.dumps(updated_messages),
                        sources=json.dumps(response["sources"]),
                        domain=domain
                    )
                    db.session.add(new_conversation)
                    response["history_id"] = new_history.id
                    current_app.logger.info(f"New history created with id {new_history.id} for user {user_id}")
                else:
                    history_to_use = (
                        db.session.query(History).filter_by(id=history_id).first()
                        if history_id
                        else latest_history
                    )
                    if not history_to_use:
                        return jsonify({"error": "Historique spécifié introuvable."}), 404

                    conversation = db.session.query(Conversation).filter_by(history_id=history_to_use.id).first()
                    if conversation:
                        existing_messages = json.loads(conversation.messages) if conversation.messages else []
                        existing_sources = json.loads(conversation.sources) if conversation.sources else []
                        existing_messages.extend([new_message, assistant_message])
                        conversation.messages = json.dumps(existing_messages)
                        conversation.sources = json.dumps(list(set(existing_sources + response["sources"])))
                        conversation.domain = domain
                    else:
                        new_conversation = Conversation(
                            history_id=history_to_use.id,
                            messages=json.dumps(updated_messages),
                            sources=json.dumps(response["sources"]),
                            domain=domain
                        )
                        db.session.add(new_conversation)
                    response["history_id"] = history_to_use.id

                db.session.commit()
                current_app.logger.info(f"Conversation saved for history_id {response['history_id']}")
            except Exception as e:
                db.session.rollback()
                current_app.logger.error(f"Failed to save conversation: {str(e)}")
                return jsonify({"error": "Erreur lors de la gestion de l'historique ou de la conversation."}), 500

        return jsonify(response), 200

    except Exception as e:
        current_app.logger.error(f"Internal server error: {str(e)}", exc_info=True)
        return jsonify({"error": "Une erreur interne est survenue."}), 500
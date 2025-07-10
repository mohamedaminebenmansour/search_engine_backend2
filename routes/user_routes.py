from flask import Blueprint, request, jsonify, current_app
from extensions import db
from models.user_model import User
from models.history_model import History, Conversation
from utils.auth_utils import token_required, hash_password, verify_password, generate_jwt
import json
from datetime import datetime

user_bp = Blueprint("user", __name__)

@user_bp.route("/register", methods=["POST", "OPTIONS"])
def register():
    if request.method == "OPTIONS":
        current_app.logger.debug("Received OPTIONS request for /register")
        return jsonify({}), 200

    try:
        current_app.logger.debug("Processing POST request for /register")
        data = request.get_json()
        current_app.logger.debug(f"Request JSON data: {data}")
        if not data:
            current_app.logger.warning("No JSON data in request")
            return jsonify({"error": "Requête JSON manquante"}), 400

        username = data.get("username")
        email = data.get("email")
        password = data.get("password")
        current_app.logger.debug(f"Extracted fields - username: {username}, email: {email}, password: [REDACTED]")

        if not all([username, email, password]):
            current_app.logger.warning("Missing required fields")
            return jsonify({"error": "Tous les champs (username, email, password) sont requis"}), 400

        existing_user = User.query.filter_by(email=email).first()
        current_app.logger.debug(f"Checked for existing user with email {email}: {existing_user}")
        if existing_user:
            current_app.logger.warning(f"Email {email} already in use")
            return jsonify({"error": "Cet email est déjà utilisé"}), 400

        password_hash = hash_password(password)
        user = User(username=username, email=email, password_hash=password_hash)
        current_app.logger.debug(f"Created new user object: {user}")
        db.session.add(user)
        db.session.commit()
        current_app.logger.info(f"User {username} registered successfully with id {user.id}")

        token = generate_jwt(user.id)
        current_app.logger.debug(f"Generated JWT token: {token}")
        return jsonify({"token": token, "user_id": user.id, "username": user.username}), 201

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error in /register: {str(e)}", exc_info=True)
        return jsonify({"error": "Une erreur interne est survenue."}), 500

@user_bp.route("/login", methods=["POST", "OPTIONS"])

@user_bp.route("/history", methods=["GET", "PUT", "DELETE", "OPTIONS"])
@token_required
def handle_history(current_user):
    if request.method == "OPTIONS":
        current_app.logger.debug("Received OPTIONS request for /history")
        return jsonify({}), 200

    try:
        current_app.logger.debug(f"Processing {request.method} request for /history for user_id {current_user.id}")

        if request.method == "GET":
            current_app.logger.debug("Fetching history for user")
            stmt = (
                db.session.query(History)
                .filter_by(user_id=current_user.id)
                .order_by(History.created_at.desc())
            )
            history = db.session.scalars(stmt).all()
            current_app.logger.debug(f"Fetched history entries: {len(history)} items")
            
            history_data = []
            for h in history:
                conversation = db.session.query(Conversation).filter_by(history_id=h.id).first()
                current_app.logger.debug(f"Processing history entry {h.id}, conversation found: {bool(conversation)}")
                history_data.append({
                    "id": h.id,
                    "search_query": h.search_query,
                    "conversation": {
                        "messages": json.loads(conversation.messages) if conversation else [],
                        "sources": json.loads(conversation.sources) if conversation and conversation.sources else []
                    },
                    "timestamp": h.created_at.isoformat()
                })
            current_app.logger.info(f"Returning history data with {len(history_data)} entries")
            return jsonify({"history": history_data}), 200

        elif request.method == "PUT":
            current_app.logger.debug("Updating history entry")
            data = request.get_json()
            current_app.logger.debug(f"Request JSON data: {data}")
            if not data:
                current_app.logger.warning("No JSON data in request")
                return jsonify({"error": "Requête JSON manquante"}), 400

            history_id = data.get("history_id")
            new_query = data.get("query")
            current_app.logger.debug(f"Extracted fields - history_id: {history_id}, new_query: {new_query}")

            if not all([history_id, new_query]):
                current_app.logger.warning("Missing required fields")
                return jsonify({"error": "Les champs history_id et query sont requis"}), 400

            history = db.session.query(History).filter_by(id=history_id, user_id=current_user.id).first()
            current_app.logger.debug(f"Queried history entry with id {history_id}: {history}")
            if not history:
                current_app.logger.warning(f"History entry {history_id} not found for user {current_user.id}")
                return jsonify({"error": "Historique non trouvé"}), 404

            history.search_query = new_query
            db.session.commit()
            current_app.logger.info(f"Updated history entry {history_id} with new query: {new_query}")
            return jsonify({"message": "Historique mis à jour avec succès"}), 200

        elif request.method == "DELETE":
            current_app.logger.debug("Deleting history entry")
            data = request.get_json()
            current_app.logger.debug(f"Request JSON data: {data}")
            if not data:
                current_app.logger.warning("No JSON data in request")
                return jsonify({"error": "Requête JSON manquante"}), 400

            history_id = data.get("history_id")
            current_app.logger.debug(f"Extracted field - history_id: {history_id}")
            if not history_id:
                current_app.logger.warning("Missing required field history_id")
                return jsonify({"error": "Le champ history_id est requis"}), 400

            history = db.session.query(History).filter_by(id=history_id, user_id=current_user.id).first()
            current_app.logger.debug(f"Queried history entry with id {history_id}: {history}")
            if not history:
                current_app.logger.warning(f"History entry {history_id} not found for user {current_user.id}")
                return jsonify({"error": "Historique non trouvé"}), 404

            db.session.delete(history)
            db.session.commit()
            current_app.logger.info(f"Deleted history entry {history_id}")
            return jsonify({"message": "Historique supprimé avec succès"}), 200

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error in /history: {str(e)}", exc_info=True)
        return jsonify({"error": "Une erreur interne est survenue."}), 500
@user_bp.route("/login", methods=["POST", "OPTIONS"])
def login():
    if request.method == "OPTIONS":
        current_app.logger.debug("Received OPTIONS request for /login")
        return jsonify({}), 200

    try:
        current_app.logger.debug("Processing POST request for /login")
        data = request.get_json()
        current_app.logger.debug(f"Request JSON data: {data}")
        if not data:
            current_app.logger.warning("No JSON data in request")
            return jsonify({"error": "Requête JSON manquante"}), 400

        email = data.get("email")
        password = data.get("password")
        current_app.logger.debug(f"Extracted fields - email: {email}, password: [REDACTED]")

        if not all([email, password]):
            current_app.logger.warning("Missing required fields")
            return jsonify({"error": "Les champs email et password sont requis"}), 400

        user = User.query.filter_by(email=email).first()
        current_app.logger.debug(f"Queried user with email {email}: {user}")
        if not user or not verify_password(password, user.password):  # Fixed typo here
            current_app.logger.warning(f"Invalid email or password for email {email}")
            return jsonify({"error": "Email ou mot de passe incorrect"}), 401

        token = generate_jwt(user.id)
        current_app.logger.debug(f"Generated JWT token for user {user.username}: {token}")
        current_app.logger.info(f"User {user.username} logged in successfully")
        return jsonify({"token": token, "user_id": user.id, "username": user.username}), 200

    except Exception as e:
        current_app.logger.error(f"Error in /login: {str(e)}", exc_info=True)
        return jsonify({"error": "Une erreur interne est survenue."}), 500

# ... (history route unchanged)
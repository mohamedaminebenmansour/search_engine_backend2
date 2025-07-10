from flask import Blueprint, request, jsonify, current_app
from services.search_service import hybrid_search
from models.history_model import History
from models.user_model import User
from extensions import db
import jwt
import traceback

search_bp = Blueprint("search", __name__)

@search_bp.route('/search', methods=['POST', 'OPTIONS'], strict_slashes=False)
def search():
    current_app.logger.info("=== Début de la requête /api/search ===")
    current_app.logger.debug(f"Headers: {dict(request.headers)}")
    current_app.logger.debug(f"Données brutes: {request.get_data()}")

    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        if not request.is_json:
            current_app.logger.warning("Content-Type incorrect: %s", request.content_type)

        data = request.get_json()
        if not data:
            return jsonify({"error": "Données JSON manquantes ou Content-Type incorrect."}), 400

        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "Le champ 'query' est vide ou manquant."}), 400

        current_app.logger.info(f"Requête utilisateur : '{query}'")

        current_user = None
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            try:
                token = auth_header.split(" ")[1]
                payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=["HS256"])
                current_user = User.query.get(payload['user_id'])
            except (jwt.ExpiredSignatureError, jwt.InvalidTokenError) as e:
                current_app.logger.warning("JWT invalide : %s", str(e))
                pass

        if current_user:
            try:
                history_entry = History(user_id=current_user.id, query=query)
                db.session.add(history_entry)
                db.session.commit()
                current_app.logger.info("Requête enregistrée dans l'historique")
            except Exception as db_error:
                db.session.rollback()
                current_app.logger.error("Erreur DB : %s", str(db_error))
                current_app.logger.error(traceback.format_exc())

        try:
            result = hybrid_search(query, user_id=current_user.id if current_user else None)
            final_result = {
                "answer": result["answer"],
                "sources": result.get("sources", [])
            }
        except Exception as search_error:
            current_app.logger.error("Erreur dans hybrid_search: %s", str(search_error))
            current_app.logger.error(traceback.format_exc())
            return jsonify({"error": "Erreur pendant la recherche."}), 500

        return jsonify(final_result)

    except Exception as e:
        current_app.logger.error("Erreur interne : %s", str(e))
        current_app.logger.error(traceback.format_exc())
        return jsonify({"error": "Une erreur interne du serveur est survenue lors de la recherche."}), 500

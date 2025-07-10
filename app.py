from flask import Flask, request, jsonify
from flask_cors import CORS
from extensions import db, bcrypt
from routes.search_routes import search_bp
from routes.auth_routes import auth_bp
from routes.user_routes import user_bp
from routes.scraping_routes import scraping_bp
from routes.chat_routes import chat_bp
from flask_migrate import Migrate  # Import Migrate


app = Flask(__name__)

# Add this after creating the Flask app
from flask import make_response

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "http://localhost:8501")
        response.headers.add("Access-Control-Allow-Headers", "Authorization, Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        response.headers.add("Access-Control-Allow-Credentials", "true")
        return response

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:8501'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Headers'] = 'Authorization, Content-Type'
    return response
CORS(app, resources={
    r"/api/*": {
        "origins": "http://localhost:8501",
        "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
        "allow_headers": ["Authorization", "Content-Type"],
        "supports_credentials": True,
        "expose_headers": ["Authorization"]
    }
})

# Config
app.config['LOG_LEVEL'] = 'DEBUG'  # En développement
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'votre_clef_secrete_super_secure'

# Init les extensions
db.init_app(app)
bcrypt.init_app(app)

# Initialize Migrate
migrate = Migrate(app, db)  # Initialize Migrate

# les routes
app.register_blueprint(search_bp, url_prefix="/api")
app.register_blueprint(auth_bp, url_prefix="/api/auth")
app.register_blueprint(user_bp, url_prefix="/api")
app.register_blueprint(scraping_bp, url_prefix="/api")
app.register_blueprint(chat_bp, url_prefix="/api")


with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=5000)
  
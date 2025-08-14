from deployment.flask_api import app

if __name__ == "__main__":
    # This script runs the Flask application.
    # debug=True allows for auto-reloading when you save changes.
    # Set debug=False for a production environment.
    app.run(debug=True, port=5001)
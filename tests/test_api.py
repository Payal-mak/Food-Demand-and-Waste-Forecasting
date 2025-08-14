# tests/test_api.py
import pytest
import os
import sys

# --- The Fix: Add project root to the Python path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
# ----------------------------------------------------

from deployment.flask_api import app

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_api_success(client):
    """Test the /predict endpoint for a successful prediction."""
    test_data = {
        "features": [
            [101, 12.1, 0, 28.5, 105.1, 6.1], [120, 14.5, 0, 29.1, 105.3, 6.2],
            [115, 13.8, 0, 30.2, 105.7, 6.3], [130, 15.6, 0, 31.0, 106.0, 6.4],
            [110, 13.2, 1, 25.5, 106.2, 6.5], [98, 11.8, 0, 29.5, 106.5, 6.6],
            [125, 15.0, 0, 30.0, 106.8, 6.7], [140, 16.8, 0, 31.5, 107.1, 6.8],
            [135, 16.2, 0, 32.0, 107.3, 6.9], [122, 14.6, 0, 30.5, 107.6, 7.0]
        ]
    }
    response = client.post('/predict', json=test_data)
    # Check for a successful status code
    assert response.status_code == 200
    # Check that the response contains the expected keys
    assert 'predicted_demand' in response.json
    assert 'predicted_waste' in response.json
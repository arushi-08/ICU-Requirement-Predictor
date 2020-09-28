import json
from main import app

def test_api():
    with open('testdata.json') as f:
        test_data = json.load(f)
    with app.test_client() as client:
        for test_case in test_data:
            features = test_case['features']
            expected_response = test_case['expected_response']
            expected_status_code = test_case['expected_status_code']
            # Test client uses "query_string" instead of "params"
            response = client.get('/predict_', query_string=features)
            # Check that we got the correct status code back.
            assert response.status_code == expected_status_code
            # response.data returns a byte array, convert to a dict.
            assert json.loads(response.data) == expected_response

import requests
import json

def test_health_endpoint(base_url):
    """Test the health endpoint"""
    try:
        response = requests.get(f"{base_url}/health")
        print("Health Check:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        print("-" * 50)
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_calculate_ats_only(base_url):
    """Test the ATS calculation endpoint with text input"""
    try:
        data = {
            "job_description": "We are looking for a Python developer with experience in Django, Flask, and REST APIs. Knowledge of JavaScript and SQL is preferred.",
            "resume_text": "Experienced Python developer with 3 years of experience in Django and Flask. Built multiple web applications and REST APIs. Skills include Python, JavaScript, SQL, Git, and Docker."
        }
        
        response = requests.post(
            f"{base_url}/calculate-ats-only",
            headers={'Content-Type': 'application/json'},
            data=json.dumps(data)
        )
        
        print("ATS Calculation Test:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        print("-" * 50)
        return response.status_code == 200
    except Exception as e:
        print(f"ATS calculation test failed: {e}")
        return False

def test_analyze_resume_endpoint(base_url):
    """Test the analyze resume endpoint"""
    # Note: This requires a PDF file to test properly
    print("Analyze Resume Endpoint:")
    print("This endpoint requires a PDF file upload.")
    print("You can test it manually using curl or Postman:")
    print(f"""
    curl -X POST {base_url}/analyze-resume \\
      -F "job_description=Python developer with Django experience" \\
      -F "resume_file=@path/to/your/resume.pdf"
    """)
    print("-" * 50)

def main():
    # Test with local Flask server
    base_url = "http://localhost:5000"
    
    print("Testing ATS Scoring API")
    print("=" * 50)
    
    # Test endpoints
    health_ok = test_health_endpoint(base_url)
    if health_ok:
        test_calculate_ats_only(base_url)
    
    test_analyze_resume_endpoint(base_url)
    
    print("\nTo test with your deployed API, change the base_url to your deployed URL")

if __name__ == "__main__":
    main()

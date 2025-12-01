import requests

BASE_URL = "https://your-koyeb-url.koyeb.app"  # Replace with your actual URL

# Test health first
print("1. Testing health endpoint...")
health_response = requests.get(f"{BASE_URL}/health")
print(f"Status: {health_response.status_code}")
print(f"Response: {health_response.text}")
print()

# Upload PDF
print("2. Uploading PDF...")
with open("test.pdf", "rb") as f:
    upload_response = requests.post(
        f"{BASE_URL}/upload",
        files={"file": f}
    )
print(f"Status: {upload_response.status_code}")
print(f"Response: {upload_response.json()}")
print()

# Extract text
print("3. Extracting text...")
extract_response = requests.post(
    f"{BASE_URL}/extract/test.pdf",
    data={
        "page_number": 1,
        "format": "json",
        "dpi": 50
    }
)
print(f"Status: {extract_response.status_code}")
print(f"Headers: {extract_response.headers}")
print(f"Raw Response: {extract_response.text[:500]}")  # First 500 chars

# Try to parse JSON
try:
    result = extract_response.json()
    print(f"JSON Response: {result}")
except Exception as e:
    print(f"ERROR: Could not parse JSON: {e}")
    print(f"Full response text: {extract_response.text}")

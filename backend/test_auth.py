import requests
import json

def test_auth_endpoints():
    base_url = 'http://localhost:8000/auth'
    
    # Test registration
    print("Testing registration...")
    register_data = {
        'username': 'testuser123',
        'password': 'testpass123',
        'email': 'test@example.com'
    }
    
    try:
        response = requests.post(
            f'{base_url}/register/',
            headers={'Content-Type': 'application/json'},
            data=json.dumps(register_data)
        )
        print(f"Registration Status: {response.status_code}")
        print(f"Registration Response: {response.json()}")
        
        if response.status_code == 201:
            print("✅ Registration successful!")
            
            # Test login with the same credentials
            print("\nTesting login...")
            login_data = {
                'username': 'testuser123',
                'password': 'testpass123'
            }
            
            login_response = requests.post(
                f'{base_url}/login/',
                headers={'Content-Type': 'application/json'},
                data=json.dumps(login_data)
            )
            print(f"Login Status: {login_response.status_code}")
            print(f"Login Response: {login_response.json()}")
            
            if login_response.status_code == 200:
                print("✅ Login successful!")
            else:
                print("❌ Login failed!")
        else:
            print("❌ Registration failed!")
            
    except Exception as e:
        print(f"Test error: {e}")

if __name__ == "__main__":
    test_auth_endpoints()
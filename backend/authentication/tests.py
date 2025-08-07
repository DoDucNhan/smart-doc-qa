"""
Unit Tests for Authentication System

WHY WE TEST AUTHENTICATION:
- Authentication is critical for security
- User registration/login must work reliably
- Tokens must be generated and validated correctly
- Error cases must be handled properly
"""

from django.test import TestCase
from django.contrib.auth.models import User
from django.urls import reverse
from rest_framework.test import APITestCase
from rest_framework import status
from rest_framework.authtoken.models import Token
import json


class UserModelTestCase(TestCase):
    """
    Test the Django User model functionality
    
    WHY THIS TEST IS IMPORTANT:
    - Ensures user creation works correctly
    - Validates password hashing and verification
    - Tests user model constraints and validation
    """
    
    def setUp(self):
        """
        Set up test data before each test method runs
        
        WHY WE USE setUp:
        - Creates consistent test environment
        - Avoids code duplication across tests
        - Ensures each test starts with clean state
        """
        self.test_username = "testuser"
        self.test_email = "test@example.com"
        self.test_password = "securepassword123"
    
    def test_user_creation(self):
        """
        Test that users can be created successfully
        
        WHAT THIS TESTS:
        - User.objects.create_user() works correctly
        - User fields are set properly
        - User is saved to database
        
        WHY THIS IS IMPORTANT:
        - User creation is fundamental to the system
        - Ensures database constraints work
        - Validates model field behavior
        """
        user = User.objects.create_user(
            username=self.test_username,
            email=self.test_email,
            password=self.test_password
        )
        
        # Verify user was created
        self.assertEqual(user.username, self.test_username)
        self.assertEqual(user.email, self.test_email)
        self.assertTrue(user.is_active)  # Users should be active by default
        self.assertFalse(user.is_staff)  # Regular users shouldn't be staff
        
        # Verify user exists in database
        self.assertTrue(User.objects.filter(username=self.test_username).exists())
    
    def test_password_hashing(self):
        """
        Test that passwords are properly hashed and not stored in plain text
        
        WHAT THIS TESTS:
        - Passwords are hashed using Django's password system
        - Plain text passwords are not stored
        - Password verification works correctly
        
        WHY THIS IS CRITICAL FOR SECURITY:
        - Plain text passwords are a major security risk
        - Hashed passwords protect users if database is compromised
        - Proper password verification prevents unauthorized access
        """
        user = User.objects.create_user(
            username=self.test_username,
            password=self.test_password
        )
        
        # Password should NOT be stored as plain text
        self.assertNotEqual(user.password, self.test_password)
        
        # Password should be hashed (Django uses pbkdf2_sha256 by default)
        self.assertTrue(user.password.startswith('pbkdf2_sha256$'))
        
        # check_password should verify the original password
        self.assertTrue(user.check_password(self.test_password))
        
        # Wrong password should fail verification
        self.assertFalse(user.check_password("wrongpassword"))
    
    def test_username_uniqueness(self):
        """
        Test that usernames must be unique
        
        WHAT THIS TESTS:
        - Database enforces username uniqueness
        - Duplicate usernames raise proper exceptions
        - Database integrity is maintained
        
        WHY THIS IS IMPORTANT:
        - Prevents user identity conflicts
        - Ensures reliable user lookup
        - Maintains data integrity
        """
        # Create first user
        User.objects.create_user(
            username=self.test_username,
            password=self.test_password
        )
        
        # Attempting to create user with same username should fail
        with self.assertRaises(Exception):  # Django raises IntegrityError
            User.objects.create_user(
                username=self.test_username,  # Same username
                password="differentpassword"
            )


class TokenModelTestCase(TestCase):
    """
    Test the Token authentication model
    
    WHY WE TEST TOKENS:
    - Tokens are used for API authentication
    - Token generation must be secure and unique
    - Token-user relationships must work correctly
    """
    
    def setUp(self):
        """Set up test user for token tests"""
        self.user = User.objects.create_user(
            username="tokenuser",
            password="testpass123"
        )
    
    def test_token_creation(self):
        """
        Test that tokens can be created for users
        
        WHAT THIS TESTS:
        - Token.objects.create() works correctly
        - Token is associated with correct user
        - Token has proper format and length
        
        WHY THIS IS IMPORTANT:
        - Tokens enable stateless authentication
        - Each user needs a unique token
        - Token format must be consistent for API clients
        """
        token = Token.objects.create(user=self.user)
        
        # Verify token is associated with user
        self.assertEqual(token.user, self.user)
        
        # Verify token has proper format (40 character hex string)
        self.assertEqual(len(token.key), 40)
        self.assertTrue(all(c in '0123456789abcdef' for c in token.key))
    
    def test_token_uniqueness(self):
        """
        Test that each user gets a unique token
        
        WHAT THIS TESTS:
        - Tokens are unique across all users
        - Each user can have only one token
        - Token generation is properly random
        
        WHY THIS IS IMPORTANT:
        - Prevents token collisions and security issues
        - Ensures each user has unique authentication
        - Validates token generation algorithm
        """
        user2 = User.objects.create_user(username="user2", password="pass123")
        
        token1 = Token.objects.create(user=self.user)
        token2 = Token.objects.create(user=user2)
        
        # Tokens should be different
        self.assertNotEqual(token1.key, token2.key)
        
        # Each user should have their own token
        self.assertEqual(token1.user, self.user)
        self.assertEqual(token2.user, user2)
    
    def test_get_or_create_token(self):
        """
        Test get_or_create functionality for tokens
        
        WHAT THIS TESTS:
        - get_or_create returns existing token if present
        - get_or_create creates new token if none exists
        - No duplicate tokens are created for same user
        
        WHY THIS IS IMPORTANT:
        - Prevents multiple tokens per user
        - Ensures consistent token for each user
        - Optimizes database usage
        """
        # First call should create token
        token1, created1 = Token.objects.get_or_create(user=self.user)
        self.assertTrue(created1)  # Should be newly created
        
        # Second call should return existing token
        token2, created2 = Token.objects.get_or_create(user=self.user)
        self.assertFalse(created2)  # Should not create new token
        self.assertEqual(token1.key, token2.key)  # Same token


class AuthenticationAPITestCase(APITestCase):
    """
    Test the authentication API endpoints
    
    WHY WE TEST API ENDPOINTS:
    - APIs are the interface between frontend and backend
    - HTTP status codes must be correct
    - Request/response formats must be consistent
    - Error handling must be robust
    """
    
    def setUp(self):
        """Set up test data for API tests"""
        self.register_url = reverse('register')  # /auth/register/
        self.login_url = reverse('login')        # /auth/login/
        
        self.valid_user_data = {
            'username': 'apitest',
            'password': 'testpass123',
            'email': 'api@test.com'
        }
        
        self.invalid_user_data = {
            'username': '',  # Invalid: empty username
            'password': '123',  # Invalid: too short
            'email': 'invalid-email'  # Invalid: bad format
        }
    
    def test_user_registration_success(self):
        """
        Test successful user registration via API
        
        WHAT THIS TESTS:
        - POST to /auth/register/ works correctly
        - Returns 201 status code for successful creation
        - Response includes required fields (token, user_id)
        - User is actually created in database
        
        WHY THIS IS IMPORTANT:
        - Registration is the entry point for new users
        - Frontend needs consistent response format
        - Status codes help frontend handle responses
        - Database consistency must be maintained
        """
        response = self.client.post(
            self.register_url, 
            self.valid_user_data, 
            format='json'
        )
        
        # Check HTTP status code
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        
        # Check response contains required fields
        response_data = response.json()
        self.assertIn('token', response_data)
        self.assertIn('user_id', response_data)
        self.assertIn('username', response_data)
        
        # Verify token format
        token = response_data['token']
        self.assertEqual(len(token), 40)
        
        # Verify user was created in database
        user_exists = User.objects.filter(
            username=self.valid_user_data['username']
        ).exists()
        self.assertTrue(user_exists)
        
        # Verify token was created and associated with user
        user = User.objects.get(username=self.valid_user_data['username'])
        db_token = Token.objects.get(user=user)
        self.assertEqual(db_token.key, token)
    
    def test_user_registration_duplicate_username(self):
        """
        Test registration with duplicate username
        
        WHAT THIS TESTS:
        - Duplicate usernames are rejected
        - Returns 400 status code for bad request
        - Error message is clear and helpful
        - Database integrity is maintained
        
        WHY THIS IS IMPORTANT:
        - Prevents user conflicts and confusion
        - Provides clear feedback to users
        - Maintains data integrity
        - Helps frontend show appropriate error messages
        """
        # Create first user
        self.client.post(self.register_url, self.valid_user_data, format='json')
        
        # Try to create second user with same username
        response = self.client.post(
            self.register_url, 
            self.valid_user_data,  # Same data
            format='json'
        )
        
        # Should return 400 Bad Request
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        
        # Should contain error message
        response_data = response.json()
        self.assertIn('error', response_data)
        self.assertIn('already exists', response_data['error'].lower())
    
    def test_user_registration_missing_fields(self):
        """
        Test registration with missing required fields
        
        WHAT THIS TESTS:
        - Missing username/password are handled correctly
        - Returns 400 status code
        - Clear error messages are provided
        - No partial user creation occurs
        
        WHY THIS IS IMPORTANT:
        - Validates input before database operations
        - Prevents incomplete user records
        - Provides helpful feedback for form validation
        - Maintains data quality
        """
        incomplete_data = {'username': 'testuser'}  # Missing password
        
        response = self.client.post(
            self.register_url, 
            incomplete_data, 
            format='json'
        )
        
        # Should return 400 Bad Request
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        
        # Should contain error message
        response_data = response.json()
        self.assertIn('error', response_data)
        
        # User should not be created
        user_exists = User.objects.filter(username='testuser').exists()
        self.assertFalse(user_exists)
    
    def test_user_login_success(self):
        """
        Test successful user login via API
        
        WHAT THIS TESTS:
        - Users can login with correct credentials
        - Returns 200 status code
        - Response includes authentication token
        - Token matches the one from registration
        
        WHY THIS IS IMPORTANT:
        - Login enables access to protected features
        - Token consistency across registration/login
        - Proper authentication flow
        - Frontend can store and use token
        """
        # First register a user
        register_response = self.client.post(
            self.register_url, 
            self.valid_user_data, 
            format='json'
        )
        register_token = register_response.json()['token']
        
        # Then login with same credentials
        login_data = {
            'username': self.valid_user_data['username'],
            'password': self.valid_user_data['password']
        }
        
        login_response = self.client.post(
            self.login_url, 
            login_data, 
            format='json'
        )
        
        # Check response
        self.assertEqual(login_response.status_code, status.HTTP_200_OK)
        
        response_data = login_response.json()
        self.assertIn('token', response_data)
        self.assertIn('user_id', response_data)
        
        # Token should be the same as from registration
        login_token = response_data['token']
        self.assertEqual(login_token, register_token)
    
    def test_user_login_invalid_credentials(self):
        """
        Test login with invalid credentials
        
        WHAT THIS TESTS:
        - Wrong username/password combinations are rejected
        - Returns 401 Unauthorized status
        - No token is provided for invalid login
        - Security: prevents unauthorized access
        
        WHY THIS IS IMPORTANT:
        - Prevents unauthorized access to accounts
        - Provides appropriate HTTP status codes
        - Security against brute force attacks
        - Clear feedback for authentication failures
        """
        # Register a user first
        self.client.post(self.register_url, self.valid_user_data, format='json')
        
        # Try login with wrong password
        invalid_login = {
            'username': self.valid_user_data['username'],
            'password': 'wrongpassword'
        }
        
        response = self.client.post(
            self.login_url, 
            invalid_login, 
            format='json'
        )
        
        # Should return 401 Unauthorized
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
        
        # Should contain error message
        response_data = response.json()
        self.assertIn('error', response_data)
        self.assertIn('invalid', response_data['error'].lower())
        
        # Should not contain token
        self.assertNotIn('token', response_data)
    
    def test_user_login_nonexistent_user(self):
        """
        Test login with non-existent username
        
        WHAT THIS TESTS:
        - Non-existent usernames are handled correctly
        - Returns 401 status (not 404, for security)
        - Error message doesn't reveal user existence
        - Prevents user enumeration attacks
        
        WHY THIS IS IMPORTANT:
        - Security: prevents username enumeration
        - Consistent error responses for all auth failures
        - Doesn't leak information about valid usernames
        - Standard security practice
        """
        nonexistent_login = {
            'username': 'nonexistentuser',
            'password': 'anypassword'
        }
        
        response = self.client.post(
            self.login_url, 
            nonexistent_login, 
            format='json'
        )
        
        # Should return 401 (not 404, for security)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
        
        # Error message should be generic (don't reveal user doesn't exist)
        response_data = response.json()
        self.assertIn('error', response_data)
        # Message should be same as for wrong password
        self.assertIn('invalid', response_data['error'].lower())


class TokenAuthenticationTestCase(APITestCase):
    """
    Test token-based authentication for protected endpoints
    
    WHY WE TEST TOKEN AUTHENTICATION:
    - Tokens protect API endpoints from unauthorized access
    - Frontend needs to use tokens correctly
    - Token validation must be secure and reliable
    - Error handling for invalid/missing tokens
    """
    
    def setUp(self):
        """Set up authenticated user for token tests"""
        self.user = User.objects.create_user(
            username='tokentest',
            password='testpass123'
        )
        self.token = Token.objects.create(user=self.user)
        self.documents_url = '/api/documents/'  # Protected endpoint
    
    def test_access_protected_endpoint_with_valid_token(self):
        """
        Test accessing protected endpoint with valid token
        
        WHAT THIS TESTS:
        - Valid tokens allow access to protected endpoints
        - Authorization header format is correct
        - Protected endpoints return expected data
        - Token validation works properly
        
        WHY THIS IS IMPORTANT:
        - Ensures authenticated users can access their data
        - Validates token authentication implementation
        - Tests the primary authentication flow
        - Ensures API security works as designed
        """
        # Set authorization header
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {self.token.key}')
        
        response = self.client.get(self.documents_url)
        
        # Should allow access
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Response should be JSON (documents list)
        self.assertEqual(response['content-type'], 'application/json')
    
    def test_access_protected_endpoint_without_token(self):
        """
        Test accessing protected endpoint without authentication
        
        WHAT THIS TESTS:
        - Unauthenticated requests are rejected
        - Returns 401 Unauthorized status
        - Protected endpoints require authentication
        - Security: prevents unauthorized data access
        
        WHY THIS IS IMPORTANT:
        - Ensures API security is enforced
        - Prevents unauthorized access to user data
        - Validates authentication middleware
        - Standard security requirement
        """
        # Don't set any authorization header
        response = self.client.get(self.documents_url)
        
        # Should deny access
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
    
    def test_access_protected_endpoint_with_invalid_token(self):
        """
        Test accessing protected endpoint with invalid token
        
        WHAT THIS TESTS:
        - Invalid tokens are rejected
        - Returns 401 Unauthorized status
        - Token validation catches fake/malformed tokens
        - Security against token manipulation
        
        WHY THIS IS IMPORTANT:
        - Prevents access with fake tokens
        - Validates token format and existence
        - Security against token manipulation attacks
        - Ensures only valid tokens work
        """
        # Use fake token
        fake_token = 'fakeinvalidtoken1234567890'
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {fake_token}')
        
        response = self.client.get(self.documents_url)
        
        # Should deny access
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
    
    def test_access_protected_endpoint_with_malformed_header(self):
        """
        Test accessing protected endpoint with malformed auth header
        
        WHAT THIS TESTS:
        - Malformed Authorization headers are rejected
        - Header format validation works
        - Returns appropriate error status
        - Handles edge cases in token format
        
        WHY THIS IS IMPORTANT:
        - Robust handling of malformed requests
        - Security against header manipulation
        - Clear error responses for client debugging
        - Prevents server errors from bad input
        """
        # Use malformed authorization header
        self.client.credentials(HTTP_AUTHORIZATION='Bearer invalidformat')
        
        response = self.client.get(self.documents_url)
        
        # Should deny access
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


class AuthenticationIntegrationTestCase(APITestCase):
    """
    Integration tests for complete authentication workflow
    
    WHY WE TEST INTEGRATION:
    - Tests complete user journey from registration to API usage
    - Validates that all components work together
    - Catches issues that unit tests might miss
    - Ensures realistic user scenarios work
    """
    
    def test_complete_authentication_workflow(self):
        """
        Test complete workflow: register → login → access protected endpoint
        
        WHAT THIS TESTS:
        - End-to-end authentication flow
        - All components work together
        - Realistic user interaction pattern
        - Data consistency across operations
        
        WHY THIS IS IMPORTANT:
        - Simulates real user behavior
        - Catches integration issues
        - Validates complete system functionality
        - Ensures smooth user experience
        """
        # Step 1: Register new user
        register_data = {
            'username': 'integrationtest',
            'password': 'testpass123',
            'email': 'integration@test.com'
        }
        
        register_response = self.client.post(
            '/auth/register/', 
            register_data, 
            format='json'
        )
        
        self.assertEqual(register_response.status_code, status.HTTP_201_CREATED)
        register_token = register_response.json()['token']
        
        # Step 2: Login with same credentials
        login_data = {
            'username': register_data['username'],
            'password': register_data['password']
        }
        
        login_response = self.client.post(
            '/auth/login/', 
            login_data, 
            format='json'
        )
        
        self.assertEqual(login_response.status_code, status.HTTP_200_OK)
        login_token = login_response.json()['token']
        
        # Tokens should match
        self.assertEqual(register_token, login_token)
        
        # Step 3: Use token to access protected endpoint
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {login_token}')
        
        protected_response = self.client.get('/api/documents/')
        self.assertEqual(protected_response.status_code, status.HTTP_200_OK)
        
        # Should return empty list for new user
        documents = protected_response.json()
        self.assertEqual(documents, [])
    
    def test_token_persistence_across_requests(self):
        """
        Test that tokens work consistently across multiple requests
        
        WHAT THIS TESTS:
        - Token doesn't expire or change between requests
        - Multiple API calls with same token work
        - Session consistency is maintained
        - Token state is properly managed
        
        WHY THIS IS IMPORTANT:
        - Users make multiple API calls per session
        - Token stability is crucial for user experience
        - Validates token storage and retrieval
        - Ensures stateless authentication works
        """
        # Register and get token
        user_data = {
            'username': 'persisttest',
            'password': 'testpass123',
            'email': 'persist@test.com'
        }
        
        response = self.client.post('/auth/register/', user_data, format='json')
        token = response.json()['token']
        
        # Set credentials
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {token}')
        
        # Make multiple requests with same token
        for i in range(5):
            response = self.client.get('/api/documents/')
            self.assertEqual(response.status_code, status.HTTP_200_OK)
            
        # Token should still work after multiple uses
        final_response = self.client.get('/api/documents/')
        self.assertEqual(final_response.status_code, status.HTTP_200_OK)
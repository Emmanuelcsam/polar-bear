#!/usr/bin/env python3
"""
Test script for Habitica and Gemini API connections
"""

import os
import sys
import json
import requests
from pathlib import Path
import importlib

# Try to import optional dependencies
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: python-dotenv not installed, reading .env manually")
    # Manual .env loading
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Note: google-generativeai not installed")

def test_habitica_api():
    """Test Habitica API connection"""
    print("\n🎮 Testing Habitica API...")
    print("-" * 50)
    
    # Get credentials
    user_id = os.environ.get('HABITICA_USER_ID')
    api_key = os.environ.get('HABITICA_API_KEY')
    
    if not user_id or not api_key:
        print("❌ Habitica credentials not found in .env file")
        return False
    
    print(f"✓ Found credentials")
    print(f"  User ID: {user_id[:8]}...")
    print(f"  API Key: {api_key[:8]}...")
    
    # Test API connection
    headers = {
        'x-api-user': user_id,
        'x-api-key': api_key,
        'x-client': '3a326108-1895-4c23-874e-37668c75f2ad-ProjectTracker',
        'Content-Type': 'application/json'
    }
    
    try:
        # Test user endpoint
        print("\n📡 Testing user endpoint...")
        response = requests.get('https://habitica.com/api/v3/user', headers=headers)
        response.raise_for_status()
        
        user_data = response.json().get('data', {})
        print(f"✅ Connection successful!")
        print(f"   Username: {user_data.get('profile', {}).get('name', 'Unknown')}")
        print(f"   Level: {user_data.get('stats', {}).get('lvl', 0)}")
        print(f"   HP: {user_data.get('stats', {}).get('hp', 0):.0f}/{user_data.get('stats', {}).get('maxHealth', 0)}")
        print(f"   Gold: {user_data.get('stats', {}).get('gp', 0):.0f}")
        
        # Test tasks endpoint
        print("\n📝 Testing tasks endpoint...")
        response = requests.get('https://habitica.com/api/v3/tasks/user', headers=headers)
        response.raise_for_status()
        
        tasks = response.json().get('data', [])
        project_tasks = [t for t in tasks if 'project-tracker' in t.get('tags', [])]
        print(f"✅ Found {len(tasks)} total tasks")
        print(f"   Project tracker tasks: {len(project_tasks)}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"   Status: {e.response.status_code}")
            print(f"   Response: {e.response.text[:200]}...")
        return False

def test_gemini_api():
    """Test Google Gemini API connection"""
    print("\n🤖 Testing Google Gemini API...")
    print("-" * 50)
    
    if not GENAI_AVAILABLE:
        print("⚠️  google-generativeai module not available")
        print("   Testing with direct API calls instead...")
        return test_gemini_api_direct()
    
    # Get API key
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    
    if not api_key:
        print("❌ Gemini API key not found in .env file")
        return False
    
    print(f"✓ Found API key: {api_key[:10]}...")
    
    try:
        # Configure Gemini
        print("\n🔧 Configuring Gemini...")
        genai.configure(api_key=api_key)
        
        # List available models
        print("\n📋 Available models:")
        models = genai.list_models()
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                print(f"   - {model.name}")
        
        # Test with gemini-pro
        print("\n🧪 Testing gemini-pro model...")
        model = genai.GenerativeModel('gemini-pro')
        
        # Simple test prompt
        prompt = "Say 'Hello, World!' and confirm the API is working."
        response = model.generate_content(prompt)
        
        print(f"✅ API connection successful!")
        print(f"   Response: {response.text[:100]}...")
        
        # Test token counting
        print("\n📊 Testing token counting...")
        count_response = model.count_tokens(prompt)
        print(f"   Prompt tokens: {count_response.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        return False

def test_gemini_api_direct():
    """Test Gemini API with direct HTTP calls"""
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    
    if not api_key:
        print("❌ Gemini API key not found")
        return False
    
    print(f"✓ Found API key: {api_key[:10]}...")
    
    try:
        # Test with direct API call
        print("\n🧪 Testing gemini-1.5-flash model via REST API...")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        
        data = {
            "contents": [{
                "parts": [{
                    "text": "Say 'Hello, World!' and confirm the API is working."
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 100
            }
        }
        
        response = requests.post(url, json=data, headers={'Content-Type': 'application/json'})
        response.raise_for_status()
        
        result = response.json()
        if 'candidates' in result and result['candidates']:
            text = result['candidates'][0]['content']['parts'][0]['text']
            print(f"✅ API connection successful!")
            print(f"   Response: {text[:100]}...")
            return True
        else:
            print(f"❌ Unexpected response format")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"   Response: {e.response.text[:200]}...")
        return False

def test_script_integration():
    """Test the actual script integrations"""
    print("\n🔍 Testing script integrations...")
    print("-" * 50)
    
    # Test habitica-integration.py
    print("\n📄 Testing habitica-integration.py...")
    try:
        # Import using the file name with hyphens replaced
        sys.path.insert(0, '.')
        import importlib
        habitica_module = importlib.import_module('habitica-integration')
        HabiticaProjectGamification = habitica_module.HabiticaProjectGamification
        
        hab = HabiticaProjectGamification()
        user_data = hab._api_request('GET', '/user')
        if user_data:
            print("✅ habitica-integration.py working")
        else:
            print("❌ habitica-integration.py failed to connect")
    except Exception as e:
        print(f"❌ Error in habitica-integration.py: {e}")
    
    # Test ai-project-advisor.py
    print("\n📄 Testing ai-project-advisor.py...")
    try:
        # Import using the file name with hyphens replaced
        ai_module = importlib.import_module('ai-project-advisor')
        AIProjectAdvisor = ai_module.AIProjectAdvisor
        
        advisor = AIProjectAdvisor()
        # Just test that it imports and initializes
        print("✅ ai-project-advisor.py imports successfully")
        
        # Test Gemini configuration
        advisor.provider = 'gemini'
        advisor.api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        advisor.model = 'gemini-pro'
        
        if advisor.api_key:
            test_prompt = "Test connection"
            headers = advisor._gemini_headers()
            request_data = advisor._gemini_request(test_prompt)
            print("✅ Gemini configuration in ai-project-advisor.py looks correct")
        
    except Exception as e:
        print(f"❌ Error in ai-project-advisor.py: {e}")

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n📦 Checking dependencies...")
    print("-" * 50)
    
    dependencies = {
        'requests': ('requests', True),  # (package_name, required)
        'dotenv': ('python-dotenv', False),
        'google.generativeai': ('google-generativeai', False)
    }
    
    missing_required = []
    missing_optional = []
    
    for module, (package, required) in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} NOT installed {'(required)' if required else '(optional)'}")
            if required:
                missing_required.append(package)
            else:
                missing_optional.append(package)
    
    if missing_required:
        print(f"\n❌ Missing required dependencies: {', '.join(missing_required)}")
        print(f"   Run: pip install {' '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"\n⚠️  Missing optional dependencies: {', '.join(missing_optional)}")
        print("   APIs will be tested using direct HTTP calls")
    
    return True

def main():
    """Run all tests"""
    print("🧪 API INTEGRATION TEST SUITE")
    print("=" * 70)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return
    
    # Check .env file
    env_file = Path('.env')
    if not env_file.exists():
        print("\n❌ .env file not found!")
        print("   Please create a .env file with your API keys")
        return
    
    # Run tests
    habitica_ok = test_habitica_api()
    gemini_ok = test_gemini_api()
    
    # Test script integrations
    test_script_integration()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    print(f"Habitica API: {'✅ PASSED' if habitica_ok else '❌ FAILED'}")
    print(f"Gemini API: {'✅ PASSED' if gemini_ok else '❌ FAILED'}")
    
    if habitica_ok and gemini_ok:
        print("\n✨ All API connections working!")
    else:
        print("\n⚠️  Some API connections failed. Check the errors above.")

if __name__ == "__main__":
    main()
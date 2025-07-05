#!/usr/bin/env python3
"""
Test script functionality for Habitica and AI Advisor scripts
"""

import os
import sys
import json
from pathlib import Path
import subprocess
import time
import importlib

def test_habitica_functionality():
    """Test Habitica integration functionality"""
    print("\n🎮 Testing Habitica Integration Functionality...")
    print("-" * 70)
    
    try:
        # Import the module
        sys.path.insert(0, '.')
        import importlib
        habitica_module = importlib.import_module('habitica-integration')
        HabiticaProjectGamification = habitica_module.HabiticaProjectGamification
        
        # Create instance
        gamification = HabiticaProjectGamification()
        
        # Test API connection
        print("📡 Testing API connection...")
        user_data = gamification._api_request('GET', '/user')
        
        if user_data:
            print(f"✅ Connected to Habitica")
            print(f"   User: {user_data.get('profile', {}).get('name', 'Unknown')}")
            print(f"   Level: {user_data.get('stats', {}).get('lvl', 0)}")
            
            # Test habit setup check
            print("\n🎯 Checking for project habits...")
            tasks = gamification._api_request('GET', '/tasks/user')
            if tasks:
                project_tasks = [t for t in tasks if 'project-tracker' in t.get('tags', [])]
                print(f"   Found {len(project_tasks)} project tracker tasks")
                
                if project_tasks:
                    print("   Project habits already set up:")
                    for task in project_tasks[:3]:
                        print(f"   - {task.get('text', 'Unknown task')}")
                else:
                    print("   No project habits found (run with setup mode)")
            
            # Test productivity score calculation (mock data)
            print("\n📊 Testing productivity score calculation...")
            test_changes = {
                'files_added': 5,
                'recent_activity': 10,
                'health_change': 5,
                'duplicates': 2,
                'security_issues': 0
            }
            test_data = {'health_score': 85}
            
            score = gamification.calculate_productivity_score(test_changes, test_data)
            print(f"   Calculated productivity score: {score}")
            print(f"   Multiplier: {gamification.productivity_multiplier:.2f}x")
            
            return True
        else:
            print("❌ Failed to connect to Habitica API")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Habitica functionality: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ai_advisor_functionality():
    """Test AI Project Advisor functionality"""
    print("\n🤖 Testing AI Project Advisor Functionality...")
    print("-" * 70)
    
    try:
        # Import the module
        ai_module = importlib.import_module('ai-project-advisor')
        AIProjectAdvisor = ai_module.AIProjectAdvisor
        
        # Create instance
        advisor = AIProjectAdvisor()
        
        # Test with Gemini provider
        print("🔧 Setting up Gemini provider...")
        advisor.provider = 'gemini'
        advisor.api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        advisor.model = 'gemini-1.5-flash'  # Updated model
        
        if not advisor.api_key:
            print("❌ No Gemini API key found")
            return False
        
        print(f"✅ Configured with {advisor.providers[advisor.provider]['name']}")
        print(f"   Model: {advisor.model}")
        
        # Test prompt creation
        print("\n📝 Testing prompt generation...")
        test_project_data = {
            'name': 'Test Project',
            'has_data': True,
            'dashboard': {
                'overview': {
                    'total_files': 100,
                    'total_size': 1048576  # 1MB
                },
                'languages': {
                    'primary': 'Python',
                    'counts': {'Python': 80, 'JavaScript': 20}
                },
                'health_score': 85
            }
        }
        
        prompt = advisor.create_analysis_prompt(test_project_data, 'health')
        print(f"   Generated prompt length: {len(prompt)} characters")
        print(f"   Focus area: Project health and code quality")
        
        # Test API connection with a simple prompt
        print("\n🧪 Testing AI API connection...")
        test_prompt = "In one sentence, confirm the API is working by saying 'API connection successful'."
        
        suggestion = advisor.get_ai_suggestions(test_prompt)
        if suggestion:
            print(f"✅ AI API working")
            print(f"   Response: {suggestion[:100]}...")
            return True
        else:
            print("❌ Failed to get AI response")
            return False
            
    except Exception as e:
        print(f"❌ Error testing AI Advisor functionality: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_script_compatibility():
    """Test that scripts can work together"""
    print("\n🔗 Testing Script Compatibility...")
    print("-" * 70)
    
    try:
        # Check if both can be imported together
        habitica_module = importlib.import_module('habitica-integration')
        ai_module = importlib.import_module('ai-project-advisor')
        
        print("✅ Both scripts can be imported together")
        
        # Check shared dependencies
        print("\n📦 Checking shared configurations...")
        
        # Both use config_loader
        from config_loader import ConfigLoader
        config = ConfigLoader()
        print("✅ ConfigLoader working for both scripts")
        
        # Check stats directory structure
        stats_dir = Path('.project-stats')
        if stats_dir.exists():
            print(f"✅ Stats directory exists: {stats_dir}")
            
            # Check for required subdirectories
            ai_suggestions_dir = stats_dir / 'ai_suggestions'
            habitica_sessions_dir = stats_dir / 'habitica_sessions'
            
            for dir_path, name in [(ai_suggestions_dir, 'AI suggestions'), 
                                  (habitica_sessions_dir, 'Habitica sessions')]:
                if dir_path.exists():
                    print(f"   ✓ {name} directory exists")
                else:
                    print(f"   ✗ {name} directory missing (will be created on first use)")
        else:
            print("⚠️  No .project-stats directory (run analysis tools first)")
        
        return True
        
    except Exception as e:
        print(f"❌ Compatibility error: {e}")
        return False

def main():
    """Run all functionality tests"""
    print("🧪 SCRIPT FUNCTIONALITY TEST SUITE")
    print("=" * 70)
    
    # Load environment
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
    
    # Run tests
    habitica_ok = test_habitica_functionality()
    ai_ok = test_ai_advisor_functionality()
    compat_ok = test_script_compatibility()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 FUNCTIONALITY TEST SUMMARY")
    print("=" * 70)
    print(f"Habitica Integration: {'✅ PASSED' if habitica_ok else '❌ FAILED'}")
    print(f"AI Project Advisor: {'✅ PASSED' if ai_ok else '❌ FAILED'}")
    print(f"Script Compatibility: {'✅ PASSED' if compat_ok else '❌ FAILED'}")
    
    if habitica_ok and ai_ok and compat_ok:
        print("\n✨ All functionality tests passed!")
        print("\n💡 Next steps:")
        print("   1. Run project analysis tools to generate data")
        print("   2. Use habitica-integration.py to gamify your productivity")
        print("   3. Use ai-project-advisor.py to get AI insights")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
AI Project Advisor - Get intelligent suggestions from AI about your project
Supports multiple AI providers: OpenAI, Claude, Google Gemini, Hugging Face
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Optional, Tuple
import time

class AIProjectAdvisor:
    def __init__(self):
        self.stats_dir = Path('.project-stats')
        self.provider = None
        self.api_key = None
        self.model = None
        
        # Provider configurations
        self.providers = {
            'openai': {
                'name': 'OpenAI',
                'models': ['gpt-4-turbo-preview', 'gpt-4', 'gpt-3.5-turbo'],
                'default_model': 'gpt-3.5-turbo',
                'base_url': 'https://api.openai.com/v1/chat/completions',
                'headers_fn': self._openai_headers,
                'request_fn': self._openai_request
            },
            'claude': {
                'name': 'Anthropic Claude',
                'models': ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
                'default_model': 'claude-3-haiku-20240307',
                'base_url': 'https://api.anthropic.com/v1/messages',
                'headers_fn': self._claude_headers,
                'request_fn': self._claude_request
            },
            'gemini': {
                'name': 'Google Gemini',
                'models': ['gemini-pro', 'gemini-1.5-pro-latest'],
                'default_model': 'gemini-pro',
                'base_url': 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
                'headers_fn': self._gemini_headers,
                'request_fn': self._gemini_request
            },
            'huggingface': {
                'name': 'Hugging Face',
                'models': ['meta-llama/Llama-2-70b-chat-hf', 'mistralai/Mixtral-8x7B-Instruct-v0.1'],
                'default_model': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
                'base_url': 'https://api-inference.huggingface.co/models/{model}',
                'headers_fn': self._huggingface_headers,
                'request_fn': self._huggingface_request
            }
        }
        
        # Analysis templates
        self.analysis_areas = {
            'health': 'Project health and code quality improvements',
            'productivity': 'Developer productivity and workflow optimization',
            'architecture': 'Code structure and architecture recommendations',
            'performance': 'Performance optimization opportunities',
            'security': 'Security vulnerabilities and best practices',
            'growth': 'Project growth strategies and predictions',
            'team': 'Collaboration and team productivity tips',
            'technical_debt': 'Technical debt identification and reduction'
        }
        
    def setup_provider(self):
        """Interactive setup for AI provider"""
        print("🤖 AI PROJECT ADVISOR SETUP")
        print("="*50)
        print("\nChoose your AI provider:")
        
        providers_list = list(self.providers.keys())
        for i, key in enumerate(providers_list, 1):
            print(f"{i}. {self.providers[key]['name']}")
        
        while True:
            choice = input("\nSelect provider (1-4): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(providers_list):
                self.provider = providers_list[int(choice) - 1]
                break
            print("Invalid choice, please try again.")
        
        # Get API key
        print(f"\n📋 {self.providers[self.provider]['name']} Setup")
        print("-"*40)
        
        # Check environment variable first
        env_keys = {
            'openai': 'OPENAI_API_KEY',
            'claude': 'ANTHROPIC_API_KEY',
            'gemini': 'GOOGLE_API_KEY',
            'huggingface': 'HUGGINGFACE_API_KEY'
        }
        
        env_key = os.environ.get(env_keys[self.provider])
        if env_key:
            print(f"✓ Found API key in environment variable")
            self.api_key = env_key
        else:
            self.api_key = input(f"Enter your {self.providers[self.provider]['name']} API key: ").strip()
        
        # Select model
        print(f"\nAvailable models:")
        models = self.providers[self.provider]['models']
        for i, model in enumerate(models, 1):
            default = " (recommended)" if model == self.providers[self.provider]['default_model'] else ""
            print(f"{i}. {model}{default}")
        
        model_choice = input("\nSelect model (press Enter for default): ").strip()
        if model_choice.isdigit() and 1 <= int(model_choice) <= len(models):
            self.model = models[int(model_choice) - 1]
        else:
            self.model = self.providers[self.provider]['default_model']
        
        print(f"\n✅ Setup complete: {self.providers[self.provider]['name']} with {self.model}")
    
    def collect_project_data(self, project_path: str) -> Dict:
        """Collect all available project data"""
        print("\n📊 Collecting project data...")
        
        project_data = {
            'path': project_path,
            'name': Path(project_path).name,
            'timestamp': datetime.now().isoformat(),
            'has_data': False
        }
        
        # Load latest dashboard
        dashboard_file = self.stats_dir / 'latest_dashboard.json'
        if dashboard_file.exists():
            with open(dashboard_file, 'r') as f:
                project_data['dashboard'] = json.load(f)
                project_data['has_data'] = True
                print("✓ Loaded dashboard data")
        
        # Load recent health report
        health_reports = sorted(self.stats_dir.glob('health_report_*.json'), reverse=True)
        if health_reports:
            with open(health_reports[0], 'r') as f:
                project_data['health'] = json.load(f)
                print("✓ Loaded health report")
        
        # Load code analysis
        code_analyses = sorted(self.stats_dir.glob('code_analysis_*.json'), reverse=True)
        if code_analyses:
            with open(code_analyses[0], 'r') as f:
                project_data['code_analysis'] = json.load(f)
                print("✓ Loaded code analysis")
        
        # Load timeline data
        timeline_files = sorted(self.stats_dir.glob('timeline_*.json'), reverse=True)
        if timeline_files:
            with open(timeline_files[0], 'r') as f:
                project_data['timeline'] = json.load(f)
                print("✓ Loaded timeline data")
        
        # Load duplicate reports
        dup_reports = sorted(self.stats_dir.glob('duplicates_*.json'), reverse=True)
        if dup_reports:
            with open(dup_reports[0], 'r') as f:
                project_data['duplicates'] = json.load(f)
                print("✓ Loaded duplicate analysis")
        
        # Load productivity data if available
        productivity_db = Path.home() / '.project-productivity' / 'productivity.db'
        if productivity_db.exists():
            project_data['has_productivity'] = True
            print("✓ Found productivity data")
        
        if not project_data['has_data']:
            print("⚠️  No project data found. Run analysis tools first!")
            return None
        
        return project_data
    
    def create_analysis_prompt(self, project_data: Dict, focus_area: str) -> str:
        """Create a comprehensive prompt for AI analysis"""
        
        # Base context
        prompt = f"""You are an expert software development advisor analyzing a project called "{project_data['name']}". 
Based on the comprehensive data provided, give specific, actionable, and constructive suggestions focused on: {self.analysis_areas[focus_area]}.

PROJECT DATA SUMMARY:
"""
        
        # Add dashboard data
        if 'dashboard' in project_data:
            dashboard = project_data['dashboard']
            overview = dashboard.get('overview', {})
            
            prompt += f"""
📊 Project Overview:
- Total Files: {overview.get('total_files', 0):,}
- Total Size: {self._format_size(overview.get('total_size', 0))}
- Primary Language: {dashboard.get('languages', {}).get('primary', 'Unknown')}
- Health Score: {dashboard.get('health_score', 0)}/100
"""
            
            # Add language distribution
            if 'languages' in dashboard and dashboard['languages'].get('counts'):
                prompt += "\nLanguage Distribution:\n"
                for lang, count in list(dashboard['languages']['counts'].items())[:5]:
                    prompt += f"- {lang}: {count} files\n"
        
        # Add health issues
        if 'health' in project_data:
            health = project_data['health']
            prompt += f"\n🏥 Health Analysis (Score: {health.get('score', 0)}/100):\n"
            
            issues = health.get('issues', {})
            if issues.get('critical'):
                prompt += "Critical Issues:\n"
                for issue in issues['critical'][:5]:
                    prompt += f"- {issue}\n"
            
            if issues.get('warning'):
                prompt += "Warnings:\n"
                for issue in issues['warning'][:5]:
                    prompt += f"- {issue}\n"
            
            if health.get('recommendations'):
                prompt += "Current Recommendations:\n"
                for rec in health['recommendations'][:3]:
                    prompt += f"- {rec}\n"
        
        # Add code analysis
        if 'code_analysis' in project_data:
            code = project_data['code_analysis']
            stats = code.get('stats', {})
            
            prompt += f"""
💻 Code Analysis:
- Total Lines: {stats.get('total_lines', 0):,}
- Code Lines: {stats.get('code_lines', 0):,}
- Comment Ratio: {self._calc_ratio(stats.get('comment_lines', 0), stats.get('total_lines', 1)):.1f}%
- Functions: {code.get('function_count', 0)}
- Classes: {code.get('class_count', 0)}
"""
            
            # Top dependencies
            if code.get('top_imports'):
                prompt += "\nMost Used Dependencies:\n"
                for dep, count in list(code['top_imports'].items())[:5]:
                    prompt += f"- {dep}: {count} imports\n"
        
        # Add growth/timeline data
        if 'timeline' in project_data:
            timeline = project_data['timeline']
            prompt += f"\n📈 Project Timeline:\n"
            prompt += f"- Total Files Tracked: {timeline.get('total_files', 0)}\n"
            
            if timeline.get('date_range'):
                date_range = timeline['date_range']
                if date_range.get('oldest') and date_range.get('newest'):
                    try:
                        oldest = datetime.fromisoformat(date_range['oldest'].replace('Z', '+00:00'))
                        newest = datetime.fromisoformat(date_range['newest'].replace('Z', '+00:00'))
                        age_days = (newest - oldest).days
                        prompt += f"- Project Age: {age_days} days\n"
                        prompt += f"- Average Growth: {timeline.get('total_files', 0) / max(1, age_days):.1f} files/day\n"
                    except:
                        pass
        
        # Add duplicate information
        if 'duplicates' in project_data:
            dup = project_data['duplicates']
            prompt += f"\n🔍 Duplicate Files:\n"
            prompt += f"- Total Duplicates: {dup.get('total_duplicates', 0)}\n"
            prompt += f"- Wasted Space: {self._format_size(dup.get('total_waste', 0))}\n"
        
        # Add specific focus area context
        prompt += f"\n🎯 FOCUS AREA: {self.analysis_areas[focus_area]}\n"
        prompt += "\nBased on this data, provide:\n"
        prompt += "1. 3-5 specific issues or opportunities you've identified\n"
        prompt += "2. Concrete, actionable recommendations for each issue\n"
        prompt += "3. Priority order for implementing the recommendations\n"
        prompt += "4. Expected impact and benefits of each recommendation\n"
        prompt += "5. Any potential risks or considerations\n"
        prompt += "\nBe specific, practical, and constructive. Reference the actual data provided."
        
        return prompt
    
    def get_ai_suggestions(self, prompt: str) -> Optional[str]:
        """Get suggestions from the configured AI provider"""
        print(f"\n🤖 Getting AI suggestions from {self.providers[self.provider]['name']}...")
        
        try:
            provider_config = self.providers[self.provider]
            headers = provider_config['headers_fn']()
            request_data = provider_config['request_fn'](prompt)
            
            # Make API request
            if self.provider == 'gemini':
                url = provider_config['base_url'].format(model=self.model)
                response = requests.post(f"{url}?key={self.api_key}", 
                                       json=request_data,
                                       headers={'Content-Type': 'application/json'})
            else:
                response = requests.post(provider_config['base_url'], 
                                       headers=headers, 
                                       json=request_data)
            
            response.raise_for_status()
            
            # Extract response based on provider
            if self.provider == 'openai':
                return response.json()['choices'][0]['message']['content']
            elif self.provider == 'claude':
                return response.json()['content'][0]['text']
            elif self.provider == 'gemini':
                return response.json()['candidates'][0]['content']['parts'][0]['text']
            elif self.provider == 'huggingface':
                return response.json()[0]['generated_text']
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API Error: {e}")
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            return None
        except KeyError as e:
            print(f"❌ Unexpected response format: {e}")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def _openai_headers(self) -> Dict:
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
    
    def _openai_request(self, prompt: str) -> Dict:
        return {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are an expert software development advisor providing specific, actionable recommendations based on project analysis data.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'temperature': 0.7,
            'max_tokens': 2000
        }
    
    def _claude_headers(self) -> Dict:
        return {
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        }
    
    def _claude_request(self, prompt: str) -> Dict:
        return {
            'model': self.model,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': 2000,
            'temperature': 0.7
        }
    
    def _gemini_headers(self) -> Dict:
        return {
            'Content-Type': 'application/json'
        }
    
    def _gemini_request(self, prompt: str) -> Dict:
        return {
            'contents': [
                {
                    'parts': [
                        {
                            'text': prompt
                        }
                    ]
                }
            ],
            'generationConfig': {
                'temperature': 0.7,
                'maxOutputTokens': 2000
            }
        }
    
    def _huggingface_headers(self) -> Dict:
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
    
    def _huggingface_request(self, prompt: str) -> Dict:
        return {
            'inputs': prompt,
            'parameters': {
                'max_new_tokens': 2000,
                'temperature': 0.7,
                'return_full_text': False
            }
        }
    
    def save_suggestions(self, suggestions: str, focus_area: str, project_name: str):
        """Save AI suggestions to file"""
        suggestions_dir = self.stats_dir / 'ai_suggestions'
        suggestions_dir.mkdir(exist_ok=True)
        
        filename = f"suggestions_{focus_area}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = suggestions_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# AI Project Suggestions - {focus_area.replace('_', ' ').title()}\n\n")
            f.write(f"**Project**: {project_name}\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"**AI Provider**: {self.providers[self.provider]['name']}\n")
            f.write(f"**Model**: {self.model}\n\n")
            f.write("---\n\n")
            f.write(suggestions)
        
        print(f"\n💾 Suggestions saved to: {filepath}")
    
    def display_suggestions(self, suggestions: str, focus_area: str):
        """Display AI suggestions in a formatted way"""
        print("\n" + "="*70)
        print(f"🤖 AI SUGGESTIONS - {focus_area.replace('_', ' ').upper()}")
        print("="*70)
        print(f"\nProvider: {self.providers[self.provider]['name']} ({self.model})")
        print("-"*70)
        print("\n" + suggestions)
        print("\n" + "="*70)
    
    def interactive_analysis(self, project_path: str):
        """Run interactive analysis session"""
        # Collect project data
        project_data = self.collect_project_data(project_path)
        if not project_data:
            return
        
        while True:
            print("\n" + "="*50)
            print("🎯 ANALYSIS FOCUS AREAS")
            print("="*50)
            
            areas_list = list(self.analysis_areas.keys())
            for i, area in enumerate(areas_list, 1):
                print(f"{i}. {area.replace('_', ' ').title()} - {self.analysis_areas[area]}")
            
            print(f"\n0. Analyze all areas (comprehensive report)")
            print("Q. Quit")
            
            choice = input("\nSelect focus area: ").strip().lower()
            
            if choice == 'q':
                break
            elif choice == '0':
                # Analyze all areas
                print("\n🔄 Generating comprehensive analysis...")
                all_suggestions = []
                
                for area in areas_list[:4]:  # Limit to avoid token limits
                    print(f"\n📊 Analyzing {area.replace('_', ' ')}...")
                    prompt = self.create_analysis_prompt(project_data, area)
                    suggestions = self.get_ai_suggestions(prompt)
                    
                    if suggestions:
                        all_suggestions.append(f"## {area.replace('_', ' ').title()}\n\n{suggestions}")
                    
                    time.sleep(2)  # Rate limiting
                
                if all_suggestions:
                    full_report = "\n\n---\n\n".join(all_suggestions)
                    self.display_suggestions(full_report, "comprehensive")
                    self.save_suggestions(full_report, "comprehensive", project_data['name'])
                
            elif choice.isdigit() and 1 <= int(choice) <= len(areas_list):
                focus_area = areas_list[int(choice) - 1]
                
                # Create prompt
                prompt = self.create_analysis_prompt(project_data, focus_area)
                
                # Get suggestions
                suggestions = self.get_ai_suggestions(prompt)
                
                if suggestions:
                    self.display_suggestions(suggestions, focus_area)
                    self.save_suggestions(suggestions, focus_area, project_data['name'])
                    
                    # Ask for follow-up
                    follow_up = input("\nWould you like to explore another area? (y/n): ").lower()
                    if follow_up != 'y':
                        break
            else:
                print("Invalid choice, please try again.")
    
    def _format_size(self, size: int) -> str:
        """Format bytes to human readable"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    
    def _calc_ratio(self, part: int, whole: int) -> float:
        """Calculate percentage ratio"""
        return (part / whole * 100) if whole > 0 else 0


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Get AI-powered suggestions for your project'
    )
    parser.add_argument('path', nargs='?', default='.',
                       help='Project path to analyze')
    parser.add_argument('--provider', choices=['openai', 'claude', 'gemini', 'huggingface'],
                       help='AI provider to use')
    parser.add_argument('--api-key', help='API key for the provider')
    parser.add_argument('--model', help='Specific model to use')
    parser.add_argument('--focus', choices=list(AIProjectAdvisor().analysis_areas.keys()),
                       help='Focus area for analysis')
    parser.add_argument('--quick', action='store_true',
                       help='Quick analysis without interactive mode')
    
    args = parser.parse_args()
    
    # Initialize advisor
    advisor = AIProjectAdvisor()
    
    # Setup provider
    if args.provider and args.api_key:
        advisor.provider = args.provider
        advisor.api_key = args.api_key
        advisor.model = args.model or advisor.providers[args.provider]['default_model']
        print(f"✓ Using {advisor.providers[args.provider]['name']} with {advisor.model}")
    else:
        advisor.setup_provider()
    
    # Check for project stats
    if not advisor.stats_dir.exists():
        print("\n❌ No project statistics found!")
        print("   Run these commands first:")
        print("   python quick-stats.py")
        print("   python health-checker.py")
        print("   python project-dashboard.py")
        return
    
    print(f"\n📁 Analyzing project: {Path(args.path).resolve()}")
    
    if args.quick and args.focus:
        # Quick single analysis
        project_data = advisor.collect_project_data(args.path)
        if project_data:
            prompt = advisor.create_analysis_prompt(project_data, args.focus)
            suggestions = advisor.get_ai_suggestions(prompt)
            if suggestions:
                advisor.display_suggestions(suggestions, args.focus)
                advisor.save_suggestions(suggestions, args.focus, project_data['name'])
    else:
        # Interactive mode
        advisor.interactive_analysis(args.path)
    
    print("\n✨ Analysis complete!")


if __name__ == "__main__":
    main()

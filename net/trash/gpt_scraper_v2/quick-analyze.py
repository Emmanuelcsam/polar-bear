#!/usr/bin/env python3
"""
Quick Analyze - Simplified ChatGPT Analyzer
One-click analysis with minimal questions
"""

import os
import sys
from pathlib import Path

# Import the main analyzer
try:
    from gptscraper import ChatGPTAnalyzer, ConversationType
except ImportError:
    print("❌ Error: gptscraper.py not found in the current directory!")
    print("Make sure this script is in the same folder as gptscraper.py")
    sys.exit(1)


def find_conversations_file():
    """Try to find conversations.json automatically"""
    # Check common locations
    possible_locations = [
        'conversations.json',
        'export/conversations.json',
        'chatgpt_export/conversations.json',
        str(Path.home() / 'Downloads' / 'conversations.json'),
        str(Path.home() / 'Desktop' / 'conversations.json'),
    ]
    
    # Also check for any .json files in current directory
    for file in Path('.').glob('*.json'):
        if 'conversation' in file.name.lower():
            possible_locations.insert(0, str(file))
    
    for location in possible_locations:
        if os.path.exists(location):
            return location
            
    return None


def quick_analyze():
    """Run analysis with minimal configuration"""
    print("\n" + "="*60)
    print("🚀 ChatGPT Quick Analyzer")
    print("="*60)
    print("\nI'll analyze your ChatGPT conversations with minimal setup!\n")
    
    # Try to find the file automatically
    json_file = find_conversations_file()
    
    if json_file:
        file_size = os.path.getsize(json_file) / 1024 / 1024  # MB
        print(f"✅ Found: {json_file} ({file_size:.1f} MB)")
        use_this = input("\nUse this file? (Y/n): ").strip().lower()
        
        if use_this == 'n':
            json_file = None
    
    if not json_file:
        print("\n📁 Please provide your conversations.json file")
        print("(You can drag and drop it here)")
        json_file = input("\nFile path: ").strip().strip('"').strip("'")
        
        if not os.path.exists(json_file):
            print(f"\n❌ File not found: {json_file}")
            print("\nTo get your conversations.json:")
            print("1. Go to ChatGPT → Settings → Data Controls")
            print("2. Click 'Export data'")
            print("3. Download from email and extract the ZIP")
            return 1
    
    # Quick format selection
    print("\n📊 Choose your report format:")
    print("1. Quick HTML report (recommended)")
    print("2. Full analysis (HTML + Excel + Extract code)")
    
    choice = input("\nYour choice (1-2, default=1): ").strip() or "1"
    
    if choice == "1":
        formats = ['html']
        extract = False
        output_dir = 'quick_analysis'
    else:
        formats = ['html', 'csv']
        extract = True
        output_dir = 'full_analysis'
    
    # Create analyzer
    print(f"\n🔍 Analyzing your conversations...")
    print("This may take a moment for large exports...\n")
    
    analyzer = ChatGPTAnalyzer(
        mode='export',
        export_file=json_file,
        output_dir=output_dir,
        use_cache=True
    )
    
    try:
        # Extract conversations
        conversations = analyzer.extract_conversations_from_export()
        
        if not conversations:
            print("❌ No conversations found in the file!")
            return 1
        
        # Quick stats
        total = len(conversations)
        code_count = sum(1 for c in conversations if c.has_code)
        research_count = sum(1 for c in conversations if c.has_research)
        
        print(f"📊 Found {total} conversations:")
        print(f"   💻 {code_count} with code")
        print(f"   🔬 {research_count} with research")
        print(f"   🔗 {sum(1 for c in conversations if c.has_links)} with links")
        
        # Generate reports
        print(f"\n📝 Generating {', '.join(formats).upper()} report(s)...")
        analyzer.generate_reports(conversations, formats)
        
        # Extract content if requested
        if extract:
            relevant = [c for c in conversations 
                       if c.conversation_type != ConversationType.GENERAL]
            
            if relevant:
                print(f"\n📤 Extracting content from {len(relevant)} conversations...")
                extract_ids = [c.id for c in relevant]
                analyzer.extract_code_and_research(conversations, extract_ids)
        
        # Success!
        print("\n" + "="*60)
        print("✨ Analysis Complete!")
        print("="*60)
        print(f"\n📁 Results saved to: {os.path.abspath(output_dir)}/")
        
        # Open HTML report
        html_files = list(Path(output_dir).glob('*.html'))
        if html_files:
            newest_html = max(html_files, key=os.path.getctime)
            try:
                import webbrowser
                webbrowser.open(f'file://{newest_html.absolute()}')
                print("🌐 Opening report in your browser...")
            except:
                print(f"🌐 Open this file in your browser: {newest_html}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n✋ Cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nFor more options, run: python gptscraper.py")
        return 1
    finally:
        analyzer.cleanup()


def main():
    """Main entry point"""
    print("🤖 ChatGPT Quick Analyzer")
    print("The fastest way to analyze your conversations!\n")
    
    if '--help' in sys.argv or '-h' in sys.argv:
        print("Usage: python quick_analyze.py")
        print("\nThis script will:")
        print("1. Find your conversations.json automatically")
        print("2. Ask minimal questions")
        print("3. Generate a beautiful HTML report")
        print("4. Open it in your browser")
        print("\nFor advanced options, use: python gptscraper.py")
        return 0
    
    return quick_analyze()


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
ChatGPT Analyzer - Example Usage Script
Demonstrates how to use the analyzer programmatically
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gptscraper import ChatGPTAnalyzer, ConversationType, ConversationMetadata


def example_basic_analysis():
    """Basic example: Analyze export file and generate HTML report"""
    print("=== Basic Analysis Example ===\n")
    
    # Create analyzer
    analyzer = ChatGPTAnalyzer(
        mode='export',
        export_file='conversations.json',
        output_dir='my_analysis'
    )
    
    # Extract and analyze conversations
    conversations = analyzer.extract_conversations_from_export()
    print(f"Found {len(conversations)} conversations")
    
    # Generate HTML report
    analyzer.generate_reports(conversations, formats=['html'])
    print("✅ HTML report generated!")


def example_filtered_analysis():
    """Example: Analyze only specific types of conversations"""
    print("\n=== Filtered Analysis Example ===\n")
    
    analyzer = ChatGPTAnalyzer(
        mode='export',
        export_file='conversations.json'
    )
    
    # Get all conversations
    all_conversations = analyzer.extract_conversations_from_export()
    
    # Filter only code conversations
    code_conversations = [
        conv for conv in all_conversations 
        if conv.conversation_type == ConversationType.CODE
    ]
    print(f"Found {len(code_conversations)} code conversations")
    
    # Filter only Python conversations
    python_conversations = [
        conv for conv in all_conversations
        if 'python' in conv.languages
    ]
    print(f"Found {len(python_conversations)} Python conversations")
    
    # Generate report for Python conversations only
    if python_conversations:
        analyzer.generate_reports(python_conversations, formats=['html', 'csv'])
        print("✅ Python conversations report generated!")


def example_date_range_analysis():
    """Example: Analyze conversations from a specific date range"""
    print("\n=== Date Range Analysis Example ===\n")
    
    analyzer = ChatGPTAnalyzer(
        mode='export',
        export_file='conversations.json'
    )
    
    conversations = analyzer.extract_conversations_from_export()
    
    # Filter conversations from the last 30 days
    cutoff_date = datetime.now() - timedelta(days=30)
    recent_conversations = []
    
    for conv in conversations:
        if conv.update_time:
            try:
                # Parse timestamp (adjust format as needed)
                update_date = datetime.fromisoformat(conv.update_time.replace('Z', '+00:00'))
                if update_date > cutoff_date:
                    recent_conversations.append(conv)
            except:
                pass
                
    print(f"Found {len(recent_conversations)} conversations from the last 30 days")
    
    if recent_conversations:
        analyzer.generate_reports(recent_conversations, formats=['markdown'])
        print("✅ Recent conversations report generated!")


def example_content_extraction():
    """Example: Extract code and research content"""
    print("\n=== Content Extraction Example ===\n")
    
    analyzer = ChatGPTAnalyzer(
        mode='export',
        export_file='conversations.json'
    )
    
    conversations = analyzer.extract_conversations_from_export()
    
    # Find conversations with significant code
    code_heavy_conversations = [
        conv for conv in conversations
        if conv.code_blocks >= 3  # At least 3 code blocks
    ]
    
    print(f"Found {len(code_heavy_conversations)} code-heavy conversations")
    
    if code_heavy_conversations:
        # Extract code from these conversations
        conv_ids = [conv.id for conv in code_heavy_conversations[:5]]  # First 5
        analyzer.extract_code_and_research(conversations, selected_ids=conv_ids)
        print("✅ Code extracted from selected conversations!")


def example_statistics_summary():
    """Example: Generate detailed statistics"""
    print("\n=== Statistics Summary Example ===\n")
    
    analyzer = ChatGPTAnalyzer(
        mode='export',
        export_file='conversations.json'
    )
    
    conversations = analyzer.extract_conversations_from_export()
    
    # Calculate statistics
    total_messages = sum(conv.total_messages for conv in conversations)
    total_words = sum(conv.word_count for conv in conversations)
    total_code_blocks = sum(conv.code_blocks for conv in conversations)
    
    # Language distribution
    language_counts = {}
    for conv in conversations:
        for lang in conv.languages:
            language_counts[lang] = language_counts.get(lang, 0) + 1
            
    # Type distribution
    type_counts = {}
    for conv in conversations:
        conv_type = conv.conversation_type.value
        type_counts[conv_type] = type_counts.get(conv_type, 0) + 1
        
    print(f"📊 ChatGPT Usage Statistics:")
    print(f"   Total Conversations: {len(conversations)}")
    print(f"   Total Messages: {total_messages:,}")
    print(f"   Total Words: {total_words:,}")
    print(f"   Total Code Blocks: {total_code_blocks:,}")
    print(f"\n📈 Conversation Types:")
    for conv_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(conversations)) * 100
        print(f"   {conv_type.title()}: {count} ({percentage:.1f}%)")
    print(f"\n💻 Top Programming Languages:")
    for lang, count in sorted(language_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {lang}: {count} conversations")


def example_custom_analysis():
    """Example: Custom analysis with specific criteria"""
    print("\n=== Custom Analysis Example ===\n")
    
    analyzer = ChatGPTAnalyzer(
        mode='export',
        export_file='conversations.json'
    )
    
    conversations = analyzer.extract_conversations_from_export()
    
    # Find long technical discussions
    technical_discussions = [
        conv for conv in conversations
        if (conv.word_count > 5000 and  # Long conversation
            (conv.has_code or conv.has_research) and  # Technical content
            conv.total_messages > 10)  # Back-and-forth discussion
    ]
    
    print(f"Found {len(technical_discussions)} long technical discussions")
    
    # Find conversations with external resources
    resource_conversations = [
        conv for conv in conversations
        if conv.url_count > 5  # Multiple external links
    ]
    
    print(f"Found {len(resource_conversations)} conversations with many external resources")
    
    # Create custom report
    if technical_discussions:
        # Sort by word count
        technical_discussions.sort(key=lambda x: x.word_count, reverse=True)
        
        # Save custom report
        output_file = Path(analyzer.output_dir) / 'technical_discussions.md'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Long Technical Discussions\n\n")
            
            for conv in technical_discussions[:20]:  # Top 20
                f.write(f"## [{conv.title}]({conv.url})\n\n")
                f.write(f"- **Words**: {conv.word_count:,}\n")
                f.write(f"- **Messages**: {conv.total_messages}\n")
                if conv.languages:
                    f.write(f"- **Languages**: {', '.join(conv.languages)}\n")
                if conv.research_keywords:
                    f.write(f"- **Topics**: {', '.join(conv.research_keywords[:5])}\n")
                f.write("\n---\n\n")
                
        print(f"✅ Custom report saved to: {output_file}")


def example_live_mode():
    """Example: Live mode analysis (requires Chrome and credentials)"""
    print("\n=== Live Mode Example ===\n")
    
    # Check for credentials
    email = os.getenv('CHATGPT_EMAIL')
    password = os.getenv('CHATGPT_PASSWORD')
    
    if not email or not password:
        print("⚠️  Live mode requires CHATGPT_EMAIL and CHATGPT_PASSWORD environment variables")
        print("   Set them in .env file or export them in your shell")
        return
        
    analyzer = ChatGPTAnalyzer(mode='live')
    
    try:
        # Setup browser
        analyzer.setup_driver(headless=True)
        
        # Login
        if analyzer.login(email, password):
            print("✅ Logged in successfully!")
            
            # Extract conversations
            conversations = analyzer.extract_conversations_from_browser()
            print(f"Found {len(conversations)} conversations")
            
            # Generate report
            analyzer.generate_reports(conversations, formats=['html'])
            print("✅ Live analysis complete!")
        else:
            print("❌ Login failed")
            
    finally:
        analyzer.cleanup()


def main():
    """Run all examples"""
    print("🤖 ChatGPT Analyzer - Example Usage\n")
    
    # Check if export file exists
    if not os.path.exists('conversations.json'):
        print("❌ conversations.json not found!")
        print("\nTo get your conversations:")
        print("1. Go to ChatGPT Settings → Data Controls")
        print("2. Export your data")
        print("3. Extract conversations.json from the ZIP file")
        print("4. Place it in this directory")
        return
        
    # Run examples
    try:
        example_basic_analysis()
        example_filtered_analysis()
        example_date_range_analysis()
        example_content_extraction()
        example_statistics_summary()
        example_custom_analysis()
        
        # Optionally run live mode
        # example_live_mode()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n✨ All examples completed!")
    print(f"📁 Check the output directories for generated reports")


if __name__ == "__main__":
    main()

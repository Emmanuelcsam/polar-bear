#!/usr/bin/env python3
"""
ChatGPT Universal Analyzer with Content Extraction
Combines live browser analysis with offline export analysis
Supports both folder structures and sophisticated content detection
Can extract code files and research documents from conversations
"""

import os
import time
import csv
import json
import re
import sys
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path

# Optional imports for browser automation
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Warning: Selenium not installed. Browser automation features disabled.")
    print("Install with: pip install selenium")

# Optional imports for PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: ReportLab not installed. PDF generation disabled.")
    print("Install with: pip install reportlab")

# Optional import for markdown to PDF
try:
    import markdown
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False


class ChatGPTUniversalAnalyzer:
    """Universal analyzer for ChatGPT conversations - both live and exported"""
    
    def __init__(self, mode='export', export_file=None, rename_chats=False, extract_content=False):
        self.mode = mode  # 'export' or 'live'
        self.export_file = export_file
        self.rename_chats = rename_chats
        self.extract_content = extract_content
        self.driver = None
        self.selected_conversations = []
        
        # Enhanced code patterns from gptscraper.py
        self.code_patterns = [
            r'```[\w]*\n[\s\S]*?```',  # Code blocks
            r'def\s+\w+\s*\(',          # Python functions
            r'class\s+\w+\s*[:\(]',     # Python classes
            r'import\s+\w+',            # Import statements
            r'from\s+\w+\s+import',     # From imports
            r'if\s+__name__\s*==',      # Python main
            r'print\s*\(',              # Print statements
            r'for\s+\w+\s+in\s+',       # For loops
            r'while\s+.*:',             # While loops
            r'function\s+\w+\s*\(',     # JavaScript functions
            r'const\s+\w+\s*=',         # JavaScript const
            r'let\s+\w+\s*=',           # JavaScript let
            r'var\s+\w+\s*=',           # JavaScript var
            r'npm\s+install',           # NPM commands
            r'pip\s+install',           # Pip commands
            r'<[^>]+>.*<\/[^>]+>',      # HTML tags
            r'SELECT\s+.*FROM',         # SQL queries
            r'docker\s+run',            # Docker commands
            r'git\s+(clone|commit|push|pull)', # Git commands
        ]
        
        # Research keywords from gptscraper.py
        self.research_keywords = [
            'research', 'analysis', 'study', 'investigation', 'examine',
            'explore', 'analyze', 'data', 'statistics', 'findings',
            'conclusion', 'hypothesis', 'methodology', 'results',
            'survey', 'experiment', 'observation', 'theory', 'evidence',
            'paper', 'article', 'publication', 'citation', 'reference',
            'dataset', 'correlation', 'regression', 'significant'
        ]
        
        # Additional patterns for links/URLs
        self.url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        
    def setup_driver(self):
        """Setup Chrome driver for live analysis"""
        if not SELENIUM_AVAILABLE:
            raise Exception("Selenium not available. Install with: pip install selenium")
            
        options = Options()
        if not self.rename_chats:  # Run headless if not renaming
            options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        chrome_driver_path = os.getenv('CHROMEDRIVER_PATH', '/usr/local/bin/chromedriver')
        service = ChromeService(chrome_driver_path)
        self.driver = webdriver.Chrome(service=service, options=options)
        
    def login(self):
        """Login to ChatGPT"""
        email = os.getenv('CHATGPT_EMAIL')
        password = os.getenv('CHATGPT_PASSWORD')
        
        if not email or not password:
            raise Exception("Please set CHATGPT_EMAIL and CHATGPT_PASSWORD environment variables")
            
        self.driver.get('https://chat.openai.com')
        time.sleep(3)
        
        try:
            self.driver.find_element(By.LINK_TEXT, 'Log in').click()
            time.sleep(2)
        except NoSuchElementException:
            pass
            
        self.driver.find_element(By.NAME, 'email').send_keys(email + Keys.RETURN)
        time.sleep(2)
        self.driver.find_element(By.NAME, 'password').send_keys(password + Keys.RETURN)
        time.sleep(5)
        
    def extract_live_conversations(self) -> List[Dict]:
        """Extract conversations from live ChatGPT interface"""
        try:
            sidebar = self.driver.find_element(By.CSS_SELECTOR, 'nav[aria-label="Chat history"]')
        except NoSuchElementException:
            print("Could not find chat history sidebar")
            return []
            
        # Extract folders and their conversations
        folders_data = {}
        
        # First, try to find folders
        try:
            folder_buttons = sidebar.find_elements(By.CSS_SELECTOR, 'button[data-testid*="folder"]')
            for folder_btn in folder_buttons:
                folder_name = folder_btn.text.strip() or 'Unnamed Folder'
                folder_btn.click()
                time.sleep(1)
                
                # Extract conversations in this folder
                folder_convs = self._extract_conversation_links()
                folders_data[folder_name] = folder_convs
                
                # Close folder
                folder_btn.click()
                time.sleep(0.5)
        except:
            pass
            
        # Extract standalone conversations
        all_convs = self._extract_conversation_links()
        
        # Organize results
        results = []
        for conv in all_convs:
            # Check if this conversation is in a folder
            in_folder = False
            for folder_name, folder_convs in folders_data.items():
                if any(fc['url'] == conv['url'] for fc in folder_convs):
                    conv['folder'] = folder_name
                    in_folder = True
                    break
            if not in_folder:
                conv['folder'] = 'No Folder'
            results.append(conv)
            
        return results
        
    def _extract_conversation_links(self) -> List[Dict]:
        """Helper to extract conversation links from current view"""
        conversations = []
        links = self.driver.find_elements(By.CSS_SELECTOR, 'a[href*="/c/"]')
        
        for link in links:
            url = link.get_attribute('href')
            try:
                title_elem = link.find_element(By.CSS_SELECTOR, 'h3, div[class*="title"], div[class*="name"]')
                title = title_elem.text.strip() or 'Untitled Chat'
            except NoSuchElementException:
                title = 'Untitled Chat'
                
            conversations.append({
                'url': url,
                'title': title,
                'id': url.split('/c/')[-1] if '/c/' in url else None
            })
            
        return conversations
        
    def analyze_live_conversation(self, conversation: Dict) -> Dict:
        """Analyze a single conversation from live ChatGPT"""
        self.driver.get(conversation['url'])
        time.sleep(3)
        
        # Get all message content
        messages = []
        try:
            message_elems = self.driver.find_elements(By.CSS_SELECTOR, 'div[data-message-author-role]')
            for elem in message_elems:
                messages.append(elem.text)
        except:
            # Fallback to other selectors
            try:
                message_elems = self.driver.find_elements(By.CSS_SELECTOR, 'div.group')
                for elem in message_elems:
                    messages.append(elem.text)
            except:
                pass
                
        # Combine all messages for analysis
        full_text = '\n'.join(messages)
        
        # Analyze content
        analysis = self.analyze_text_content(full_text)
        
        # Add conversation metadata
        analysis.update({
            'title': conversation['title'],
            'url': conversation['url'],
            'id': conversation['id'],
            'folder': conversation.get('folder', 'No Folder'),
            'total_messages': len(messages)
        })
        
        return analysis
        
    def analyze_text_content(self, text: str) -> Dict:
        """Analyze text content for code, research, and other patterns"""
        has_code = any(re.search(pattern, text, re.IGNORECASE) for pattern in self.code_patterns)
        
        # Count code blocks
        code_blocks = len(re.findall(r'```[\w]*\n[\s\S]*?```', text))
        
        # Extract languages
        languages = self.extract_code_languages(text)
        
        # Check for research content
        text_lower = text.lower()
        research_keywords_found = [kw for kw in self.research_keywords if kw in text_lower]
        has_research = len(research_keywords_found) >= 3
        
        # Check for URLs/links
        urls = re.findall(self.url_pattern, text)
        has_links = len(urls) > 0
        
        # Determine conversation type
        conv_types = []
        if has_code:
            conv_types.append('Code')
        if has_research:
            conv_types.append('Research')
        if has_links and not has_research:
            conv_types.append('Links')
            
        return {
            'has_code': has_code,
            'has_research': has_research,
            'has_links': has_links,
            'code_blocks': code_blocks,
            'languages': languages,
            'url_count': len(urls),
            'research_keywords': research_keywords_found[:5],  # Top 5 keywords
            'detected_type': ' & '.join(conv_types) if conv_types else 'General'
        }
        
    def extract_code_languages(self, text: str) -> List[str]:
        """Extract programming languages from text"""
        languages = set()
        
        # From code blocks
        code_block_pattern = r'```(\w+)\n'
        matches = re.findall(code_block_pattern, text)
        for match in matches:
            if match and match.lower() not in ['', 'text', 'output', 'bash', 'shell', 'console']:
                languages.add(match.lower())
                
        # Detect from content patterns
        if re.search(r'def\s+\w+\s*\(|class\s+\w+|import\s+\w+|from\s+\w+\s+import', text):
            languages.add('python')
        if re.search(r'function\s+\w+\s*\(|const\s+\w+\s*=|let\s+\w+\s*=|=>\s*{', text):
            languages.add('javascript')
        if re.search(r'#include\s*<|int\s+main\s*\(|std::', text):
            languages.add('c++')
        if re.search(r'public\s+class|public\s+static\s+void\s+main', text):
            languages.add('java')
        if re.search(r'<[^>]+>.*<\/[^>]+>|<!DOCTYPE\s+html>', text, re.IGNORECASE):
            languages.add('html')
        if re.search(r'SELECT\s+.*FROM|INSERT\s+INTO|UPDATE\s+.*SET', text, re.IGNORECASE):
            languages.add('sql')
        if re.search(r'package\s+main|func\s+main\s*\(', text):
            languages.add('go')
        if re.search(r'fn\s+main\s*\(|let\s+mut\s+|impl\s+\w+\s+for', text):
            languages.add('rust')
            
        return sorted(list(languages))
        
    def rename_live_conversation(self, conversation: Dict, analysis: Dict):
        """Rename a conversation in live ChatGPT"""
        if not self.rename_chats:
            return
            
        # Navigate to conversation
        self.driver.get(conversation['url'])
        time.sleep(2)
        
        # Create new title
        prefix = f"[{analysis['detected_type']}]"
        if analysis['languages']:
            prefix += f" ({', '.join(analysis['languages'][:2])})"
            
        new_title = f"{prefix} {conversation['title']}"
        
        try:
            # Click on title to edit
            title_elem = self.driver.find_element(By.CSS_SELECTOR, 'h1, div[contenteditable="true"]')
            title_elem.click()
            time.sleep(1)
            
            # Clear and enter new title
            title_elem.clear()
            title_elem.send_keys(new_title)
            title_elem.send_keys(Keys.RETURN)
            time.sleep(1)
            
            print(f"Renamed: {conversation['title']} -> {new_title}")
        except Exception as e:
            print(f"Could not rename conversation: {e}")
            
    def analyze_export_file(self) -> List[Dict]:
        """Analyze conversations from export file"""
        with open(self.export_file, 'r', encoding='utf-8') as f:
            export_data = json.load(f)
            
        # Get all conversations
        conversations = export_data.get('conversations', [])
        projects = export_data.get('projects', [])
        
        # Create project mapping
        project_map = {}
        for project in projects:
            for conv_id in project.get('conversation_ids', []):
                project_map[conv_id] = project.get('name', 'Unnamed Project')
                
        results = []
        for conv in conversations:
            # Extract text from conversation
            text = self._extract_text_from_export_conversation(conv)
            
            # Analyze content
            analysis = self.analyze_text_content(text)
            
            # Add metadata
            analysis.update({
                'title': conv.get('title', 'Untitled'),
                'id': conv.get('id', ''),
                'url': f"https://chat.openai.com/c/{conv.get('id', '')}" if conv.get('id') else 'N/A',
                'folder': project_map.get(conv.get('id'), 'No Project'),
                'create_time': conv.get('create_time', ''),
                'update_time': conv.get('update_time', ''),
                'total_messages': self._count_messages_in_export(conv)
            })
            
            results.append(analysis)
            
        return results
        
    def _extract_text_from_export_conversation(self, conversation: Dict) -> str:
        """Extract all text from an exported conversation"""
        texts = []
        
        # Handle new format with mapping
        mapping = conversation.get('mapping', {})
        if mapping:
            for node_id, node in mapping.items():
                message = node.get('message', {})
                if message and message.get('content', {}).get('content_type') == 'text':
                    parts = message.get('content', {}).get('parts', [])
                    texts.extend(parts)
        else:
            # Handle old format
            messages = conversation.get('messages', [])
            for message in messages:
                content = message.get('content', {})
                if isinstance(content, dict):
                    text = content.get('text', '')
                else:
                    text = str(content)
                texts.append(text)
                
        return '\n'.join(str(t) for t in texts)
        
    def _count_messages_in_export(self, conversation: Dict) -> int:
        """Count messages in exported conversation"""
        mapping = conversation.get('mapping', {})
        if mapping:
            return sum(1 for node in mapping.values() 
                      if node.get('message', {}).get('content', {}).get('content_type') == 'text')
        else:
            return len(conversation.get('messages', []))
            
    def save_results(self, results: List[Dict], output_format: str = 'all'):
        """Save analysis results in multiple formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"chatgpt_analysis_{timestamp}"
        
        # Filter to only relevant conversations
        filtered_results = [r for r in results 
                          if r['has_code'] or r['has_research'] or r['has_links']]
        
        if not filtered_results:
            print("No relevant conversations found (with code, research, or links)")
            return
            
        if output_format in ['csv', 'all']:
            self._save_csv(filtered_results, f"{base_name}.csv")
            
        if output_format in ['json', 'all']:
            self._save_json(filtered_results, f"{base_name}.json")
            
        if output_format in ['markdown', 'all']:
            self._save_markdown(filtered_results, f"{base_name}.md")
            
        if output_format in ['html', 'all']:
            self._save_html(filtered_results, f"{base_name}.html")
            
    def _save_csv(self, results: List[Dict], filename: str):
        """Save results as CSV"""
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'folder', 'title', 'detected_type', 'url', 'has_code', 'has_research',
                'has_links', 'languages', 'code_blocks', 'url_count', 'total_messages',
                'research_keywords', 'create_time', 'update_time'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            
            for result in results:
                row = result.copy()
                row['languages'] = ', '.join(result.get('languages', []))
                row['research_keywords'] = ', '.join(result.get('research_keywords', []))
                writer.writerow(row)
                
        print(f"CSV saved to: {filename}")
        
    def _save_json(self, results: List[Dict], filename: str):
        """Save results as JSON"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"JSON saved to: {filename}")
        
    def _save_markdown(self, results: List[Dict], filename: str):
        """Save results as Markdown"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# ChatGPT Conversation Analysis\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            self._write_summary_stats(f, results)
            
            # Group by folder/project
            folders = defaultdict(list)
            for result in results:
                folders[result.get('folder', 'No Folder')].append(result)
                
            # Write each folder section
            for folder_name in sorted(folders.keys()):
                f.write(f"\n## 📁 {folder_name}\n\n")
                
                folder_results = folders[folder_name]
                
                # Separate by type
                code_only = [r for r in folder_results if r['has_code'] and not r['has_research']]
                research_only = [r for r in folder_results if r['has_research'] and not r['has_code']]
                both = [r for r in folder_results if r['has_code'] and r['has_research']]
                links_only = [r for r in folder_results 
                           if r['has_links'] and not r['has_code'] and not r['has_research']]
                
                if both:
                    f.write("### 🔬💻 Code & Research\n\n")
                    for chat in both:
                        self._write_chat_entry(f, chat)
                        
                if code_only:
                    f.write("### 💻 Code Development\n\n")
                    for chat in code_only:
                        self._write_chat_entry(f, chat)
                        
                if research_only:
                    f.write("### 🔬 Research\n\n")
                    for chat in research_only:
                        self._write_chat_entry(f, chat)
                        
                if links_only:
                    f.write("### 🔗 Links & Resources\n\n")
                    for chat in links_only:
                        self._write_chat_entry(f, chat)
                        
        print(f"Markdown saved to: {filename}")
        
    def _save_html(self, results: List[Dict], filename: str):
        """Save results as interactive HTML"""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>ChatGPT Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1, h2, h3 { color: #333; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .stat-card { background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }
        .stat-number { font-size: 2em; font-weight: bold; color: #007bff; }
        .folder-section { margin: 20px 0; }
        .chat-card { background: #f8f9fa; margin: 10px 0; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff; }
        .chat-card.code { border-left-color: #28a745; }
        .chat-card.research { border-left-color: #17a2b8; }
        .chat-card.both { border-left-color: #6610f2; }
        .tags { margin: 5px 0; }
        .tag { display: inline-block; background: #e9ecef; padding: 3px 8px; border-radius: 4px; margin: 2px; font-size: 0.85em; }
        .tag.language { background: #d1ecf1; color: #0c5460; }
        .tag.keyword { background: #f8d7da; color: #721c24; }
        .metadata { color: #6c757d; font-size: 0.9em; margin-top: 10px; }
        a { color: #007bff; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .filter-buttons { margin: 20px 0; }
        .filter-btn { padding: 8px 16px; margin: 5px; border: 1px solid #007bff; background: white; color: #007bff; border-radius: 5px; cursor: pointer; }
        .filter-btn.active { background: #007bff; color: white; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>ChatGPT Conversation Analysis</h1>
        <p>Generated on: {timestamp}</p>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{total}</div>
                <div>Total Conversations</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{code_count}</div>
                <div>With Code</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{research_count}</div>
                <div>With Research</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{both_count}</div>
                <div>Code & Research</div>
            </div>
        </div>
        
        <div class="filter-buttons">
            <button class="filter-btn active" onclick="filterChats('all')">All</button>
            <button class="filter-btn" onclick="filterChats('code')">Code Only</button>
            <button class="filter-btn" onclick="filterChats('research')">Research Only</button>
            <button class="filter-btn" onclick="filterChats('both')">Both</button>
        </div>
        
        <div id="content">
            {content}
        </div>
    </div>
    
    <script>
        function filterChats(type) {
            const cards = document.querySelectorAll('.chat-card');
            const buttons = document.querySelectorAll('.filter-btn');
            
            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            cards.forEach(card => {
                if (type === 'all') {
                    card.classList.remove('hidden');
                } else {
                    const hasCode = card.dataset.hasCode === 'true';
                    const hasResearch = card.dataset.hasResearch === 'true';
                    
                    if (type === 'code' && hasCode && !hasResearch) {
                        card.classList.remove('hidden');
                    } else if (type === 'research' && hasResearch && !hasCode) {
                        card.classList.remove('hidden');
                    } else if (type === 'both' && hasCode && hasResearch) {
                        card.classList.remove('hidden');
                    } else {
                        card.classList.add('hidden');
                    }
                }
            });
        }
    </script>
</body>
</html>
        """
        
        # Generate content
        folders = defaultdict(list)
        for result in results:
            folders[result.get('folder', 'No Folder')].append(result)
            
        content_html = ""
        for folder_name in sorted(folders.keys()):
            content_html += f'<div class="folder-section"><h2>📁 {folder_name}</h2>'
            
            for chat in folders[folder_name]:
                card_class = "chat-card"
                if chat['has_code'] and chat['has_research']:
                    card_class += " both"
                elif chat['has_code']:
                    card_class += " code"
                elif chat['has_research']:
                    card_class += " research"
                    
                content_html += f'''
                <div class="{card_class}" data-has-code="{str(chat['has_code']).lower()}" 
                     data-has-research="{str(chat['has_research']).lower()}">
                    <h3><a href="{chat['url']}">{chat['title']}</a></h3>
                    <div class="tags">
                '''
                
                if chat['has_code']:
                    content_html += '<span class="tag">💻 Code</span>'
                if chat['has_research']:
                    content_html += '<span class="tag">🔬 Research</span>'
                if chat['has_links']:
                    content_html += f'<span class="tag">🔗 {chat["url_count"]} Links</span>'
                    
                for lang in chat.get('languages', []):
                    content_html += f'<span class="tag language">{lang}</span>'
                    
                content_html += '</div>'
                
                if chat.get('research_keywords'):
                    content_html += '<div class="tags">Keywords: '
                    for kw in chat['research_keywords'][:5]:
                        content_html += f'<span class="tag keyword">{kw}</span>'
                    content_html += '</div>'
                    
                content_html += f'''
                    <div class="metadata">
                        Messages: {chat['total_messages']} | 
                        Code blocks: {chat['code_blocks']}
                    </div>
                </div>
                '''
                
            content_html += '</div>'
            
        # Calculate stats
        total = len(results)
        code_count = sum(1 for r in results if r['has_code'])
        research_count = sum(1 for r in results if r['has_research'])
        both_count = sum(1 for r in results if r['has_code'] and r['has_research'])
        
        # Fill template
        html = html_content.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            total=total,
            code_count=code_count,
            research_count=research_count,
            both_count=both_count,
            content=content_html
        )
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
            
        print(f"HTML saved to: {filename}")
        
    def _write_summary_stats(self, f, results: List[Dict]):
        """Write summary statistics to file"""
        total = len(results)
        code_count = sum(1 for r in results if r['has_code'])
        research_count = sum(1 for r in results if r['has_research'])
        both_count = sum(1 for r in results if r['has_code'] and r['has_research'])
        links_count = sum(1 for r in results if r['has_links'])
        
        f.write("## Summary Statistics\n\n")
        f.write(f"- **Total Conversations Analyzed**: {total}\n")
        f.write(f"- **Conversations with Code**: {code_count} ({code_count/total*100:.1f}%)\n")
        f.write(f"- **Conversations with Research**: {research_count} ({research_count/total*100:.1f}%)\n")
        f.write(f"- **Conversations with Both**: {both_count} ({both_count/total*100:.1f}%)\n")
        f.write(f"- **Conversations with Links**: {links_count} ({links_count/total*100:.1f}%)\n\n")
        
        # Language distribution
        all_languages = defaultdict(int)
        for r in results:
            for lang in r.get('languages', []):
                all_languages[lang] += 1
                
        if all_languages:
            f.write("### Programming Languages\n\n")
            for lang, count in sorted(all_languages.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- **{lang}**: {count} conversations\n")
            f.write("\n")
            
    def _write_chat_entry(self, f, chat: Dict):
        """Write a single chat entry in markdown"""
        f.write(f"- **[{chat['title']}]({chat['url']})**\n")
        
        if chat.get('languages'):
            f.write(f"  - Languages: `{', '.join(chat['languages'])}`\n")
            
        if chat['has_code']:
            f.write(f"  - Code blocks: {chat['code_blocks']}\n")
            
        if chat['has_research'] and chat.get('research_keywords'):
            f.write(f"  - Research keywords: {', '.join(chat['research_keywords'][:3])}\n")
            
        if chat['has_links']:
            f.write(f"  - Links found: {chat['url_count']}\n")
            
        f.write(f"  - Total messages: {chat['total_messages']}\n\n")
        
    def interactive_select_conversations(self, results: List[Dict]) -> List[Dict]:
        """Interactive menu to select conversations for extraction"""
        # Filter to relevant conversations
        relevant = [r for r in results if r['has_code'] or r['has_research'] or r['has_links']]
        
        if not relevant:
            print("No relevant conversations found.")
            return []
            
        print("\n" + "="*60)
        print("SELECT CONVERSATIONS FOR CONTENT EXTRACTION")
        print("="*60 + "\n")
        
        # Group by folder
        folders = defaultdict(list)
        for conv in relevant:
            folders[conv.get('folder', 'No Folder')].append(conv)
            
        # Display conversations
        conv_map = {}
        idx = 1
        
        for folder_name in sorted(folders.keys()):
            print(f"\n📁 {folder_name}")
            print("-" * 40)
            
            for conv in folders[folder_name]:
                type_emoji = ""
                if conv['has_code'] and conv['has_research']:
                    type_emoji = "🔬💻"
                elif conv['has_code']:
                    type_emoji = "💻"
                elif conv['has_research']:
                    type_emoji = "🔬"
                elif conv['has_links']:
                    type_emoji = "🔗"
                    
                print(f"{idx:3d}. {type_emoji} {conv['title'][:60]}")
                
                if conv.get('languages'):
                    print(f"      Languages: {', '.join(conv['languages'])}")
                if conv['has_code']:
                    print(f"      Code blocks: {conv['code_blocks']}")
                if conv['has_research']:
                    print(f"      Research keywords: {', '.join(conv.get('research_keywords', [])[:3])}")
                    
                conv_map[idx] = conv
                idx += 1
                
        print("\n" + "-"*60)
        print("\nCommands:")
        print("  - Enter numbers separated by commas (e.g., 1,3,5)")
        print("  - Enter ranges (e.g., 1-5)")
        print("  - Enter 'all' to select all")
        print("  - Enter 'code' to select all code conversations")
        print("  - Enter 'research' to select all research conversations")
        print("  - Enter 'quit' to cancel")
        
        while True:
            selection = input("\nYour selection: ").strip().lower()
            
            if selection == 'quit':
                return []
                
            selected_indices = set()
            
            if selection == 'all':
                selected_indices = set(conv_map.keys())
            elif selection == 'code':
                selected_indices = {k for k, v in conv_map.items() if v['has_code']}
            elif selection == 'research':
                selected_indices = {k for k, v in conv_map.items() if v['has_research']}
            else:
                # Parse numeric selections
                parts = selection.replace(' ', '').split(',')
                for part in parts:
                    if '-' in part:
                        # Range
                        try:
                            start, end = map(int, part.split('-'))
                            selected_indices.update(range(start, end + 1))
                        except:
                            print(f"Invalid range: {part}")
                    else:
                        # Single number
                        try:
                            num = int(part)
                            selected_indices.add(num)
                        except:
                            print(f"Invalid number: {part}")
                            
            # Validate indices
            selected_indices = {i for i in selected_indices if i in conv_map}
            
            if selected_indices:
                selected_convs = [conv_map[i] for i in sorted(selected_indices)]
                print(f"\nSelected {len(selected_convs)} conversations.")
                return selected_convs
            else:
                print("No valid conversations selected. Please try again.")
                
    def extract_conversation_content(self, conversation: Dict) -> Dict:
        """Extract detailed content from a conversation"""
        if self.mode == 'live':
            return self._extract_live_content(conversation)
        else:
            return self._extract_export_content(conversation)
            
    def _extract_live_content(self, conversation: Dict) -> Dict:
        """Extract content from live ChatGPT"""
        self.driver.get(conversation['url'])
        time.sleep(3)
        
        content = {
            'title': conversation['title'],
            'url': conversation['url'],
            'messages': [],
            'code_blocks': [],
            'links': [],
            'images': []
        }
        
        # Get all messages with better selectors
        message_selectors = [
            'div[data-message-author-role]',
            'div.group',
            'div.text-base',
            'div[class*="message"]'
        ]
        
        messages_found = False
        for selector in message_selectors:
            try:
                message_elems = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if message_elems:
                    messages_found = True
                    for elem in message_elems:
                        # Get role (user/assistant)
                        role = 'unknown'
                        try:
                            role_elem = elem.find_element(By.CSS_SELECTOR, '[data-message-author-role]')
                            role = role_elem.get_attribute('data-message-author-role')
                        except:
                            # Try to infer from content
                            if 'dark:bg-gray-800' in elem.get_attribute('class'):
                                role = 'user'
                            elif 'bg-gray-50' in elem.get_attribute('class'):
                                role = 'assistant'
                                
                        # Get message text
                        text = elem.text.strip()
                        
                        # Extract code blocks
                        code_blocks = elem.find_elements(By.CSS_SELECTOR, 'pre code')
                        for code_elem in code_blocks:
                            code_text = code_elem.text.strip()
                            language = 'unknown'
                            
                            # Try to get language from class
                            classes = code_elem.get_attribute('class') or ''
                            for cls in classes.split():
                                if cls.startswith('language-'):
                                    language = cls.replace('language-', '')
                                    break
                                    
                            content['code_blocks'].append({
                                'language': language,
                                'code': code_text,
                                'role': role
                            })
                            
                        # Extract links
                        links = elem.find_elements(By.CSS_SELECTOR, 'a[href]')
                        for link in links:
                            href = link.get_attribute('href')
                            if href and href.startswith('http'):
                                content['links'].append({
                                    'url': href,
                                    'text': link.text.strip(),
                                    'role': role
                                })
                                
                        content['messages'].append({
                            'role': role,
                            'content': text
                        })
                        
                    break
            except Exception as e:
                continue
                
        if not messages_found:
            print(f"Warning: Could not extract messages from {conversation['title']}")
            
        return content
        
    def _extract_export_content(self, conversation: Dict) -> Dict:
        """Extract content from exported conversation data"""
        # Read the full conversation from export
        with open(self.export_file, 'r', encoding='utf-8') as f:
            export_data = json.load(f)
            
        # Find the specific conversation
        target_conv = None
        for conv in export_data.get('conversations', []):
            if conv.get('id') == conversation['id']:
                target_conv = conv
                break
                
        if not target_conv:
            return None
            
        content = {
            'title': target_conv.get('title', 'Untitled'),
            'url': conversation['url'],
            'messages': [],
            'code_blocks': [],
            'links': [],
            'create_time': target_conv.get('create_time'),
            'update_time': target_conv.get('update_time')
        }
        
        # Extract messages from mapping
        mapping = target_conv.get('mapping', {})
        if mapping:
            # Sort messages by their position in conversation
            message_nodes = []
            for node_id, node in mapping.items():
                if node.get('message'):
                    message_nodes.append(node)
                    
            # Extract content from each message
            for node in message_nodes:
                message = node.get('message', {})
                if message.get('content', {}).get('content_type') == 'text':
                    role = message.get('author', {}).get('role', 'unknown')
                    parts = message.get('content', {}).get('parts', [])
                    full_text = '\n'.join(str(p) for p in parts)
                    
                    content['messages'].append({
                        'role': role,
                        'content': full_text
                    })
                    
                    # Extract code blocks
                    code_pattern = r'```(\w*)\n([\s\S]*?)```'
                    code_matches = re.findall(code_pattern, full_text)
                    for lang, code in code_matches:
                        content['code_blocks'].append({
                            'language': lang or 'unknown',
                            'code': code.strip(),
                            'role': role
                        })
                        
                    # Extract URLs
                    url_matches = re.findall(self.url_pattern, full_text)
                    for url in url_matches:
                        content['links'].append({
                            'url': url,
                            'text': '',
                            'role': role
                        })
                        
        return content
        
    def save_code_files(self, conversation: Dict, content: Dict, output_dir: str):
        """Save code blocks as individual files"""
        # Create directory for this conversation
        safe_title = re.sub(r'[^\w\s-]', '', conversation['title'])[:50]
        conv_dir = os.path.join(output_dir, f"{safe_title}_{conversation['id'][:8]}")
        os.makedirs(conv_dir, exist_ok=True)
        
        # Save metadata
        metadata = {
            'title': conversation['title'],
            'url': conversation['url'],
            'extracted_at': datetime.now().isoformat(),
            'total_messages': len(content['messages']),
            'total_code_blocks': len(content['code_blocks']),
            'languages': list(set(cb['language'] for cb in content['code_blocks'] if cb['language'] != 'unknown'))
        }
        
        with open(os.path.join(conv_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Save each code block
        language_counts = defaultdict(int)
        for i, code_block in enumerate(content['code_blocks']):
            language = code_block['language']
            if language == 'unknown':
                # Try to detect from content
                if 'def ' in code_block['code'] or 'import ' in code_block['code']:
                    language = 'python'
                elif 'function ' in code_block['code'] or 'const ' in code_block['code']:
                    language = 'javascript'
                    
            # Determine file extension
            extensions = {
                'python': '.py',
                'javascript': '.js',
                'java': '.java',
                'cpp': '.cpp',
                'c': '.c',
                'html': '.html',
                'css': '.css',
                'sql': '.sql',
                'go': '.go',
                'rust': '.rs',
                'typescript': '.ts',
                'jsx': '.jsx',
                'tsx': '.tsx',
                'bash': '.sh',
                'shell': '.sh',
                'yaml': '.yaml',
                'json': '.json',
                'xml': '.xml'
            }
            
            ext = extensions.get(language, '.txt')
            language_counts[language] += 1
            
            # Create filename
            filename = f"{language}_{language_counts[language]:02d}{ext}"
            filepath = os.path.join(conv_dir, filename)
            
            # Save code
            with open(filepath, 'w', encoding='utf-8') as f:
                # Add header comment
                if ext in ['.py', '.js', '.java', '.cpp', '.c', '.go', '.rs']:
                    f.write(f"# Extracted from: {conversation['title']}\n")
                    f.write(f"# Role: {code_block['role']}\n")
                    f.write(f"# Language: {language}\n")
                    f.write(f"# Extracted at: {datetime.now().isoformat()}\n\n")
                    
                f.write(code_block['code'])
                
        # Save full conversation as markdown
        with open(os.path.join(conv_dir, 'conversation.md'), 'w', encoding='utf-8') as f:
            f.write(f"# {conversation['title']}\n\n")
            f.write(f"URL: {conversation['url']}\n\n")
            f.write("---\n\n")
            
            for msg in content['messages']:
                role_emoji = "👤" if msg['role'] == 'user' else "🤖"
                f.write(f"## {role_emoji} {msg['role'].title()}\n\n")
                f.write(msg['content'])
                f.write("\n\n---\n\n")
                
        print(f"✅ Saved {len(content['code_blocks'])} code files to: {conv_dir}")
        
    def save_research_document(self, conversation: Dict, content: Dict, output_dir: str):
        """Save research conversation as formatted document"""
        safe_title = re.sub(r'[^\w\s-]', '', conversation['title'])[:50]
        
        # Create directory
        conv_dir = os.path.join(output_dir, f"{safe_title}_{conversation['id'][:8]}")
        os.makedirs(conv_dir, exist_ok=True)
        
        # Save as Markdown
        md_file = os.path.join(conv_dir, 'research.md')
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(f"# {conversation['title']}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Source:** [{conversation['url']}]({conversation['url']})\n\n")
            
            if content.get('links'):
                f.write("## References\n\n")
                for i, link in enumerate(content['links'], 1):
                    f.write(f"{i}. [{link.get('text', link['url'][:50])}]({link['url']})\n")
                f.write("\n---\n\n")
                
            f.write("## Conversation\n\n")
            
            for msg in content['messages']:
                if msg['role'] == 'user':
                    f.write(f"### 🔍 Question\n\n")
                else:
                    f.write(f"### 📊 Analysis\n\n")
                    
                f.write(msg['content'])
                f.write("\n\n")
                
        # Try to generate PDF if available
        if REPORTLAB_AVAILABLE:
            self._generate_research_pdf(conversation, content, conv_dir)
        else:
            print(f"✅ Saved research document as Markdown to: {conv_dir}")
            print("   (Install reportlab for PDF generation: pip install reportlab)")
            
    def _generate_research_pdf(self, conversation: Dict, content: Dict, output_dir: str):
        """Generate PDF from research content"""
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
        
        pdf_file = os.path.join(output_dir, 'research.pdf')
        doc = SimpleDocTemplate(pdf_file, pagesize=letter)
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        # Add title
        elements.append(Paragraph(conversation['title'], title_style))
        elements.append(Spacer(1, 12))
        
        # Add metadata
        meta_text = f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
        meta_text += f"<b>Source:</b> ChatGPT Conversation"
        elements.append(Paragraph(meta_text, styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Add messages
        for msg in content['messages']:
            if msg['role'] == 'user':
                elements.append(Paragraph("<b>Question:</b>", styles['Heading2']))
            else:
                elements.append(Paragraph("<b>Analysis:</b>", styles['Heading2']))
                
            # Clean and format content for PDF
            text = msg['content'].replace('\n\n', '<br/><br/>')
            text = text.replace('```', '<pre>')  # Simple code block handling
            
            elements.append(Paragraph(text, styles['BodyText']))
            elements.append(Spacer(1, 20))
            
        # Build PDF
        try:
            doc.build(elements)
            print(f"✅ Saved research document as PDF to: {pdf_file}")
        except Exception as e:
            print(f"⚠️  Could not generate PDF: {e}")
            
    def extract_selected_content(self, selected_conversations: List[Dict]):
        """Extract content from selected conversations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"chatgpt_extracts_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        code_dir = os.path.join(output_dir, "code")
        research_dir = os.path.join(output_dir, "research")
        
        print(f"\nExtracting content to: {output_dir}")
        print("="*60)
        
        for i, conv in enumerate(selected_conversations, 1):
            print(f"\n[{i}/{len(selected_conversations)}] Processing: {conv['title'][:60]}...")
            
            # Extract content
            content = self.extract_conversation_content(conv)
            
            if not content or not content.get('messages'):
                print("   ⚠️  No content extracted")
                continue
                
            # Save based on type
            if conv['has_code'] and content.get('code_blocks'):
                os.makedirs(code_dir, exist_ok=True)
                self.save_code_files(conv, content, code_dir)
                
            if conv['has_research']:
                os.makedirs(research_dir, exist_ok=True)
                self.save_research_document(conv, content, research_dir)
                
        print("\n" + "="*60)
        print(f"✅ Extraction complete! Files saved to: {output_dir}")
        
        # Create summary report
        summary_file = os.path.join(output_dir, 'extraction_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"ChatGPT Content Extraction Summary\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Total conversations processed: {len(selected_conversations)}\n")
            f.write(f"Output directory: {output_dir}\n\n")
            
            code_convs = [c for c in selected_conversations if c['has_code']]
            research_convs = [c for c in selected_conversations if c['has_research']]
            
            if code_convs:
                f.write(f"\nCode Conversations ({len(code_convs)}):\n")
                for conv in code_convs:
                    f.write(f"  - {conv['title']}\n")
                    if conv.get('languages'):
                        f.write(f"    Languages: {', '.join(conv['languages'])}\n")
                        
            if research_convs:
                f.write(f"\nResearch Conversations ({len(research_convs)}):\n")
                for conv in research_convs:
                    f.write(f"  - {conv['title']}\n")
        
    def run(self):
        """Main execution method"""
        if self.mode == 'export':
            if not self.export_file or not os.path.exists(self.export_file):
                raise Exception(f"Export file not found: {self.export_file}")
                
            print(f"Analyzing export file: {self.export_file}")
            results = self.analyze_export_file()
            
        elif self.mode == 'live':
            if not SELENIUM_AVAILABLE:
                raise Exception("Selenium is required for live mode. Install with: pip install selenium")
                
            print("Starting live ChatGPT analysis...")
            self.setup_driver()
            
            try:
                self.login()
                print("Successfully logged in to ChatGPT")
                
                # Extract conversations
                conversations = self.extract_live_conversations()
                print(f"Found {len(conversations)} conversations")
                
                # Analyze each conversation
                results = []
                for i, conv in enumerate(conversations):
                    print(f"Analyzing {i+1}/{len(conversations)}: {conv['title'][:50]}...")
                    analysis = self.analyze_live_conversation(conv)
                    
                    # Optionally rename
                    if self.rename_chats and (analysis['has_code'] or analysis['has_research']):
                        self.rename_live_conversation(conv, analysis)
                        
                    results.append(analysis)
                    
            finally:
                if self.driver and not self.extract_content:
                    self.driver.quit()
                    
        else:
            raise Exception(f"Unknown mode: {self.mode}")
            
        # Handle content extraction if requested
        if self.extract_content and results:
            selected = self.interactive_select_conversations(results)
            if selected:
                self.extract_selected_content(selected)
                
        # Clean up driver if still open
        if self.driver:
            self.driver.quit()
            
        return results
        

def get_user_input(prompt: str, options: List[str] = None, default: str = None) -> str:
    """Get user input with optional validation"""
    if options:
        print(f"\n{prompt}")
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        if default:
            print(f"  (Default: {default})")
            
        while True:
            choice = input("\nYour choice (number or value): ").strip()
            if not choice and default:
                return default
            
            # Check if it's a number
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx]
            except ValueError:
                # Check if it's a direct value
                if choice in options:
                    return choice
                    
            print("Invalid choice. Please try again.")
    else:
        value = input(f"\n{prompt} ").strip()
        return value if value else default


def interactive_setup():
    """Interactive configuration setup"""
    print("\n" + "="*60)
    print("🤖 ChatGPT Universal Analyzer - Interactive Setup")
    print("="*60)
    print("\nWelcome! I'll help you configure the analyzer.")
    print("Just answer a few questions to get started.\n")
    
    # Check available features
    if not SELENIUM_AVAILABLE:
        print("ℹ️  Note: Selenium not installed. Live browser features unavailable.")
        print("   To enable: pip install selenium\n")
    if not REPORTLAB_AVAILABLE:
        print("ℹ️  Note: ReportLab not installed. PDF generation unavailable.")
        print("   To enable: pip install reportlab\n")
    
    config = {}
    
    # Step 1: Choose mode
    print("STEP 1: Choose Analysis Mode")
    print("-" * 30)
    mode_options = ["Export mode (analyze downloaded JSON file)", "Live mode (browse ChatGPT in real-time)"]
    if not SELENIUM_AVAILABLE:
        mode_options = ["Export mode (analyze downloaded JSON file)"]
        print("(Live mode requires Selenium)")
        
    mode_choice = get_user_input("How would you like to analyze your conversations?", mode_options)
    config['mode'] = 'export' if 'Export' in mode_choice else 'live'
    
    # Step 2: Mode-specific configuration
    if config['mode'] == 'export':
        print("\nSTEP 2: Locate Your Export File")
        print("-" * 30)
        print("You'll need your ChatGPT conversations.json file.")
        print("To get this file:")
        print("  1. Go to ChatGPT → Settings → Data Controls")
        print("  2. Click 'Export data' → Download via email")
        print("  3. Extract the ZIP and find 'conversations.json'")
        
        while True:
            file_path = input("\nPath to conversations.json (or drag & drop file here): ").strip()
            # Remove quotes if present (from drag & drop)
            file_path = file_path.strip('"').strip("'")
            
            if os.path.exists(file_path):
                config['file'] = file_path
                break
            else:
                print(f"❌ File not found: {file_path}")
                retry = get_user_input("Would you like to:", ["Try another path", "Exit"])
                if "Exit" in retry:
                    return None
                    
    else:  # Live mode
        print("\nSTEP 2: ChatGPT Login Configuration")
        print("-" * 30)
        print("I'll need your ChatGPT credentials to browse your conversations.")
        print("These will only be used for this session and not stored.")
        
        if not os.getenv('CHATGPT_EMAIL'):
            email = input("\nChatGPT email: ").strip()
            os.environ['CHATGPT_EMAIL'] = email
        else:
            print(f"\n✓ Using email from environment: {os.getenv('CHATGPT_EMAIL')}")
            
        if not os.getenv('CHATGPT_PASSWORD'):
            import getpass
            password = getpass.getpass("ChatGPT password: ")
            os.environ['CHATGPT_PASSWORD'] = password
        else:
            print("✓ Using password from environment")
            
        # Ask about renaming
        rename_choice = get_user_input(
            "\nWould you like to rename conversations based on their content?",
            ["Yes - Add [Code], [Research] tags", "No - Just analyze"],
            "No - Just analyze"
        )
        config['rename'] = 'Yes' in rename_choice
        
    # Step 3: Choose action
    print("\nSTEP 3: Choose Your Action")
    print("-" * 30)
    action = get_user_input(
        "What would you like to do?",
        [
            "Analyze and generate reports",
            "Extract code and research documents",
            "Both - Analyze first, then extract"
        ],
        "Analyze and generate reports"
    )
    
    if "Extract" in action:
        config['extract'] = True
        config['output'] = None  # Don't generate reports when extracting
    elif "Both" in action:
        config['extract'] = True
        config['output'] = 'all'
    else:
        config['extract'] = False
        
        # Step 4: Choose output format
        print("\nSTEP 4: Choose Output Format")
        print("-" * 30)
        output_choice = get_user_input(
            "Which format(s) would you like for the analysis report?",
            [
                "All formats (CSV, JSON, Markdown, HTML)",
                "CSV only (Excel-compatible)",
                "JSON only (machine-readable)",
                "Markdown only (readable text)",
                "HTML only (interactive web page)"
            ],
            "All formats (CSV, JSON, Markdown, HTML)"
        )
        
        if "All formats" in output_choice:
            config['output'] = 'all'
        elif "CSV" in output_choice:
            config['output'] = 'csv'
        elif "JSON" in output_choice:
            config['output'] = 'json'
        elif "Markdown" in output_choice:
            config['output'] = 'markdown'
        elif "HTML" in output_choice:
            config['output'] = 'html'
            
    # Confirm configuration
    print("\n" + "="*60)
    print("📋 Configuration Summary")
    print("="*60)
    print(f"Mode: {config['mode'].upper()}")
    if config['mode'] == 'export':
        print(f"File: {config.get('file', 'N/A')}")
    else:
        print(f"Rename chats: {'Yes' if config.get('rename') else 'No'}")
    print(f"Extract content: {'Yes' if config.get('extract') else 'No'}")
    if not config.get('extract'):
        print(f"Output format: {config.get('output', 'all').upper()}")
        
    confirm = get_user_input("\nReady to start?", ["Yes, let's go!", "No, start over"], "Yes, let's go!")
    
    if "No" in confirm:
        return interactive_setup()  # Restart
        
    return config


def main():
    """Main entry point with interactive configuration"""
    try:
        # Get configuration interactively
        config = interactive_setup()
        
        if not config:
            print("\n👋 Goodbye!")
            return 0
            
        # Create analyzer with configuration
        analyzer = ChatGPTUniversalAnalyzer(
            mode=config['mode'],
            export_file=config.get('file'),
            rename_chats=config.get('rename', False),
            extract_content=config.get('extract', False)
        )
        
        # Run analysis
        print("\n" + "="*60)
        print("🚀 Starting Analysis")
        print("="*60 + "\n")
        
        results = analyzer.run()
        
        if results and not config.get('extract'):
            # Save results only if not extracting
            if config.get('output'):
                analyzer.save_results(results, config['output'])
            
            # Print summary
            print("\n" + "="*60)
            print("✅ ANALYSIS COMPLETE")
            print("="*60)
            
            relevant = [r for r in results if r['has_code'] or r['has_research'] or r['has_links']]
            print(f"\nTotal conversations analyzed: {len(results)}")
            print(f"Relevant conversations found: {len(relevant)}")
            
            if config['mode'] == 'live' and config.get('rename'):
                renamed = sum(1 for r in relevant if r['has_code'] or r['has_research'])
                print(f"Conversations renamed: {renamed}")
                
            # Ask if user wants to extract content now
            if relevant and not config.get('extract'):
                print("\n" + "-"*60)
                extract_now = get_user_input(
                    "Would you like to extract code/research from these conversations now?",
                    ["Yes", "No"],
                    "No"
                )
                if extract_now == "Yes":
                    selected = analyzer.interactive_select_conversations(results)
                    if selected:
                        analyzer.extract_selected_content(selected)
                        
        elif not results:
            print("\n❌ No conversations found to analyze")
            
        print("\n✨ All done! Thank you for using ChatGPT Analyzer.")
        
    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled. Goodbye!")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("- Make sure your conversations.json file is valid")
        print("- For live mode, check your credentials")
        print("- Ensure you have the required packages installed")
        
        if '--debug' in sys.argv:
            import traceback
            traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())
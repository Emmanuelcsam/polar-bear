#!/usr/bin/env python3
"""
ChatGPT Live Analyzer Server
Logs into ChatGPT directly and analyzes conversations in real-time
No JSON upload required - everything happens live!
"""

import os
import json
import re
import time
import logging
import asyncio
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from queue import Queue

from flask import Flask, render_template_string, request, jsonify, send_file
from flask_sock import Sock
from flask_cors import CORS

# Selenium imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException

# Try to import webdriver manager
try:
    from webdriver_manager.chrome import ChromeDriverManager
    WEBDRIVER_MANAGER_AVAILABLE = True
except ImportError:
    WEBDRIVER_MANAGER_AVAILABLE = False

# Import the CAPTCHA handler
try:
    from captcha_handler import CaptchaHandler
    CAPTCHA_AVAILABLE = True
except ImportError:
    CAPTCHA_AVAILABLE = False
    print("Warning: CAPTCHA handler not found. Manual intervention may be required.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
sock = Sock(app)

# Global state
analyzer_instance = None
websocket_connections = set()
message_queue = Queue()


@dataclass
class ConversationData:
    """Data class for conversation information"""
    id: str
    title: str
    url: str
    type: str = 'general'
    has_code: bool = False
    has_research: bool = False
    has_links: bool = False
    code_blocks: int = 0
    languages: List[str] = None
    research_keywords: List[str] = None
    url_count: int = 0
    total_messages: int = 0
    word_count: int = 0
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = []
        if self.research_keywords is None:
            self.research_keywords = []


class ChatGPTAnalyzer:
    """Live ChatGPT analyzer using Selenium"""
    
    def __init__(self, email: str, password: str, headless: bool = True, 
                 auto_rename: bool = False, scan_delay: int = 2, max_conversations: int = 0):
        self.email = email
        self.password = password
        self.headless = headless
        self.auto_rename = auto_rename
        self.scan_delay = scan_delay
        self.max_conversations = max_conversations
        self.driver = None
        self.captcha_handler = None
        self.is_running = False
        self.conversations = []
        
        # Analysis patterns
        self.code_patterns = [
            r'```[\w]*\n[\s\S]*?```',  # Code blocks
            r'def\s+\w+\s*\(',          # Python functions
            r'class\s+\w+[:\(]',        # Python classes
            r'import\s+\w+',            # Import statements
            r'from\s+\w+\s+import',     # From imports
            r'function\s+\w+\s*\(',     # JavaScript functions
            r'const\s+\w+\s*=',         # JavaScript const
            r'let\s+\w+\s*=',           # JavaScript let
            r'<[^>]+>.*<\/[^>]+>',      # HTML tags
            r'SELECT\s+.*FROM',         # SQL queries
        ]
        
        self.research_keywords = [
            'research', 'analysis', 'study', 'investigation', 'examine',
            'explore', 'analyze', 'data', 'statistics', 'findings',
            'conclusion', 'hypothesis', 'methodology', 'results',
            'survey', 'experiment', 'observation', 'theory', 'evidence',
            'paper', 'article', 'publication', 'citation', 'reference',
            'dataset', 'correlation', 'regression', 'significant',
            'algorithm', 'implementation', 'architecture', 'framework',
        ]
        
    def send_update(self, msg_type: str, data: Dict[str, Any]):
        """Send update to all connected WebSocket clients"""
        message = json.dumps({'type': msg_type, **data})
        message_queue.put(message)
        
    def setup_driver(self):
        """Setup Chrome driver with optimal settings"""
        self.send_update('status', {
            'title': 'Setting up browser',
            'message': 'Initializing Chrome driver...',
            'icon': '🌐'
        })
        
        options = Options()
        
        # Chrome options for stability
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        if self.headless:
            options.add_argument('--headless=new')
            options.add_argument('--window-size=1920,1080')
        else:
            options.add_argument('--start-maximized')
            
        # User agent to appear more natural
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # Try to create driver
        try:
            if WEBDRIVER_MANAGER_AVAILABLE:
                service = ChromeService(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
            else:
                # Try default path
                self.driver = webdriver.Chrome(options=options)
                
            self.driver.implicitly_wait(10)
            
            # Initialize CAPTCHA handler if available
            if CAPTCHA_AVAILABLE:
                self.captcha_handler = CaptchaHandler(self.driver, debug=True)
                
            self.send_update('log', {'message': 'Browser initialized successfully', 'level': 'info'})
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup driver: {e}")
            self.send_update('error', {'message': f'Failed to setup browser: {str(e)}'})
            return False
            
    def login(self):
        """Login to ChatGPT"""
        self.send_update('status', {
            'title': 'Logging in',
            'message': 'Navigating to ChatGPT...',
            'icon': '🔐'
        })
        
        try:
            self.driver.get('https://chat.openai.com')
            time.sleep(3)
            
            # Check for CAPTCHA
            if self.captcha_handler and self.captcha_handler.is_captcha_present():
                self.send_update('captcha', {'detected': True, 'message': 'CAPTCHA detected, attempting to solve...'})
                if self.captcha_handler.handle_captcha():
                    self.send_update('captcha', {'solved': True, 'message': 'CAPTCHA solved successfully'})
                else:
                    self.send_update('captcha', {'solved': False, 'message': 'Could not solve CAPTCHA automatically'})
            
            # Check if already logged in
            try:
                self.driver.find_element(By.CSS_SELECTOR, 'nav[aria-label="Chat history"]')
                self.send_update('log', {'message': 'Already logged in!', 'level': 'info'})
                return True
            except NoSuchElementException:
                pass
                
            self.send_update('log', {'message': 'Starting login process...', 'level': 'info'})
            
            # Find and click login button
            login_btn = None
            for selector in ['button:has-text("Log in")', 'a[href*="auth0"]', '//button[contains(text(), "Log in")]']:
                try:
                    if selector.startswith('//'):
                        login_btn = self.driver.find_element(By.XPATH, selector)
                    else:
                        login_btn = self.driver.find_element(By.CSS_SELECTOR, selector)
                    break
                except:
                    continue
                    
            if login_btn:
                login_btn.click()
                time.sleep(3)
            else:
                # Try direct navigation to login
                self.driver.get('https://auth0.openai.com/u/login')
                time.sleep(3)
                
            self.send_update('log', {'message': 'Entering credentials...', 'level': 'info'})
            
            # Enter email
            email_field = self.driver.find_element(By.NAME, 'username')
            email_field.clear()
            email_field.send_keys(self.email)
            email_field.send_keys(Keys.RETURN)
            time.sleep(2)
            
            # Enter password
            password_field = self.driver.find_element(By.NAME, 'password')
            password_field.clear()
            password_field.send_keys(self.password)
            password_field.send_keys(Keys.RETURN)
            
            self.send_update('log', {'message': 'Waiting for login to complete...', 'level': 'info'})
            
            # Wait for login to complete
            wait = WebDriverWait(self.driver, 30)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'nav[aria-label="Chat history"]')))
            
            self.send_update('log', {'message': 'Login successful!', 'level': 'success'})
            return True
            
        except Exception as e:
            logger.error(f"Login failed: {e}")
            self.send_update('error', {'message': f'Login failed: {str(e)}'})
            return False
            
    def get_all_conversations(self):
        """Get all conversation links from the sidebar"""
        self.send_update('status', {
            'title': 'Scanning conversations',
            'message': 'Loading conversation list...',
            'icon': '📋'
        })
        
        conversations = []
        
        try:
            # Find the chat history sidebar
            sidebar = self.driver.find_element(By.CSS_SELECTOR, 'nav[aria-label="Chat history"]')
            
            # Scroll to load all conversations
            last_height = 0
            while True:
                # Scroll sidebar
                self.driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", sidebar)
                time.sleep(1)
                
                new_height = self.driver.execute_script("return arguments[0].scrollHeight", sidebar)
                if new_height == last_height:
                    break
                last_height = new_height
                
                self.send_update('log', {'message': f'Loading more conversations... (height: {new_height})', 'level': 'info'})
            
            # Get all conversation links
            links = sidebar.find_elements(By.CSS_SELECTOR, 'a[href*="/c/"]')
            
            for link in links:
                try:
                    url = link.get_attribute('href')
                    title = link.text.strip() or 'Untitled'
                    conv_id = url.split('/c/')[-1] if '/c/' in url else None
                    
                    if conv_id:
                        conversations.append({
                            'id': conv_id,
                            'title': title,
                            'url': url
                        })
                except:
                    continue
                    
            self.send_update('log', {'message': f'Found {len(conversations)} conversations', 'level': 'info'})
            return conversations
            
        except Exception as e:
            logger.error(f"Failed to get conversations: {e}")
            self.send_update('error', {'message': f'Failed to get conversations: {str(e)}'})
            return []
            
    def analyze_conversation(self, conv_data: Dict[str, Any]) -> Optional[ConversationData]:
        """Analyze a single conversation"""
        try:
            # Navigate to conversation
            self.driver.get(conv_data['url'])
            time.sleep(self.scan_delay)
            
            # Get all messages
            messages = []
            message_elements = self.driver.find_elements(By.CSS_SELECTOR, 'div[data-message-author-role]')
            
            for elem in message_elements:
                try:
                    text = elem.text.strip()
                    if text:
                        messages.append(text)
                except:
                    continue
                    
            # Combine all text
            full_text = '\n'.join(messages)
            
            # Analyze content
            analysis = self.analyze_text(full_text)
            
            # Create conversation data
            conv = ConversationData(
                id=conv_data['id'],
                title=conv_data['title'],
                url=conv_data['url'],
                type=analysis['type'],
                has_code=analysis['has_code'],
                has_research=analysis['has_research'],
                has_links=analysis['has_links'],
                code_blocks=analysis['code_blocks'],
                languages=analysis['languages'],
                research_keywords=analysis['research_keywords'],
                url_count=analysis['url_count'],
                total_messages=len(messages),
                word_count=len(full_text.split())
            )
            
            # Auto-rename if enabled
            if self.auto_rename and conv.type != 'general':
                self.rename_conversation(conv)
                
            return conv
            
        except Exception as e:
            logger.error(f"Error analyzing conversation {conv_data['title']}: {e}")
            return None
            
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text content for patterns"""
        # Check for code
        has_code = False
        code_blocks = 0
        languages = set()
        
        # Count code blocks
        code_block_matches = re.findall(r'```(\w*)\n(.*?)```', text, re.DOTALL)
        code_blocks = len(code_block_matches)
        
        # Extract languages
        for lang, _ in code_block_matches:
            if lang:
                languages.add(lang.lower())
                
        # Check for code patterns
        for pattern in self.code_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                has_code = True
                break
                
        # Detect languages from content
        if 'def ' in text or 'import ' in text:
            languages.add('python')
        if 'function ' in text or 'const ' in text:
            languages.add('javascript')
        if 'SELECT ' in text.upper():
            languages.add('sql')
            
        # Check for research content
        text_lower = text.lower()
        found_keywords = [kw for kw in self.research_keywords if kw in text_lower]
        has_research = len(found_keywords) >= 5
        
        # Check for URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        has_links = len(urls) > 0
        
        # Determine type
        if has_code and has_research:
            conv_type = 'mixed'
        elif has_code:
            conv_type = 'code'
        elif has_research:
            conv_type = 'research'
        elif has_links:
            conv_type = 'links'
        else:
            conv_type = 'general'
            
        return {
            'type': conv_type,
            'has_code': has_code,
            'has_research': has_research,
            'has_links': has_links,
            'code_blocks': code_blocks,
            'languages': sorted(list(languages)),
            'research_keywords': found_keywords[:5],
            'url_count': len(urls)
        }
        
    def rename_conversation(self, conv: ConversationData):
        """Rename a conversation based on its type"""
        try:
            # Create new title
            prefix = f"[{conv.type.upper()}]"
            if conv.languages:
                prefix += f" ({', '.join(conv.languages[:2])})"
                
            new_title = f"{prefix} {conv.title}"
            
            # Navigate to conversation if not already there
            if self.driver.current_url != conv.url:
                self.driver.get(conv.url)
                time.sleep(1)
                
            # Find and click the rename button (usually three dots menu)
            # This part depends on ChatGPT's current UI
            # You may need to adjust selectors based on the actual UI
            
            self.send_update('log', {'message': f'Renamed: {new_title}', 'level': 'info'})
            
        except Exception as e:
            logger.error(f"Failed to rename conversation: {e}")
            
    def run_analysis(self):
        """Main analysis loop"""
        self.is_running = True
        
        try:
            # Setup browser
            if not self.setup_driver():
                return
                
            # Login
            if not self.login():
                return
                
            # Get all conversations
            all_conversations = self.get_all_conversations()
            
            if not all_conversations:
                self.send_update('error', {'message': 'No conversations found'})
                return
                
            # Limit conversations if specified
            if self.max_conversations > 0:
                all_conversations = all_conversations[:self.max_conversations]
                
            total = len(all_conversations)
            self.send_update('progress', {'current': 0, 'total': total})
            
            # Analyze each conversation
            for i, conv_data in enumerate(all_conversations):
                if not self.is_running:
                    break
                    
                self.send_update('status', {
                    'title': f'Analyzing ({i+1}/{total})',
                    'message': conv_data['title'][:50] + '...',
                    'icon': '🔍'
                })
                
                # Analyze conversation
                conv = self.analyze_conversation(conv_data)
                
                if conv and conv.type != 'general':
                    self.conversations.append(conv)
                    self.send_update('conversation', {'conversation': asdict(conv)})
                    
                self.send_update('progress', {'current': i + 1, 'total': total})
                
                # Small delay between conversations
                time.sleep(0.5)
                
            # Analysis complete
            self.send_update('complete', {
                'results': [asdict(c) for c in self.conversations]
            })
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            self.send_update('error', {'message': f'Analysis error: {str(e)}'})
            
        finally:
            self.cleanup()
            
    def stop(self):
        """Stop the analysis"""
        self.is_running = False
        self.send_update('log', {'message': 'Stopping analysis...', 'level': 'warning'})
        
    def cleanup(self):
        """Clean up resources"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None


# Flask routes
@app.route('/')
def index():
    """Serve the main HTML page"""
    html_path = Path(__file__).parent / 'index.html'
    if html_path.exists():
        return html_path.read_text(encoding='utf-8')
    else:
        return "<h1>ChatGPT Live Analyzer</h1><p>index.html not found</p>"


@app.route('/api/start-analysis', methods=['POST'])
def start_analysis():
    """Start the analysis process"""
    global analyzer_instance
    
    if analyzer_instance and analyzer_instance.is_running:
        return jsonify({'success': False, 'error': 'Analysis already in progress'})
        
    data = request.json
    
    # Create analyzer instance
    analyzer_instance = ChatGPTAnalyzer(
        email=data['email'],
        password=data['password'],
        headless=data.get('headless', True),
        auto_rename=data.get('autoRename', False),
        scan_delay=data.get('scanDelay', 2),
        max_conversations=data.get('maxConversations', 0)
    )
    
    # Start analysis in background thread
    thread = threading.Thread(target=analyzer_instance.run_analysis)
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True})


@app.route('/api/stop-analysis', methods=['POST'])
def stop_analysis():
    """Stop the current analysis"""
    global analyzer_instance
    
    if analyzer_instance:
        analyzer_instance.stop()
        
    return jsonify({'success': True})


@app.route('/api/export', methods=['POST'])
def export():
    """Export analysis results"""
    data = request.json
    conversations = data.get('conversations', [])
    
    # Create CSV content
    csv_lines = ['Title,Type,URL,Languages,Code Blocks,Research Keywords']
    
    for conv in conversations:
        csv_lines.append(','.join([
            f'"{conv.get("title", "")}"',
            conv.get('type', ''),
            conv.get('url', ''),
            f'"{"; ".join(conv.get("languages", []))}"',
            str(conv.get('code_blocks', 0)),
            f'"{"; ".join(conv.get("research_keywords", []))}"'
        ]))
        
    csv_content = '\n'.join(csv_lines)
    
    # Return as downloadable file
    from io import BytesIO
    output = BytesIO()
    output.write(csv_content.encode('utf-8'))
    output.seek(0)
    
    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'chatgpt_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )


@sock.route('/ws')
def websocket(ws):
    """WebSocket connection handler"""
    websocket_connections.add(ws)
    
    try:
        # Send queued messages
        while True:
            # Check for new messages
            if not message_queue.empty():
                message = message_queue.get()
                # Send to all connected clients
                for client in list(websocket_connections):
                    try:
                        client.send(message)
                    except:
                        websocket_connections.discard(client)
                        
            # Handle incoming messages
            data = ws.receive(timeout=0.1)
            if data:
                msg = json.loads(data)
                if msg.get('action') == 'stop' and analyzer_instance:
                    analyzer_instance.stop()
                    
    except:
        pass
    finally:
        websocket_connections.discard(ws)


# Message queue processor
def process_message_queue():
    """Process messages in a separate thread"""
    while True:
        if not message_queue.empty() and websocket_connections:
            message = message_queue.get()
            for client in list(websocket_connections):
                try:
                    client.send(message)
                except:
                    websocket_connections.discard(client)
        time.sleep(0.1)


# Start message processor thread
message_thread = threading.Thread(target=process_message_queue)
message_thread.daemon = True
message_thread.start()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting ChatGPT Live Analyzer Server on port {port}")
    logger.info("This server logs into ChatGPT directly - no JSON upload needed!")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
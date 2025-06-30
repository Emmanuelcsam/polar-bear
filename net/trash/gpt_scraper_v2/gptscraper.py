#!/usr/bin/env python3
"""
ChatGPT Universal Analyzer - Production Version
Complete solution for analyzing and extracting ChatGPT conversations
Supports both export mode (JSON) and live mode (browser automation)
"""

import os
import sys
import time
import csv
import json
import re
import logging
import argparse
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Optional, Set, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from current directory
    load_dotenv('.env')
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Configure logging based on environment
log_level = os.getenv('LOG_LEVEL', 'INFO')
log_file = os.getenv('LOG_FILE', None)

logging_config = {
    'level': getattr(logging, log_level.upper(), logging.INFO),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

if log_file:
    logging_config['filename'] = log_file
    logging_config['filemode'] = 'a'

logging.basicConfig(**logging_config)
logger = logging.getLogger(__name__)

# Try importing optional dependencies
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import NoSuchElementException, TimeoutException
    SELENIUM_AVAILABLE = True

    try:
        from webdriver_manager.chrome import ChromeDriverManager
        from webdriver_manager.core.os_manager import ChromeType
        WEBDRIVER_MANAGER_AVAILABLE = True
    except ImportError:
        WEBDRIVER_MANAGER_AVAILABLE = False

except ImportError:
    SELENIUM_AVAILABLE = False
    WEBDRIVER_MANAGER_AVAILABLE = False
    logger.warning("Selenium not installed. Live mode disabled.")

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.info("ReportLab not installed. PDF generation disabled.")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger.info("BeautifulSoup not installed. Some features may be limited.")

# Import CAPTCHA handler if available
try:
    from captcha_handler import integrate_with_scraper
    CAPTCHA_HANDLER_AVAILABLE = True
except ImportError:
    CAPTCHA_HANDLER_AVAILABLE = False
    logger.info("CAPTCHA handler not available. Manual intervention may be required.")

# Optional: Email notifications
try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    logger.info("Email support not available.")

# Configure proxy if set
if os.getenv('HTTP_PROXY') or os.getenv('HTTPS_PROXY'):
    proxies = {}
    if os.getenv('HTTP_PROXY'):
        proxies['http'] = os.getenv('HTTP_PROXY')
    if os.getenv('HTTPS_PROXY'):
        proxies['https'] = os.getenv('HTTPS_PROXY')
    if os.getenv('NO_PROXY'):
        os.environ['NO_PROXY'] = os.getenv('NO_PROXY')
    logger.info(f"Proxy configured: {proxies}")
else:
    proxies = None


class ConversationType(Enum):
    """Types of conversations"""
    CODE = "code"
    RESEARCH = "research"
    MIXED = "mixed"
    LINKS = "links"
    GENERAL = "general"


@dataclass
class ConversationMetadata:
    """Store conversation metadata"""
    id: str
    title: str
    url: str
    folder: str = "No Folder"
    create_time: Optional[str] = None
    update_time: Optional[str] = None
    total_messages: int = 0
    conversation_type: ConversationType = ConversationType.GENERAL
    has_code: bool = False
    has_research: bool = False
    has_links: bool = False
    code_blocks: int = 0
    languages: List[str] = field(default_factory=list)
    research_keywords: List[str] = field(default_factory=list)
    url_count: int = 0
    word_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'title': self.title,
            'url': self.url,
            'folder': self.folder,
            'create_time': self.create_time,
            'update_time': self.update_time,
            'total_messages': self.total_messages,
            'detected_type': self.conversation_type.value,
            'has_code': self.has_code,
            'has_research': self.has_research,
            'has_links': self.has_links,
            'code_blocks': self.code_blocks,
            'languages': self.languages,
            'research_keywords': self.research_keywords,
            'url_count': self.url_count,
            'word_count': self.word_count
        }


class ChatGPTAnalyzer:
    """Production-ready ChatGPT conversation analyzer"""

    def __init__(self,
                 mode: str = 'export',
                 export_file: Optional[str] = None,
                 output_dir: str = None,
                 cache_dir: str = '.cache',
                 use_cache: bool = None):
        """
        Initialize analyzer

        Args:
            mode: 'export' or 'live'
            export_file: Path to conversations.json for export mode
            output_dir: Directory for output files
            cache_dir: Directory for caching
            use_cache: Whether to use caching for performance
        """
        self.mode = mode

        # Load defaults from environment
        self.export_file = export_file or os.getenv('EXPORT_FILE', 'conversations.json')
        self.output_dir = output_dir or os.getenv('OUTPUT_DIR', 'chatgpt_analysis')
        self.cache_dir = cache_dir
        self.use_cache = use_cache if use_cache is not None else os.getenv('USE_CACHE', 'true').lower() == 'true'
        self.driver = None

        # Performance settings from environment
        self.max_workers = int(os.getenv('MAX_WORKERS', '8'))
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '50'))
        self.request_timeout = int(os.getenv('REQUEST_TIMEOUT', '30'))

        # CAPTCHA settings
        self.captcha_max_attempts = int(os.getenv('CAPTCHA_MAX_ATTEMPTS', '5'))
        self.captcha_timeout = int(os.getenv('CAPTCHA_TIMEOUT', '30'))

        # Screenshot settings
        self.screenshot_on_error = os.getenv('SCREENSHOT_ON_ERROR', 'true').lower() == 'true'
        self.save_debug_html = os.getenv('SAVE_DEBUG_HTML', 'false').lower() == 'true'

        # Email notification settings
        self.smtp_server = os.getenv('SMTP_SERVER')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = os.getenv('SMTP_USERNAME')
        self.smtp_password = os.getenv('SMTP_PASSWORD')
        self.notify_email = os.getenv('NOTIFY_EMAIL')

        # API keys for future features
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Enhanced patterns for better detection
        self.code_patterns = {
            'python': [
                r'def\s+\w+\s*\([^)]*\)\s*:',
                r'class\s+\w+(?:\s*\([^)]*\))?\s*:',
                r'import\s+[\w\.]+(?:\s+as\s+\w+)?',
                r'from\s+[\w\.]+\s+import',
                r'if\s+__name__\s*==\s*["\']__main__["\']',
                r'async\s+def\s+\w+',
                r'@\w+(?:\.\w+)*(?:\([^)]*\))?',  # Decorators
                r'lambda\s+[\w\s,]*:',
                r'\bprint\s*\(',
                r'\bfor\s+\w+\s+in\s+',
                r'\bwhile\s+.+:',
                r'\btry\s*:\s*\n',
                r'\bexcept\s+\w+',
                r'\bwith\s+.+\s+as\s+\w+',
            ],
            'javascript': [
                r'function\s+\w+\s*\([^)]*\)\s*{',
                r'const\s+\w+\s*=',
                r'let\s+\w+\s*=',
                r'var\s+\w+\s*=',
                r'=>\s*{',  # Arrow functions
                r'class\s+\w+\s*{',
                r'constructor\s*\([^)]*\)',
                r'async\s+function',
                r'await\s+\w+',
                r'module\.exports',
                r'export\s+(?:default\s+)?(?:class|function|const)',
                r'import\s+.*\s+from\s+["\']',
                r'require\s*\(["\']',
                r'console\.\w+\(',
            ],
            'sql': [
                r'SELECT\s+.*\s+FROM\s+\w+',
                r'INSERT\s+INTO\s+\w+',
                r'UPDATE\s+\w+\s+SET',
                r'DELETE\s+FROM\s+\w+',
                r'CREATE\s+(?:TABLE|DATABASE|INDEX)',
                r'ALTER\s+TABLE\s+\w+',
                r'DROP\s+(?:TABLE|DATABASE)',
                r'JOIN\s+\w+\s+ON',
                r'WHERE\s+\w+\s*=',
                r'GROUP\s+BY\s+\w+',
                r'ORDER\s+BY\s+\w+',
            ],
            'general': [
                r'```[\w]*\n[\s\S]*?```',  # Code blocks
                r'npm\s+(?:install|run|start)',
                r'pip\s+install',
                r'git\s+(?:clone|commit|push|pull|checkout)',
                r'docker\s+(?:run|build|compose)',
                r'curl\s+-[A-Za-z]+\s+',
                r'<[^>]+>.*?</[^>]+>',  # HTML/XML
            ]
        }

        # Research and technical keywords
        self.research_keywords = {
            'research': ['research', 'study', 'analysis', 'investigation', 'examine',
                        'explore', 'analyze', 'survey', 'experiment', 'observation'],
            'data': ['data', 'dataset', 'statistics', 'metrics', 'measurement',
                    'correlation', 'regression', 'distribution', 'sample', 'population'],
            'academic': ['paper', 'article', 'publication', 'citation', 'reference',
                        'methodology', 'hypothesis', 'theory', 'literature', 'review'],
            'technical': ['algorithm', 'implementation', 'architecture', 'framework',
                         'optimization', 'performance', 'benchmark', 'evaluation', 'testing'],
            'scientific': ['findings', 'results', 'conclusion', 'evidence', 'significant',
                          'validation', 'verification', 'peer-review', 'reproducible']
        }

        self.url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

    def setup_driver(self, headless: bool = True) -> bool:
        """Setup Chrome/Chromium driver with robust error handling"""
        if not SELENIUM_AVAILABLE:
            raise Exception("Selenium not available. Install with: pip install selenium webdriver-manager")

        options = Options()

        # Chrome/Chromium options for stability
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)

        # Headless mode setting from environment or parameter
        headless = headless and os.getenv('HEADLESS_MODE', 'true').lower() == 'true'

        if headless:
            options.add_argument('--headless')
            options.add_argument('--window-size=1920,1080')
        else:
            options.add_argument('--start-maximized')

        # User agent to appear more natural
        options.add_argument('--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

        # Proxy settings
        if os.getenv('HTTP_PROXY'):
            options.add_argument(f'--proxy-server={os.getenv("HTTP_PROXY")}')

        driver_created = False

        # Try different methods to create driver
        if WEBDRIVER_MANAGER_AVAILABLE:
            try:
                # First try Chrome
                logger.info("Trying Chrome driver...")
                service = ChromeService(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
                driver_created = True
                logger.info("✅ Chrome driver ready")
            except Exception as e:
                logger.warning(f"Chrome driver failed: {e}")

                # Try Chromium
                try:
                    logger.info("Trying Chromium driver...")
                    service = ChromeService(ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install())
                    self.driver = webdriver.Chrome(service=service, options=options)
                    driver_created = True
                    logger.info("✅ Chromium driver ready")
                except Exception as e2:
                    logger.warning(f"Chromium driver failed: {e2}")

        if not driver_created:
            # Try manual methods
            driver_paths = [
                os.getenv('CHROMEDRIVER_PATH', ''),
                '/usr/bin/chromedriver',  # Common on Ubuntu/Debian
                '/usr/lib/chromium-browser/chromedriver',  # Chromium on Ubuntu
                '/snap/bin/chromium.chromedriver',  # Snap Chromium
                '/usr/local/bin/chromedriver',
                'C:\\chromedriver\\chromedriver.exe',
                './chromedriver',
                './chromedriver.exe'
            ]

            # Also check for chromium-specific paths
            chromium_paths = [
                '/usr/bin/chromium-browser',
                '/usr/bin/chromium',
                '/snap/bin/chromium'
            ]

            for path in driver_paths:
                if path and os.path.exists(path):
                    try:
                        service = ChromeService(path)

                        # Check if we should use Chromium binary
                        for chromium_path in chromium_paths:
                            if os.path.exists(chromium_path):
                                options.binary_location = chromium_path
                                logger.info(f"Using Chromium binary at: {chromium_path}")
                                break

                        self.driver = webdriver.Chrome(service=service, options=options)
                        driver_created = True
                        logger.info(f"✅ Driver loaded from: {path}")
                        break
                    except Exception as e:
                        logger.debug(f"Failed with {path}: {e}")
                        continue

        if not driver_created:
            # Check which browser is actually installed
            chrome_installed = False
            chromium_installed = False

            if sys.platform == "linux":
                import subprocess
                try:
                    subprocess.run(['which', 'google-chrome'], check=True, capture_output=True)
                    chrome_installed = True
                except:
                    pass

                try:
                    subprocess.run(['which', 'chromium-browser'], check=True, capture_output=True)
                    chromium_installed = True
                except:
                    try:
                        subprocess.run(['which', 'chromium'], check=True, capture_output=True)
                        chromium_installed = True
                    except:
                        pass

            error_msg = f"""
Chrome/Chromium driver not found.

Browser status:
- Chrome installed: {chrome_installed}
- Chromium installed: {chromium_installed}

To fix this:
1. Install webdriver-manager: pip install webdriver-manager
2. For Ubuntu/Debian with Chromium:
   sudo apt-get install chromium-chromedriver
   or
   sudo snap install chromium
3. Set CHROMEDRIVER_PATH environment variable in .env file
4. Download ChromeDriver manually from https://chromedriver.chromium.org/
"""
            raise Exception(error_msg)

        # Set timeouts
        self.driver.implicitly_wait(10)
        self.driver.set_page_load_timeout(self.request_timeout)

        return True

    def login(self, email: str, password: str) -> bool:
        """Login to ChatGPT with enhanced error handling"""
        if not self.driver:
            raise Exception("Driver not initialized")

        try:
            logger.info("Navigating to ChatGPT...")
            self.driver.get('https://chat.openai.com')

            # Handle CAPTCHA if present
            if CAPTCHA_HANDLER_AVAILABLE:
                integrate_with_scraper(self.driver)

            # Check if already logged in
            time.sleep(3)
            if self._is_logged_in():
                logger.info("✅ Already logged in")
                return True

            logger.info("Starting login process...")

            # Click login button
            login_button = self._find_element_with_fallback([
                (By.XPATH, "//button[contains(text(), 'Log in')]"),
                (By.XPATH, "//a[contains(text(), 'Log in')]"),
                (By.CSS_SELECTOR, "button[data-testid='login-button']"),
                (By.CSS_SELECTOR, "a[href*='auth0']")
            ])

            if not login_button:
                raise Exception("Could not find login button")

            login_button.click()
            time.sleep(3)

            # Handle CAPTCHA on login page if needed
            if CAPTCHA_HANDLER_AVAILABLE:
                integrate_with_scraper(self.driver)

            # Enter email
            email_field = self._find_element_with_fallback([
                (By.NAME, 'username'),
                (By.ID, 'username'),
                (By.NAME, 'email'),
                (By.ID, 'email'),
                (By.CSS_SELECTOR, 'input[type="email"]'),
                (By.CSS_SELECTOR, 'input[autocomplete="email"]')
            ])

            if email_field:
                email_field.clear()
                email_field.send_keys(email)
                email_field.send_keys(Keys.RETURN)
                time.sleep(3)
            else:
                raise Exception("Could not find email field")

            # Enter password
            password_field = self._find_element_with_fallback([
                (By.NAME, 'password'),
                (By.ID, 'password'),
                (By.CSS_SELECTOR, 'input[type="password"]')
            ])

            if password_field:
                password_field.clear()
                password_field.send_keys(password)
                password_field.send_keys(Keys.RETURN)
            else:
                raise Exception("Could not find password field")

            # Wait for login to complete
            logger.info("Waiting for login to complete...")

            # Check for CAPTCHA after login
            if CAPTCHA_HANDLER_AVAILABLE:
                integrate_with_scraper(self.driver)

            # Wait for successful login
            wait = WebDriverWait(self.driver, 30)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'nav[aria-label="Chat history"]')))

            logger.info("✅ Successfully logged in")
            return True

        except TimeoutException:
            logger.error("Login timeout - check credentials or network")
            return False
        except Exception as e:
            logger.error(f"Login failed: {e}")
            # Save screenshot for debugging
            if self.driver:
                self.driver.save_screenshot(f"{self.output_dir}/login_error.png")
            return False
        """Check if already logged in"""
        try:
            self.driver.find_element(By.CSS_SELECTOR, 'nav[aria-label="Chat history"]')
            return True
        except NoSuchElementException:
            return False

    def _find_element_with_fallback(self, locators: List[Tuple]) -> Optional[Any]:
        """Try multiple locators to find element"""
        for by, value in locators:
            try:
                element = self.driver.find_element(by, value)
                if element and element.is_displayed():
                    return element
            except NoSuchElementException:
                continue
        return None

    def extract_conversations_from_export(self) -> List[ConversationMetadata]:
        """Extract and analyze conversations from export file"""
        if not os.path.exists(self.export_file):
            raise FileNotFoundError(f"Export file not found: {self.export_file}")

        logger.info(f"Loading export file: {self.export_file}")

        with open(self.export_file, 'r', encoding='utf-8') as f:
            export_data = json.load(f)

        conversations = []
        projects = []

        # Handle different export formats
        if isinstance(export_data, list):
            conversations = export_data
            logger.info(f"Found {len(conversations)} conversations (list format)")
        elif isinstance(export_data, dict):
            conversations = export_data.get('conversations', [])
            projects = export_data.get('projects', [])
            logger.info(f"Found {len(conversations)} conversations and {len(projects)} projects")
        else:
            raise ValueError(f"Unexpected export format: {type(export_data)}")

        # Create project mapping
        project_map = {}
        if projects:
            for project in projects:
                project_name = project.get('name', 'Unnamed Project')
                for conv_id in project.get('conversation_ids', []):
                    project_map[conv_id] = project_name

        # Process conversations
        results = []

        # Use threading for faster processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Process in chunks
            for i in range(0, len(conversations), self.chunk_size):
                chunk = conversations[i:i + self.chunk_size]

                future_to_conv = {
                    executor.submit(self._analyze_export_conversation, conv, project_map): conv
                    for conv in chunk
                }

                for j, future in enumerate(as_completed(future_to_conv)):
                    if (i + j) % 50 == 0 and (i + j) > 0:
                        logger.info(f"Processed {i + j}/{len(conversations)} conversations...")

                    try:
                        metadata = future.result()
                        if metadata:
                            results.append(metadata)
        # Extract conversations
        for i in range(0, len(conversations), self.chunk_size):
            chunk = conversations[i:i + self.chunk_size]
            print(f"Processing conversations {i+1} to {min(i+self.chunk_size, len(conversations))}...")

            future_to_conv = {
                executor.submit(self._analyze_export_conversation, conv, project_map): conv
                for conv in chunk
            }

            for j, future in enumerate(as_completed(future_to_conv)):
                if (i + j) % 50 == 0 and (i + j) > 0:
                    logger.info(f"Processed {i + j}/{len(conversations)} conversations...")

                try:
                    metadata = future.result()
                    if metadata:
                        results.append(metadata)
                except Exception as e:
                    conv = future_to_conv[future]
                    logger.error(f"Error processing conversation {conv.get('id', 'unknown')}: {e}")

        logger.info(f"✅ Analyzed {len(results)} conversations")

        # Send notification if configured
        self.send_notification(
            "ChatGPT Export Analysis Complete",
            f"Successfully analyzed {len(results)} conversations from export."
        )

        return results

    def _analyze_export_conversation(self,
                                   conversation: Dict[str, Any],
                                   project_map: Dict[str, str]) -> Optional[ConversationMetadata]:
        """Analyze a single exported conversation"""
        try:
            # Extract basic metadata
            conv_id = conversation.get('id', '')
            if not conv_id:
                return None

            # Check cache
            if self.use_cache:
                cached = self._get_cached_analysis(conv_id)
                if cached:
                    return cached

            metadata = ConversationMetadata(
                id=conv_id,
                title=conversation.get('title', 'Untitled'),
                url=f"https://chat.openai.com/c/{conv_id}",
                folder=project_map.get(conv_id, 'No Project'),
                create_time=conversation.get('create_time'),
                update_time=conversation.get('update_time')
            )

            # Extract text content
            text_content = self._extract_text_from_conversation(conversation)

            # Analyze content
            analysis = self._analyze_text_content(text_content)

            # Update metadata with analysis
            metadata.has_code = analysis['has_code']
            metadata.has_research = analysis['has_research']
            metadata.has_links = analysis['has_links']
            metadata.code_blocks = analysis['code_blocks']
            metadata.languages = analysis['languages']
            metadata.research_keywords = analysis['research_keywords']
            metadata.url_count = analysis['url_count']
            metadata.word_count = analysis['word_count']
            metadata.total_messages = self._count_messages(conversation)

            # Determine conversation type
            if metadata.has_code and metadata.has_research:
                metadata.conversation_type = ConversationType.MIXED
            elif metadata.has_code:
                metadata.conversation_type = ConversationType.CODE
            elif metadata.has_research:
                metadata.conversation_type = ConversationType.RESEARCH
            elif metadata.has_links:
                metadata.conversation_type = ConversationType.LINKS
            else:
                metadata.conversation_type = ConversationType.GENERAL

            # Cache result
            if self.use_cache:
                self._cache_analysis(conv_id, metadata)

            return metadata

        except Exception as e:
            logger.error(f"Error analyzing conversation: {e}")
            return None

    def _extract_text_from_conversation(self, conversation: Dict[str, Any]) -> str:
        """Extract all text from a conversation"""
        texts = []

        # Handle mapping format (new export format)
        mapping = conversation.get('mapping', {})
        if mapping:
            for node_id, node in mapping.items():
                message = node.get('message', {})
                if message and message.get('content', {}).get('content_type') == 'text':
                    parts = message.get('content', {}).get('parts', [])
                    for part in parts:
                        if isinstance(part, str):
                            texts.append(part)
        else:
            # Handle old format
            messages = conversation.get('messages', [])
            for message in messages:
                if isinstance(message, str):
                    texts.append(message)
                elif isinstance(message, dict):
                    # Extract text from various possible structures
                    content = message.get('content', message.get('text', ''))
                    if isinstance(content, str):
                        texts.append(content)
                    elif isinstance(content, dict):
                        parts = content.get('parts', [])
                        texts.extend(str(p) for p in parts if p)

        return '\n'.join(texts)

    def _count_messages(self, conversation: Dict[str, Any]) -> int:
        """Count messages in a conversation"""
        mapping = conversation.get('mapping', {})
        if mapping:
            return sum(1 for node in mapping.values()
                      if node.get('message', {}).get('content', {}).get('content_type') == 'text')
        else:
            return len(conversation.get('messages', []))

    def _analyze_text_content(self, text: str) -> Dict[str, Any]:
        """Analyze text content for patterns"""
        analysis = {
            'has_code': False,
            'has_research': False,
            'has_links': False,
            'code_blocks': 0,
            'languages': [],
            'research_keywords': [],
            'url_count': 0,
            'word_count': len(text.split())
        }

        # Check for code
        code_blocks = re.findall(r'```[\w]*\n[\s\S]*?```', text)
        analysis['code_blocks'] = len(code_blocks)

        # Detect languages
        languages = set()

        # From code blocks
        for block in code_blocks:
            lang_match = re.match(r'```(\w+)', block)
            if lang_match:
                lang = lang_match.group(1).lower()
                if lang and lang not in ['text', 'output', 'bash', 'shell', 'console']:
                    languages.add(lang)

        # From patterns
        for lang, patterns in self.code_patterns.items():
            if lang != 'general':
                for pattern in patterns:
                    if re.search(pattern, text, re.MULTILINE):
                        languages.add(lang)
                        break

        analysis['languages'] = sorted(list(languages))
        analysis['has_code'] = bool(analysis['code_blocks'] or analysis['languages'])

        # Check for research content
        text_lower = text.lower()
        found_keywords = []

        for category, keywords in self.research_keywords.items():
            category_matches = [kw for kw in keywords if kw in text_lower]
            found_keywords.extend(category_matches)

        # Remove duplicates and sort by frequency
        keyword_counts = defaultdict(int)
        for kw in found_keywords:
            keyword_counts[kw] += text_lower.count(kw)

        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        analysis['research_keywords'] = [kw for kw, _ in sorted_keywords[:10]]

        # Consider it research if we have enough keywords
        analysis['has_research'] = len(set(found_keywords)) >= 5

        # Check for URLs
        urls = self.url_pattern.findall(text)
        analysis['url_count'] = len(urls)
        analysis['has_links'] = bool(urls)

        return analysis

    def _get_cached_analysis(self, conv_id: str) -> Optional[ConversationMetadata]:
        """Get cached analysis if available"""
        cache_file = os.path.join(self.cache_dir, f"{conv_id}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    # Recreate ConversationMetadata from dict
                    metadata = ConversationMetadata(**data)
                    return metadata
            except Exception:
                pass
        return None

    def _cache_analysis(self, conv_id: str, metadata: ConversationMetadata):
        """Cache analysis results"""
        cache_file = os.path.join(self.cache_dir, f"{conv_id}.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump(metadata.to_dict(), f)
        except Exception:
            pass

    def extract_conversations_from_browser(self) -> List[ConversationMetadata]:
        """Extract conversations from live ChatGPT"""
        if not self.driver:
            raise Exception("Driver not initialized")

        results = []

        try:
            # Get all conversation links
            logger.info("Extracting conversation list...")

            # Wait for sidebar to load
            wait = WebDriverWait(self.driver, 10)
            sidebar = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'nav[aria-label="Chat history"]'))
            )

            # Scroll to load all conversations
            self._scroll_sidebar(sidebar)

            # Get all conversation links
            links = self.driver.find_elements(By.CSS_SELECTOR, 'a[href*="/c/"]')
            logger.info(f"Found {len(links)} conversations")

            # Process each conversation
            for i, link in enumerate(links):
                try:
                    # Get URL and title
                    url = link.get_attribute('href')
                    title = link.text.strip() or 'Untitled'
                    conv_id = url.split('/c/')[-1] if '/c/' in url else None

                    if not conv_id:
                        continue

                    logger.info(f"Analyzing conversation {i+1}/{len(links)}: {title[:50]}...")

                    # Navigate to conversation
                    self.driver.get(url)
                    time.sleep(2)

                    # Extract and analyze
                    metadata = self._analyze_live_conversation(conv_id, title, url)
                    if metadata:
                        results.append(metadata)

                except Exception as e:
                    logger.error(f"Error processing conversation: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error extracting conversations: {e}")

        return results

    def _scroll_sidebar(self, sidebar):
        """Scroll sidebar to load all conversations"""
        last_height = 0
        while True:
            # Scroll sidebar
            self.driver.execute_script(
                "arguments[0].scrollTop = arguments[0].scrollHeight",
                sidebar
            )
            time.sleep(1)

            # Check if more conversations loaded
            new_height = self.driver.execute_script(
                "return arguments[0].scrollHeight",
                sidebar
            )

            if new_height == last_height:
                break

            last_height = new_height

    def _analyze_live_conversation(self, conv_id: str, title: str, url: str) -> Optional[ConversationMetadata]:
        """Analyze a live conversation"""
        try:
            # Create metadata
            metadata = ConversationMetadata(
                id=conv_id,
                title=title,
                url=url
            )

            # Extract messages
            messages = []
            message_elements = self.driver.find_elements(By.CSS_SELECTOR, 'div[data-message-author-role]')

            for elem in message_elements:
                try:
                    text = elem.text.strip()
                    if text:
                        messages.append(text)
                except Exception:
                    continue

            # Combine and analyze
            full_text = '\n'.join(messages)
            analysis = self._analyze_text_content(full_text)

            # Update metadata
            metadata.total_messages = len(messages)
            metadata.has_code = analysis['has_code']
            metadata.has_research = analysis['has_research']
            metadata.has_links = analysis['has_links']
            metadata.code_blocks = analysis['code_blocks']
            metadata.languages = analysis['languages']
            metadata.research_keywords = analysis['research_keywords']
            metadata.url_count = analysis['url_count']
            metadata.word_count = analysis['word_count']

            # Determine type
            if metadata.has_code and metadata.has_research:
                metadata.conversation_type = ConversationType.MIXED
            elif metadata.has_code:
                metadata.conversation_type = ConversationType.CODE
            elif metadata.has_research:
                metadata.conversation_type = ConversationType.RESEARCH
            elif metadata.has_links:
                metadata.conversation_type = ConversationType.LINKS


        return metadata

        except Exception as e:
            logger.error(f"Error analyzing conversation: {e}")
            return None

    def send_notification(self, subject: str, body: str):
        """Send email notification if configured"""
        if not EMAIL_AVAILABLE:
            logger.debug("Email support not available")
            return

        if not all([self.smtp_server, self.smtp_username, self.smtp_password, self.notify_email]):
            logger.debug("Email notification not configured")
            return

        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_username
            msg['To'] = self.notify_email
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)

            server.send_message(msg)
            server.quit()

            logger.info(f"Notification sent to {self.notify_email}")
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

    def generate_reports(self, conversations: List[ConversationMetadata], formats: List[str] = None):
        """Generate analysis reports in multiple formats"""
        if formats is None:
            formats = os.getenv('DEFAULT_FORMATS', 'html').split(',')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Filter to relevant conversations
        relevant = [c for c in conversations
                   if c.conversation_type != ConversationType.GENERAL]

        if not relevant:
            logger.warning("No relevant conversations found")
            return

        logger.info(f"Generating reports for {len(relevant)} conversations...")

        for format_type in formats:
            if format_type == 'html':
                self._generate_html_report(relevant, timestamp)
            elif format_type == 'csv':
                self._generate_csv_report(relevant, timestamp)
            elif format_type == 'json':
                self._generate_json_report(relevant, timestamp)
            elif format_type == 'markdown':
                self._generate_markdown_report(relevant, timestamp)
            elif format_type == 'pdf' and REPORTLAB_AVAILABLE:
                self._generate_pdf_report(relevant, timestamp)

    def _generate_html_report(self, conversations: List[ConversationMetadata], timestamp: str):
        """Generate interactive HTML report"""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChatGPT Conversation Analysis - {timestamp}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f0f2f5;
            color: #333;
            line-height: 1.6;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 3rem 0;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            font-size: 3rem;
            margin-bottom: 1rem;
            animation: fadeIn 1s ease-out;
        }}
        .header p {{
            font-size: 1.2rem;
            opacity: 0.9;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 2rem;
            margin: -4rem auto 3rem;
            padding: 0 2rem;
            max-width: 1400px;
        }}
        .stat-card {{
            background: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        }}
        .stat-icon {{
            font-size: 3rem;
            margin-bottom: 1rem;
        }}
        .stat-number {{
            font-size: 3rem;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        .stat-label {{
            color: #666;
            font-weight: 500;
            font-size: 1.1rem;
        }}
        .filters {{
            background: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            margin-bottom: 3rem;
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
            align-items: center;
        }}
        .filter-group {{
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }}
        .filter-label {{
            font-weight: 600;
            color: #555;
        }}
        .filter-btn {{
            padding: 0.75rem 1.5rem;
            border: 2px solid #e0e0e0;
            background: white;
            color: #666;
            border-radius: 30px;
            cursor: pointer;
            font-weight: 500;
            font-size: 0.95rem;
            transition: all 0.3s ease;
            white-space: nowrap;
        }}
        .filter-btn:hover {{
            border-color: #667eea;
            color: #667eea;
            transform: translateY(-2px);
        }}
        .filter-btn.active {{
            background: #667eea;
            color: white;
            border-color: #667eea;
        }}
        .search-container {{
            flex: 1;
            min-width: 300px;
            position: relative;
        }}
        .search-box {{
            width: 100%;
            padding: 0.75rem 1.5rem 0.75rem 3rem;
            border: 2px solid #e0e0e0;
            border-radius: 30px;
            font-size: 1rem;
            transition: all 0.3s ease;
        }}
        .search-box:focus {{
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }}
        .search-icon {{
            position: absolute;
            left: 1rem;
            top: 50%;
            transform: translateY(-50%);
            color: #999;
        }}
        .folder-section {{
            margin-bottom: 3rem;
        }}
        .folder-header {{
            background: white;
            padding: 1.5rem 2rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        .folder-title {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        .folder-title h2 {{
            font-size: 1.8rem;
            color: #333;
        }}
        .folder-count {{
            background: #f0f2f5;
            color: #666;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 600;
        }}
        .conversations-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
            gap: 1.5rem;
        }}
        .conversation-card {{
            background: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
            border-left: 4px solid #667eea;
            position: relative;
            overflow: hidden;
        }}
        .conversation-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}
        .conversation-card.code {{
            border-left-color: #10b981;
        }}
        .conversation-card.research {{
            border-left-color: #3b82f6;
        }}
        .conversation-card.mixed {{
            background: linear-gradient(to right, #10b98110 0%, #3b82f610 100%);
            border-left: 4px solid transparent;
            border-image: linear-gradient(135deg, #10b981 0%, #3b82f6 100%) 1;
        }}
        .conversation-card h3 {{
            margin-bottom: 1rem;
            font-size: 1.2rem;
            line-height: 1.4;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }}
        .conversation-card h3 a {{
            color: #333;
            text-decoration: none;
            transition: color 0.2s;
        }}
        .conversation-card h3 a:hover {{
            color: #667eea;
        }}
        .tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}
        .tag {{
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.4rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }}
        .tag.type {{
            background: #e0e7ff;
            color: #4338ca;
        }}
        .tag.language {{
            background: #d1fae5;
            color: #065f46;
        }}
        .tag.keyword {{
            background: #fee2e2;
            color: #991b1b;
        }}
        .tag.links {{
            background: #fef3c7;
            color: #92400e;
        }}
        .metadata {{
            color: #666;
            font-size: 0.9rem;
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid #f0f0f0;
        }}
        .metadata span {{
            display: flex;
            align-items: center;
            gap: 0.3rem;
        }}
        .hidden {{
            display: none !important;
        }}
        .empty-state {{
            text-align: center;
            padding: 5rem 2rem;
            color: #999;
        }}
        .empty-state h3 {{
            font-size: 2rem;
            margin-bottom: 1rem;
            color: #666;
        }}
        .charts-section {{
            margin-bottom: 3rem;
        }}
        .chart-container {{
            background: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 2rem;
        }}
        .chart-title {{
            font-size: 1.5rem;
            margin-bottom: 1.5rem;
            color: #333;
        }}
        .language-chart {{
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }}
        .language-bar {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        .language-name {{
            min-width: 100px;
            font-weight: 600;
            color: #555;
        }}
        .language-progress {{
            flex: 1;
            height: 30px;
            background: #f0f2f5;
            border-radius: 15px;
            overflow: hidden;
            position: relative;
        }}
        .language-fill {{
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            transition: width 1s ease-out;
            position: relative;
        }}
        .language-count {{
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            color: white;
            font-weight: 600;
            font-size: 0.9rem;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        @media (max-width: 768px) {{
            .header h1 {{ font-size: 2rem; }}
            .stats-grid {{ margin-top: -3rem; }}
            .stat-card {{ padding: 1.5rem; }}
            .stat-number {{ font-size: 2.5rem; }}
            .conversations-grid {{ grid-template-columns: 1fr; }}
            .filters {{ flex-direction: column; align-items: stretch; }}
            .search-container {{ min-width: 100%; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 ChatGPT Conversation Analysis</h1>
        <p>Generated on {date} at {time}</p>
    </div>

    <div class="stats-grid">
        <div class="stat-card" onclick="filterConversations('all')">
            <div class="stat-icon">📊</div>
            <div class="stat-number">{total_conversations}</div>
            <div class="stat-label">Total Conversations</div>
        </div>
        <div class="stat-card" onclick="filterConversations('code')">
            <div class="stat-icon">💻</div>
            <div class="stat-number">{code_conversations}</div>
            <div class="stat-label">Code Conversations</div>
        </div>
        <div class="stat-card" onclick="filterConversations('research')">
            <div class="stat-icon">🔬</div>
            <div class="stat-number">{research_conversations}</div>
            <div class="stat-label">Research Conversations</div>
        </div>
        <div class="stat-card" onclick="filterConversations('mixed')">
            <div class="stat-icon">🔀</div>
            <div class="stat-number">{mixed_conversations}</div>
            <div class="stat-label">Mixed Content</div>
        </div>
    </div>

    <div class="container">
        <div class="filters">
            <div class="filter-group">
                <span class="filter-label">Type:</span>
                <button class="filter-btn active" onclick="filterConversations('all')">All</button>
                <button class="filter-btn" onclick="filterConversations('code')">💻 Code</button>
                <button class="filter-btn" onclick="filterConversations('research')">🔬 Research</button>
                <button class="filter-btn" onclick="filterConversations('mixed')">🔀 Mixed</button>
                <button class="filter-btn" onclick="filterConversations('links')">🔗 Links</button>
            </div>
            <div class="search-container">
                <span class="search-icon">🔍</span>
                <input type="text" class="search-box" placeholder="Search conversations by title, language, or keyword..."
                       onkeyup="searchConversations(this.value)">
            </div>
        </div>

        <div class="charts-section">
            <div class="chart-container">
                <h3 class="chart-title">📊 Programming Languages Distribution</h3>
                <div class="language-chart">
                    {language_chart}
                </div>
            </div>
        </div>

        <div id="content">
            {folders_content}
        </div>

        <div id="empty-state" class="empty-state hidden">
            <h3>No conversations found</h3>
            <p>Try adjusting your filters or search terms</p>
        </div>
    </div>

    <script>
        let currentFilter = 'all';
        let searchTerm = '';

        function filterConversations(type) {{
            currentFilter = type;

            // Update button states
            const buttons = document.querySelectorAll('.filter-btn');
            buttons.forEach(btn => {{
                btn.classList.remove('active');
                if (btn.textContent.toLowerCase().includes(type) ||
                    (type === 'all' && btn.textContent === 'All')) {{
                    btn.classList.add('active');
                }}
            }});

            applyFilters();
        }}

        function searchConversations(term) {{
            searchTerm = term.toLowerCase();
            applyFilters();
        }}

        function applyFilters() {{
            const cards = document.querySelectorAll('.conversation-card');
            let visibleCount = 0;

            cards.forEach(card => {{
                const type = card.dataset.type;
                const title = card.querySelector('h3').textContent.toLowerCase();
                const tags = Array.from(card.querySelectorAll('.tag')).map(t => t.textContent.toLowerCase());

                const matchesFilter = currentFilter === 'all' || type === currentFilter;
                const matchesSearch = !searchTerm ||
                    title.includes(searchTerm) ||
                    tags.some(tag => tag.includes(searchTerm));

                const shouldShow = matchesFilter && matchesSearch;
                card.classList.toggle('hidden', !shouldShow);

                if (shouldShow) visibleCount++;
            }});

            // Show/hide folders
            document.querySelectorAll('.folder-section').forEach(folder => {{
                const visibleCards = folder.querySelectorAll('.conversation-card:not(.hidden)').length;
                folder.classList.toggle('hidden', visibleCards === 0);
            }});

            // Show empty state if needed
            document.getElementById('empty-state').classList.toggle('hidden', visibleCount > 0);
        }}

        // Animate language bars on load
        window.addEventListener('load', () => {{
            const bars = document.querySelectorAll('.language-fill');
            bars.forEach((bar, index) => {{
                const width = bar.style.width;
                bar.style.width = '0';
                setTimeout(() => {{
                    bar.style.width = width;
                }}, 100 * index);
            }});
        }});
    </script>
</body>
</html>
        """

        # Prepare data
        total = len(conversations)
        code_count = sum(1 for c in conversations if c.conversation_type == ConversationType.CODE)
        research_count = sum(1 for c in conversations if c.conversation_type == ConversationType.RESEARCH)
        mixed_count = sum(1 for c in conversations if c.conversation_type == ConversationType.MIXED)

        # Language statistics
        language_counts = defaultdict(int)
        for conv in conversations:
            for lang in conv.languages:
                language_counts[lang] += 1

        # Generate language chart
        language_chart = ""
        if language_counts:
            sorted_langs = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            max_count = sorted_langs[0][1] if sorted_langs else 1

            for lang, count in sorted_langs:
                percentage = (count / max_count) * 100
                language_chart += f"""
                <div class="language-bar">
                    <span class="language-name">{lang.title()}</span>
                    <div class="language-progress">
                        <div class="language-fill" style="width: {percentage}%">
                            <span class="language-count">{count}</span>
                        </div>
                    </div>
                </div>
                """

        # Group by folders
        folders = defaultdict(list)
        for conv in conversations:
            folders[conv.folder].append(conv)

        # Generate folder content
        folders_content = ""
        for folder_name in sorted(folders.keys()):
            folder_convs = folders[folder_name]
            folders_content += f"""
            <div class="folder-section">
                <div class="folder-header">
                    <div class="folder-title">
                        <h2>📁 {folder_name}</h2>
                        <span class="folder-count">{len(folder_convs)} conversations</span>
                    </div>
                </div>
                <div class="conversations-grid">
            """

            for conv in sorted(folder_convs, key=lambda x: x.update_time or '', reverse=True):
                # Determine card class
                card_class = "conversation-card"
                if conv.conversation_type == ConversationType.CODE:
                    card_class += " code"
                elif conv.conversation_type == ConversationType.RESEARCH:
                    card_class += " research"
                elif conv.conversation_type == ConversationType.MIXED:
                    card_class += " mixed"

                folders_content += f"""
                <div class="{card_class}" data-type="{conv.conversation_type.value}">
                    <h3><a href="{conv.url}" target="_blank">{conv.title}</a></h3>
                    <div class="tags">
                """

                # Type tag
                type_emoji = {
                    ConversationType.CODE: "💻",
                    ConversationType.RESEARCH: "🔬",
                    ConversationType.MIXED: "🔀",
                    ConversationType.LINKS: "🔗"
                }.get(conv.conversation_type, "📝")

                folders_content += f'<span class="tag type">{type_emoji} {conv.conversation_type.value.title()}</span>'

                # Language tags
                for lang in conv.languages[:3]:
                    folders_content += f'<span class="tag language">{lang}</span>'

                # Code blocks
                if conv.code_blocks > 0:
                    folders_content += f'<span class="tag type">📝 {conv.code_blocks} blocks</span>'

                # Links
                if conv.has_links:
                    folders_content += f'<span class="tag links">🔗 {conv.url_count} links</span>'

                folders_content += '</div>'

                # Research keywords
                if conv.research_keywords:
                    folders_content += '<div class="tags">'
                    for kw in conv.research_keywords[:4]:
                        folders_content += f'<span class="tag keyword">{kw}</span>'
                    folders_content += '</div>'

                # Metadata
                folders_content += f"""
                    <div class="metadata">
                        <span>💬 {conv.total_messages} messages</span>
                        <span>📝 {conv.word_count:,} words</span>
                    </div>
                </div>
                """

            folders_content += """
                </div>
            </div>
            """

        # Format HTML
        html = html_template.format(
            timestamp=timestamp,
            date=datetime.now().strftime('%B %d, %Y'),
            time=datetime.now().strftime('%I:%M %p'),
            total_conversations=total,
            code_conversations=code_count,
            research_conversations=research_count,
            mixed_conversations=mixed_count,
            language_chart=language_chart,
            folders_content=folders_content
        )

        # Save file
        output_file = os.path.join(self.output_dir, f'chatgpt_analysis_{timestamp}.html')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"✅ HTML report saved to: {output_file}")

        # Try to open in browser
        try:
            import webbrowser
            webbrowser.open(f'file://{os.path.abspath(output_file)}')
            logger.info("📂 Opening report in browser...")
        except Exception:
            pass

    def _generate_csv_report(self, conversations: List[ConversationMetadata], timestamp: str):
        """Generate CSV report"""
        output_file = os.path.join(self.output_dir, f'chatgpt_analysis_{timestamp}.csv')

        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            fieldnames = [
                'folder', 'title', 'type', 'url', 'has_code', 'has_research',
                'has_links', 'languages', 'code_blocks', 'url_count',
                'total_messages', 'word_count', 'research_keywords',
                'create_time', 'update_time'
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for conv in conversations:
                row = {
                    'folder': conv.folder,
                    'title': conv.title,
                    'type': conv.conversation_type.value,
                    'url': conv.url,
                    'has_code': conv.has_code,
                    'has_research': conv.has_research,
                    'has_links': conv.has_links,
                    'languages': ', '.join(conv.languages),
                    'code_blocks': conv.code_blocks,
                    'url_count': conv.url_count,
                    'total_messages': conv.total_messages,
                    'word_count': conv.word_count,
                    'research_keywords': ', '.join(conv.research_keywords[:5]),
                    'create_time': conv.create_time or '',
                    'update_time': conv.update_time or ''
                }
                writer.writerow(row)

        logger.info(f"✅ CSV report saved to: {output_file}")

    def _generate_json_report(self, conversations: List[ConversationMetadata], timestamp: str):
        """Generate JSON report"""
        output_file = os.path.join(self.output_dir, f'chatgpt_analysis_{timestamp}.json')

        data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_conversations': len(conversations),
                'analyzer_version': '2.0'
            },
            'conversations': [conv.to_dict() for conv in conversations]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ JSON report saved to: {output_file}")

    def _generate_markdown_report(self, conversations: List[ConversationMetadata], timestamp: str):
        """Generate Markdown report"""
        output_file = os.path.join(self.output_dir, f'chatgpt_analysis_{timestamp}.md')

        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("# 🤖 ChatGPT Conversation Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}\n\n")
            f.write("---\n\n")

            # Summary
            f.write("## 📊 Executive Summary\n\n")

            total = len(conversations)
            code_count = sum(1 for c in conversations if c.conversation_type == ConversationType.CODE)
            research_count = sum(1 for c in conversations if c.conversation_type == ConversationType.RESEARCH)
            mixed_count = sum(1 for c in conversations if c.conversation_type == ConversationType.MIXED)

            f.write("### Key Metrics\n\n")
            f.write("| Metric | Count | Percentage |\n")
            f.write("|--------|-------|------------|\n")
            f.write(f"| **Total Conversations** | {total} | 100% |\n")
            f.write(f"| **Code Conversations** | {code_count} | {code_count/total*100:.1f}% |\n")
            f.write(f"| **Research Conversations** | {research_count} | {research_count/total*100:.1f}% |\n")
            f.write(f"| **Mixed Content** | {mixed_count} | {mixed_count/total*100:.1f}% |\n\n")

            # Language distribution
            language_counts = defaultdict(int)
            for conv in conversations:
                for lang in conv.languages:
                    language_counts[lang] += 1

            if language_counts:
                f.write("### 💻 Programming Languages\n\n")
                sorted_langs = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)

                for lang, count in sorted_langs[:10]:
                    bar_length = int((count / sorted_langs[0][1]) * 20)
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    f.write(f"**{lang:12}** {bar} {count} conversations\n")
                f.write("\n")

            # Conversations by folder
            f.write("## 📁 Conversations by Category\n\n")

            folders = defaultdict(list)
            for conv in conversations:
                folders[conv.folder].append(conv)

            for folder_name in sorted(folders.keys()):
                folder_convs = folders[folder_name]
                f.write(f"### 📂 {folder_name}\n\n")
                f.write(f"*{len(folder_convs)} conversations*\n\n")

                # Group by type
                by_type = defaultdict(list)
                for conv in folder_convs:
                    by_type[conv.conversation_type].append(conv)

                for conv_type, convs in by_type.items():
                    type_emoji = {
                        ConversationType.CODE: "💻",
                        ConversationType.RESEARCH: "🔬",
                        ConversationType.MIXED: "🔀",
                        ConversationType.LINKS: "🔗"
                    }.get(conv_type, "📝")

                    f.write(f"#### {type_emoji} {conv_type.value.title()}\n\n")

                    for conv in sorted(convs, key=lambda x: x.update_time or '', reverse=True):
                        f.write(f"**[{conv.title}]({conv.url})**\n\n")

                        # Summary line
                        summary_parts = []
                        if conv.code_blocks > 0:
                            summary_parts.append(f"💻 {conv.code_blocks} code blocks")
                        if conv.languages:
                            summary_parts.append(f"Languages: {', '.join(conv.languages[:3])}")
                        if conv.has_links:
                            summary_parts.append(f"🔗 {conv.url_count} links")
                        summary_parts.append(f"💬 {conv.total_messages} messages")

                        f.write(f"*{' • '.join(summary_parts)}*\n\n")

                        if conv.research_keywords:
                            f.write(f"**Topics:** {', '.join(conv.research_keywords[:5])}\n\n")

                        f.write("---\n\n")

        logger.info(f"✅ Markdown report saved to: {output_file}")

    def _generate_pdf_report(self, conversations: List[ConversationMetadata], timestamp: str):
        """Generate PDF report"""
        output_file = os.path.join(self.output_dir, f'chatgpt_analysis_{timestamp}.pdf')

        doc = SimpleDocTemplate(output_file, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER
        )

        elements.append(Paragraph("ChatGPT Conversation Analysis Report", title_style))
        elements.append(Spacer(1, 20))

        # Summary table
        summary_data = [
            ['Metric', 'Count', 'Percentage'],
            ['Total Conversations', str(len(conversations)), '100%'],
            ['Code Conversations',
             str(sum(1 for c in conversations if c.conversation_type == ConversationType.CODE)),
             f"{sum(1 for c in conversations if c.conversation_type == ConversationType.CODE)/len(conversations)*100:.1f}%"],
            ['Research Conversations',
             str(sum(1 for c in conversations if c.conversation_type == ConversationType.RESEARCH)),
             f"{sum(1 for c in conversations if c.conversation_type == ConversationType.RESEARCH)/len(conversations)*100:.1f}%"]
        ]

        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        elements.append(summary_table)
        elements.append(PageBreak())

        # Conversation details
        for conv in conversations[:50]:  # Limit to first 50 for PDF size
            elements.append(Paragraph(conv.title, styles['Heading2']))
            elements.append(Paragraph(f"Type: {conv.conversation_type.value}", styles['Normal']))
            if conv.languages:
                elements.append(Paragraph(f"Languages: {', '.join(conv.languages)}", styles['Normal']))
            elements.append(Spacer(1, 12))

        doc.build(elements)
        logger.info(f"✅ PDF report saved to: {output_file}")

    def extract_code_and_research(self, conversations: List[ConversationMetadata], selected_ids: List[str] = None):
        """Extract code and research content from conversations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extract_dir = os.path.join(self.output_dir, f'extracted_content_{timestamp}')
        code_dir = os.path.join(extract_dir, 'code')
        research_dir = os.path.join(extract_dir, 'research')

        os.makedirs(code_dir, exist_ok=True)
        os.makedirs(research_dir, exist_ok=True)

        # Filter conversations
        if selected_ids:
            conversations = [c for c in conversations if c.id in selected_ids]

        logger.info(f"Extracting content from {len(conversations)} conversations...")

        # Extract content based on mode
        if self.mode == 'export':
            self._extract_from_export(conversations, code_dir, research_dir)
        else:
            self._extract_from_browser(conversations, code_dir, research_dir)

        logger.info(f"✅ Content extracted to: {extract_dir}")

        # Send notification if configured
        self.send_notification(
            "Content Extraction Complete",
            f"Successfully extracted content from {len(conversations)} conversations.\n"
            f"Output directory: {extract_dir}"
        )

    def _extract_from_export(self, conversations: List[ConversationMetadata], code_dir: str, research_dir: str):
        """Extract content from export file"""
        # Load full export data
        with open(self.export_file, 'r', encoding='utf-8') as f:
            export_data = json.load(f)

        # Build conversation map
        conv_map = {}
        if isinstance(export_data, list):
            for conv in export_data:
                conv_map[conv.get('id', '')] = conv
        elif isinstance(export_data, dict):
            for conv in export_data.get('conversations', []):
                conv_map[conv.get('id', '')] = conv

        # Process each conversation
        for metadata in conversations:
            if metadata.id not in conv_map:
                continue

            conv_data = conv_map[metadata.id]

            # Extract text
            text = self._extract_text_from_conversation(conv_data)

            # Create safe filename
            safe_title = re.sub(r'[^\w\s-]', '', metadata.title)[:50]
            base_name = f"{safe_title}_{metadata.id[:8]}"

            # Extract code if present
            if metadata.has_code:
                conv_code_dir = os.path.join(code_dir, base_name)
                os.makedirs(conv_code_dir, exist_ok=True)

                # Extract code blocks
                code_blocks = re.findall(r'```(\w*)\n([\s\S]*?)```', text)

                for i, (lang, code) in enumerate(code_blocks):
                    if not lang:
                        # Try to detect language
                        if 'def ' in code or 'import ' in code:
                            lang = 'python'
                        elif 'function ' in code or 'const ' in code:
                            lang = 'javascript'
                        else:
                            lang = 'txt'

                    # Determine extension
                    extensions = {
                        'python': 'py', 'py': 'py',
                        'javascript': 'js', 'js': 'js',
                        'java': 'java',
                        'cpp': 'cpp', 'c++': 'cpp',
                        'c': 'c',
                        'html': 'html',
                        'css': 'css',
                        'sql': 'sql',
                        'go': 'go',
                        'rust': 'rs',
                        'typescript': 'ts', 'ts': 'ts'
                    }

                    ext = extensions.get(lang.lower(), 'txt')
                    filename = f"{lang}_{i+1:02d}.{ext}"

                    with open(os.path.join(conv_code_dir, filename), 'w', encoding='utf-8') as f:
                        f.write(f"# Extracted from: {metadata.title}\n")
                        f.write(f"# Conversation ID: {metadata.id}\n")
                        f.write(f"# Language: {lang}\n\n")
                        f.write(code)

                # Save metadata
                with open(os.path.join(conv_code_dir, 'metadata.json'), 'w') as f:
                    json.dump({
                        'title': metadata.title,
                        'url': metadata.url,
                        'languages': metadata.languages,
                        'code_blocks': len(code_blocks),
                        'extracted_at': datetime.now().isoformat()
                    }, f, indent=2)

            # Extract research if present
            if metadata.has_research:
                # Create markdown document
                md_file = os.path.join(research_dir, f"{base_name}.md")

                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(f"# {metadata.title}\n\n")
                    f.write(f"**Source:** [{metadata.url}]({metadata.url})\n")
                    f.write(f"**Keywords:** {', '.join(metadata.research_keywords[:10])}\n\n")
                    f.write("---\n\n")

                    # Extract structured content
                    # This is simplified - you could add more sophisticated extraction
                    sections = text.split('\n\n')
                    for section in sections:
                        if section.strip():
                            f.write(f"{section}\n\n")


                # Save based on type
                if metadata.has_code and code_blocks:
                    content = {
                        'title': metadata.title,
                        'url': metadata.url,
                        'code_blocks': code_blocks,
                        'messages': []
                    }
                    self._save_code_files(metadata, content, code_dir, base_name)

                if metadata.has_research:
                    # Create full content for research
                    content = {
                        'title': metadata.title,
                        'url': metadata.url,
                        'messages': [],
                        'links': []
                    }

                    # Extract full conversation for research
                    mapping = conv_data.get('mapping', {})
                    if mapping:
                        for node_id, node in mapping.items():
                            message = node.get('message', {})
                            if message and message.get('content', {}).get('content_type') == 'text':
                                role = message.get('author', {}).get('role', 'unknown')
                                parts = message.get('content', {}).get('parts', [])
                                full_text = '\n'.join(str(p) for p in parts)
                                content['messages'].append({'role': role, 'content': full_text})

                    self._save_research_document(metadata, content, research_dir, base_name)

    def _save_code_files(self, metadata: ConversationMetadata, content: Dict, code_dir: str, base_name: str):
        """Save code blocks as individual files"""
        conv_dir = os.path.join(code_dir, base_name)
        os.makedirs(conv_dir, exist_ok=True)

        # Save metadata
        metadata_dict = {
            'title': metadata.title,
            'url': metadata.url,
            'extracted_at': datetime.now().isoformat(),
            'total_code_blocks': len(content['code_blocks']),
            'languages': metadata.languages
        }

        with open(os.path.join(conv_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata_dict, f, indent=2)

        # Save each code block
        language_counts = defaultdict(int)
        for code_block in content['code_blocks']:
            language = code_block['language']
            if language == 'unknown':
                # Try to detect from content
                if 'def ' in code_block['code'] or 'import ' in code_block['code']:
                    language = 'python'
                elif 'function ' in code_block['code'] or 'const ' in code_block['code']:
                    language = 'javascript'

            # Determine file extension
            extensions = {
                'python': '.py', 'py': '.py',
                'javascript': '.js', 'js': '.js',
                'java': '.java',
                'cpp': '.cpp', 'c++': '.cpp',
                'c': '.c',
                'html': '.html',
                'css': '.css',
                'sql': '.sql',
                'go': '.go',
                'rust': '.rs',
                'typescript': '.ts', 'ts': '.ts'
            }

            ext = extensions.get(language.lower(), '.txt')
            language_counts[language] += 1

            # Create filename
            filename = f"{language}_{language_counts[language]:02d}{ext}"
            filepath = os.path.join(conv_dir, filename)

            # Save code
            with open(filepath, 'w', encoding='utf-8') as f:
                # Add header comment
                if ext in ['.py', '.js', '.java', '.cpp', '.c', '.go', '.rs']:
                    f.write(f"# Extracted from: {metadata.title}\n")
                    f.write(f"# Role: {code_block.get('role', 'unknown')}\n")
                    f.write(f"# Language: {language}\n")
                    f.write(f"# Extracted at: {datetime.now().isoformat()}\n\n")

                f.write(code_block['code'])

        logger.info(f"✅ Saved {len(content['code_blocks'])} code files to: {conv_dir}")

    def _save_research_document(self, metadata: ConversationMetadata, content: Dict, research_dir: str, base_name: str):
        """Save research conversation as formatted document"""
        conv_dir = os.path.join(research_dir, base_name)
        os.makedirs(conv_dir, exist_ok=True)

        # Save as Markdown
        md_file = os.path.join(conv_dir, 'research.md')
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(f"# {metadata.title}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Source:** [{metadata.url}]({metadata.url})\n\n")

            if metadata.research_keywords:
                f.write(f"**Keywords:** {', '.join(metadata.research_keywords[:10])}\n\n")

            f.write("---\n\n")

            if content.get('messages'):
                for msg in content['messages']:
                    if msg['role'] == 'user':
                        f.write(f"### 🔍 Question\n\n")
                    else:
                        f.write(f"### 📊 Analysis\n\n")

                    f.write(msg['content'])
                    f.write("\n\n")

        logger.info(f"✅ Saved research document to: {conv_dir}")

    def _extract_from_browser(self, conversations: List[ConversationMetadata], code_dir: str, research_dir: str):
        """Extract content from browser (live mode)"""
        logger.info("Extracting content from live conversations...")

        for metadata in conversations:
            try:
                # Navigate to conversation
                self.driver.get(metadata.url)
                time.sleep(3)

                # Extract messages
                messages = []
                message_elements = self.driver.find_elements(By.CSS_SELECTOR, 'div[data-message-author-role]')

                for elem in message_elements:
                    try:
                        role = elem.get_attribute('data-message-author-role')
                        text = elem.text.strip()
                        if text:
                            messages.append({'role': role, 'content': text})
                    except Exception:
                        continue

                # Create content structure
                content = {
                    'title': metadata.title,
                    'url': metadata.url,
                    'messages': messages,
                    'code_blocks': [],
                    'links': []
                }

                # Extract code blocks and links from messages
                for msg in messages:
                    text = msg['content']

                    # Extract code blocks
                    code_pattern = r'```(\w*)\n([\s\S]*?)```'
                    code_matches = re.findall(code_pattern, text)
                    for lang, code in code_matches:
                        content['code_blocks'].append({
                            'language': lang or 'unknown',
                            'code': code.strip(),
                            'role': msg['role']
                        })

                    # Extract URLs
                    url_matches = re.findall(self.url_pattern, text)
                    for url in url_matches:
                        content['links'].append({
                            'url': url,
                            'text': '',
                            'role': msg['role']
                        })

                # Save based on type
                safe_title = re.sub(r'[^\w\s-]', '', metadata.title)[:50]
                base_name = f"{safe_title}_{metadata.id[:8]}"

                if metadata.has_code and content['code_blocks']:
                    self._save_code_files(metadata, content, code_dir, base_name)

                if metadata.has_research:
                    self._save_research_document(metadata, content, research_dir, base_name)

            except Exception as e:
                logger.error(f"Error extracting from {metadata.title}: {e}")
                continue

    def cleanup(self):
        """Clean up resources"""
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass


def ask_question(question, options=None, default=None, password=False):
    """Ask user a question and get input"""
    print(f"\n{question}")

    if options:
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        if default:
            default_idx = options.index(default) + 1 if default in options else None
            if default_idx:
                print(f"\n(Press Enter for default: {default})")
    elif default:
        print(f"(Press Enter for default: {default})")

    if password:
        import getpass
        answer = getpass.getpass("Answer: ")
    else:
        answer = input("Answer: ").strip()

    if not answer and default:
        return default

    if options and answer.isdigit():
        idx = int(answer) - 1
        if 0 <= idx < len(options):
            return options[idx]

    return answer


def interactive_configuration():
    """Interactive configuration through questions"""
    print("\n" + "="*70)
    print("🤖 ChatGPT Analyzer - Interactive Configuration")
    print("="*70)

    if DOTENV_AVAILABLE and os.path.exists('.env'):
        print("\n✅ Found .env file - loading configuration...")
        print("   (You can override these settings below)")
    elif not DOTENV_AVAILABLE:
        print("\n💡 Tip: Install python-dotenv to use .env files:")
        print("   pip install python-dotenv")

    print("\nI'll ask you a few questions to configure the analyzer.")
    print("Press Ctrl+C at any time to cancel.\n")

    config = {}

    # Question 1: Mode selection
    mode_options = ["Export mode (analyze downloaded JSON file)", "Live mode (browse ChatGPT in real-time)"]
    mode_answer = ask_question(
        "How would you like to analyze your ChatGPT conversations?",
        options=mode_options,
        default=mode_options[0]
    )
    config['mode'] = 'export' if 'Export' in mode_answer else 'live'

    # Mode-specific questions
    if config['mode'] == 'export':
        print("\n📥 Export Mode Selected")
        print("You'll need your ChatGPT export file (conversations.json)")
        print("To get it: ChatGPT → Settings → Data Controls → Export data")

        default_file = os.getenv('EXPORT_FILE', 'conversations.json')

        while True:
            file_path = ask_question(
                "\nWhat's the path to your conversations.json file?",
                default=default_file
            )

            # Handle drag-and-drop quotes
            file_path = file_path.strip('"').strip("'")

            if os.path.exists(file_path):
                config['file'] = file_path
                file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
                print(f"✅ Found file ({file_size:.1f} MB)")
                break
            else:
                print(f"❌ File not found: {file_path}")
                retry = ask_question("Try again?", options=["Yes", "No"], default="Yes")
                if retry == "No":
                    return None

    else:  # Live mode
        print("\n🌐 Live Mode Selected")
        print("I'll need your ChatGPT credentials to browse your conversations.")
        print("(Credentials are only used for this session and not stored)")

        # Check environment variables first
        env_email = os.getenv('CHATGPT_EMAIL')
        env_password = os.getenv('CHATGPT_PASSWORD')

        if env_email:
            use_env = ask_question(
                f"\nFound email in environment: {env_email}\nUse this email?",
                options=["Yes", "No"],
                default="Yes"
            )
            if use_env == "Yes":
                config['email'] = env_email
            else:
                config['email'] = ask_question("Enter your ChatGPT email:")
        else:
            config['email'] = ask_question("\nEnter your ChatGPT email:")

        if env_password:
            config['password'] = env_password
            print("✅ Using password from environment variable")
        else:
            config['password'] = ask_question(
                "Enter your ChatGPT password:",
                password=True
            )

        # Headless mode
        default_headless = os.getenv('HEADLESS_MODE', 'true').lower() == 'true'
        headless_default = "Yes (faster)" if default_headless else "No (see what's happening)"

        headless = ask_question(
            "\nRun browser in background (headless mode)?",
            options=["Yes (faster)", "No (see what's happening)"],
            default=headless_default
        )
        config['headless'] = 'Yes' in headless

    # Question 2: Output directory
    default_output = os.getenv('OUTPUT_DIR', 'chatgpt_analysis')
    output_dir = ask_question(
        "\nWhere should I save the analysis results?",
        default=default_output
    )
    config['output_dir'] = output_dir

    # Question 3: Report formats
    print("\n📊 Report Formats")
    print("Which report formats would you like?")

    format_options = [
        "HTML only (beautiful interactive report)",
        "HTML + CSV (report + spreadsheet)",
        "All formats (HTML, CSV, JSON, Markdown" + (", PDF" if REPORTLAB_AVAILABLE else "") + ")",
        "Custom selection"
    ]

    format_answer = ask_question(
        "Select format option:",
        options=format_options,
        default=format_options[0]
    )

    if "HTML only" in format_answer:
        config['formats'] = ['html']
    elif "HTML + CSV" in format_answer:
        config['formats'] = ['html', 'csv']
    elif "All formats" in format_answer:
        config['formats'] = ['html', 'csv', 'json', 'markdown']
        if REPORTLAB_AVAILABLE:
            config['formats'].append('pdf')
    else:
        # Custom selection
        available_formats = ['html', 'csv', 'json', 'markdown']
        if REPORTLAB_AVAILABLE:
            available_formats.append('pdf')

        selected_formats = []
        print("\nSelect formats (enter numbers separated by commas):")
        for i, fmt in enumerate(available_formats, 1):
            descriptions = {
                'html': 'Interactive web report',
                'csv': 'Excel-compatible spreadsheet',
                'json': 'Machine-readable data',
                'markdown': 'Readable text document',
                'pdf': 'Professional PDF report'
            }
            print(f"  {i}. {fmt.upper()} - {descriptions.get(fmt, '')}")

        selection = ask_question("Your selection (e.g., 1,2,3):", default="1")

        for num in selection.split(','):
            try:
                idx = int(num.strip()) - 1
                if 0 <= idx < len(available_formats):
                    selected_formats.append(available_formats[idx])
            except ValueError:
                pass

        config['formats'] = selected_formats if selected_formats else ['html']

    # Question 4: Content extraction
    extract = ask_question(
        "\nWould you like to extract code and research content from conversations?",
        options=["Yes", "No", "Ask me after analysis"],
        default="Ask me after analysis"
    )

    if extract == "Yes":
        config['extract'] = True
        config['extract_immediate'] = True
    elif extract == "No":
        config['extract'] = False
        config['extract_immediate'] = False
    else:
        config['extract'] = True
        config['extract_immediate'] = False

    # Question 5: Advanced options
    show_advanced = ask_question(
        "\nWould you like to configure advanced options?",
        options=["No (use defaults)", "Yes"],
        default="No (use defaults)"
    )

    if "Yes" in show_advanced:
        # Caching
        use_cache = ask_question(
            "\nUse caching for faster re-analysis?",
            options=["Yes (recommended)", "No"],
            default="Yes (recommended)"
        )
        config['use_cache'] = 'Yes' in use_cache

        # Debug mode
        debug = ask_question(
            "\nEnable debug logging?",
            options=["No", "Yes"],
            default="No"
        )
        config['debug'] = debug == "Yes"
    else:
        config['use_cache'] = True
        config['debug'] = False

    # Confirmation
    print("\n" + "="*70)
    print("📋 Configuration Summary")
    print("="*70)
    print(f"\nMode: {config['mode'].upper()}")

    if config['mode'] == 'export':
        print(f"Input file: {os.path.basename(config['file'])}")
    else:
        print(f"Email: {config['email']}")
        print(f"Browser mode: {'Headless' if config.get('headless', True) else 'Visible'}")

    print(f"Output directory: {config['output_dir']}")
    print(f"Report formats: {', '.join(config['formats']).upper()}")
    print(f"Extract content: {'Yes' if config.get('extract_immediate', False) else 'After analysis' if config.get('extract', False) else 'No'}")

    confirm = ask_question(
        "\nReady to start the analysis?",
        options=["Yes, let's go!", "No, start over", "Cancel"],
        default="Yes, let's go!"
    )

    if confirm == "No, start over":
        return interactive_configuration()
    elif confirm == "Cancel":
        return None

    return config


def main():
    """Main entry point with interactive configuration"""
    # Check if running with command line arguments
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        # Legacy command line mode
        print("Note: This script now uses interactive configuration by default.")
        print("To use command line arguments, use --cli flag")

    # Allow --cli flag to use old argument parser
    if '--cli' in sys.argv:
        return main_cli()

    try:
        # Interactive configuration
        config = interactive_configuration()

        if not config:
            print("\n👋 Configuration cancelled. Goodbye!")
            return 0

        # Configure logging
        if config.get('debug'):
            logging.basicConfig(level=logging.DEBUG)

        # Create analyzer
        analyzer = ChatGPTAnalyzer(
            mode=config['mode'],
            export_file=config.get('file'),
            output_dir=config['output_dir'],
            use_cache=config.get('use_cache', True)
        )

        print("\n" + "="*70)
        print("🚀 Starting Analysis")
        print("="*70 + "\n")

        # Run analysis
        if config['mode'] == 'export':
            conversations = analyzer.extract_conversations_from_export()
        else:
            analyzer.setup_driver(headless=config.get('headless', True))

            if analyzer.login(config['email'], config['password']):
                conversations = analyzer.extract_conversations_from_browser()
            else:
                raise Exception("Login failed")

        # Generate reports
        if conversations:
            print(f"\n✅ Found {len(conversations)} conversations")

            relevant = [c for c in conversations
                       if c.conversation_type != ConversationType.GENERAL]
            print(f"📊 {len(relevant)} relevant conversations (with code/research/links)")

            print("\n📝 Generating reports...")
            analyzer.generate_reports(conversations, config['formats'])

            # Extract content if requested
            if config.get('extract') and relevant:
                if config.get('extract_immediate'):
                    # Extract all relevant conversations
                    print(f"\n📤 Extracting content from {len(relevant)} conversations...")
                    selected_ids = [c.id for c in relevant]
                    analyzer.extract_code_and_research(conversations, selected_ids)
                else:
                    # Interactive selection
                    extract_now = ask_question(
                        "\nWould you like to extract code/research content now?",
                        options=["Yes", "No"],
                        default="Yes"
                    )

                    if extract_now == "Yes":
                        print("\n" + "="*60)
                        print("SELECT CONVERSATIONS FOR EXTRACTION")
                        print("="*60 + "\n")

                        # Display conversations
                        for i, conv in enumerate(relevant, 1):
                            type_icon = {
                                ConversationType.CODE: "💻",
                                ConversationType.RESEARCH: "🔬",
                                ConversationType.MIXED: "🔀",
                                ConversationType.LINKS: "🔗"
                            }.get(conv.conversation_type, "📝")

                            print(f"{i:3d}. {type_icon} {conv.title[:60]}")
                            if conv.languages:
                                print(f"      Languages: {', '.join(conv.languages[:3])}")

                        print(f"\nEnter conversation numbers (e.g., 1,3,5-10)")
                        print("Or type 'all' for all, 'none' to skip")

                        selection = input("\nYour selection: ").strip().lower()

                        selected_ids = []
                        if selection == 'all':
                            selected_ids = [c.id for c in relevant]
                        elif selection != 'none':
                            # Parse selection (supports ranges)
                            for part in selection.split(','):
                                part = part.strip()
                                if '-' in part:
                                    # Range
                                    try:
                                        start, end = map(int, part.split('-'))
                                        for i in range(start-1, min(end, len(relevant))):
                                            if 0 <= i < len(relevant):
                                                selected_ids.append(relevant[i].id)
                                    except ValueError:
                                        pass
                                else:
                                    # Single number
                                    try:
                                        idx = int(part) - 1
                                        if 0 <= idx < len(relevant):
                                            selected_ids.append(relevant[idx].id)
                                    except ValueError:
                                        pass

                        if selected_ids:
                            print(f"\n📤 Extracting {len(selected_ids)} conversations...")
                            analyzer.extract_code_and_research(conversations, selected_ids)

        else:
            print("\n❌ No conversations found")

        print("\n" + "="*70)
        print("✨ Analysis Complete!")
        print("="*70)
        print(f"\n📁 Results saved to: {os.path.abspath(config['output_dir'])}")

        # Send notification if configured
        if 'analyzer' in locals():
            analyzer.send_notification(
                "ChatGPT Analysis Complete",
                f"Analysis completed successfully!\n\n"
                f"Total conversations: {len(conversations) if 'conversations' in locals() else 0}\n"
                f"Output directory: {os.path.abspath(config['output_dir'])}"
            )

        # Try to open HTML report if generated
        if 'html' in config['formats']:
            html_files = list(Path(config['output_dir']).glob('*.html'))
            if html_files:
                newest_html = max(html_files, key=os.path.getctime)
                try:
                    import webbrowser
                    webbrowser.open(f'file://{newest_html.absolute()}')
                    print("📂 Opening HTML report in your browser...")
                except:
                    print(f"📂 Open this file in your browser: {newest_html}")

    except KeyboardInterrupt:
        print("\n\n✋ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if config.get('debug'):
            import traceback
            traceback.print_exc()
        return 1
    finally:
        if 'analyzer' in locals():
            analyzer.cleanup()

    return 0


def main_cli():
    """Legacy CLI mode with argparse"""
    import argparse

    parser = argparse.ArgumentParser(
        description='ChatGPT Universal Analyzer - Analyze and extract your ChatGPT conversations'
    )

    parser.add_argument('--mode', choices=['export', 'live'], default='export',
                      help='Analysis mode (default: export)')
    parser.add_argument('--file', help='Path to conversations.json for export mode')
    parser.add_argument('--email', help='ChatGPT email for live mode')
    parser.add_argument('--password', help='ChatGPT password')
    parser.add_argument('--output-dir', default='chatgpt_analysis',
                      help='Output directory')
    parser.add_argument('--formats', nargs='+',
                      choices=['html', 'csv', 'json', 'markdown', 'pdf', 'all'],
                      default=['html'])
    parser.add_argument('--extract', action='store_true',
                      help='Extract code and research content')
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    # Rest of the old main() code...
    # [Previous argparse-based implementation]
    print("CLI mode is deprecated. Please remove --cli flag to use interactive mode.")
    return 1


if __name__ == "__main__":
    # Show environment configuration status
    if DOTENV_AVAILABLE and os.path.exists('.env'):
        print("✅ Using configuration from .env file")
    else:
        print("💡 Tip: Create a .env file for easier configuration")
        print("   Run: python setup_env.py")

    sys.exit(main())

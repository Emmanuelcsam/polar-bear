"""
Synapse Tools: A consolidated backend for analysis, AI assistance, and auto-healing.
This module provides the core logic for the IDE's advanced features, offering a
clean interface for the main application to use.
"""

import subprocess
import os
import sys
import json
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# --- Configuration for API Keys ---
# The API key will be stored in a file within the app's config directory for persistence.
CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".synapse_ide")
API_KEY_FILE = os.path.join(CONFIG_DIR, "openai.key")

def save_api_key(api_key: str):
    """Saves the OpenAI API key to a local file for persistence."""
    os.makedirs(os.path.dirname(API_KEY_FILE), exist_ok=True)
    with open(API_KEY_FILE, 'w') as f:
        f.write(api_key)

def load_api_key() -> str | None:
    """Loads the OpenAI API key from the local file."""
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, 'r') as f:
            return f.read().strip()
    return os.getenv("OPENAI_API_KEY") # Fallback to environment variable

# --- Code Analysis ---

class CodeAnalyzer:
    """
    Analyzes Python code using Pylint to find errors, warnings, and style issues.
    It returns structured data that's easy for the UI to display.
    """
    def __init__(self, code_content: str):
        """Initializes with the code content, which is written to a temporary file for analysis."""
        self.temp_file_path = "temp_script_for_analysis.py"
        with open(self.temp_file_path, "w", encoding="utf-8") as f:
            f.write(code_content)

    def analyze(self) -> dict:
        """
        Runs Pylint and returns a structured dictionary of the findings.
        """
        print("Running static analysis with Pylint...")
        results = {"score": 10.0, "summary": "Excellent! No issues found.", "issues": []}
        
        try:
            from pylint.lint import Run
            from pylint.reporters.text import TextReporter
            from io import StringIO

            pylint_output = StringIO()
            reporter = TextReporter(pylint_output)
            
            # Run Pylint with a focused set of checks.
            Run([self.temp_file_path, "--errors-only", "--disable=missing-docstring"], reporter=reporter, do_exit=False)
            
            output_str = pylint_output.getvalue()
            
            # Parse Pylint output into a structured list of issues
            for line in output_str.splitlines():
                if ":" in line and any(c in line for c in "WCREF"): # Standard Pylint message codes
                    parts = line.split(':')
                    if len(parts) >= 4 and parts[1].strip().isdigit():
                        issue = {
                            "line": int(parts[1].strip()),
                            "type_code": parts[2].strip(),
                            "message": ":".join(parts[3:]).strip()
                        }
                        results["issues"].append(issue)
            
            # Update summary based on findings
            if results["issues"]:
                error_count = sum(1 for i in results["issues"] if i['type_code'].startswith('E'))
                warning_count = len(results["issues"]) - error_count
                results["summary"] = f"Found {error_count} error(s) and {warning_count} warning(s)."

        except ImportError:
            results["summary"] = "Pylint not found. Please install it to enable code analysis."
        except Exception as e:
            results["summary"] = f"An unexpected error occurred during analysis: {e}"
        finally:
            if os.path.exists(self.temp_file_path):
                os.remove(self.temp_file_path)
        
        return results

# --- AI-Powered Assistance ---

class AISuggestor:
    """Provides code suggestions and improvements using OpenAI's models."""
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or load_api_key()
        self.openai_client = None
        if self.api_key:
            try:
                self.openai_client = OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")

    def is_ready(self) -> bool:
        """Checks if the suggestor is configured with an API key."""
        return self.openai_client is not None

    def get_suggestion(self, code: str, analysis_issues: list) -> str:
        """
        Requests a code improvement suggestion from OpenAI's GPT-4o-mini model,
        providing context about the code and any issues found by the analyzer.
        """
        if not self.is_ready():
            return "AI Suggestor is not configured. Please provide an OpenAI API key in the settings."

        issue_summary = "\n".join([f"- Line {i['line']}: {i['message']} ({i['type_code']})" for i in analysis_issues])
        prompt = f"""
        You are an expert Python programming assistant. Your task is to fix and improve the following code snippet from an IDE.

        Please adhere to these rules:
        1. Correct the specific issues listed below.
        2. Improve the code's readability and efficiency, following PEP 8 standards.
        3. Add comments only where the logic is complex or non-obvious.
        4. Return ONLY the complete, corrected Python code. Do not include any explanations, markdown, or other text.

        --- START OF CODE ---
        {code}
        --- END OF CODE ---

        --- ISSUES FOUND ---
        {issue_summary if issue_summary else "No specific issues found, but please review for general improvements."}
        --- END OF ISSUES ---
        """
        
        try:
            print("Sending request to OpenAI for code suggestion...")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            suggestion = response.choices[0].message.content.strip()
            # Clean up the response to ensure it's just code
            if suggestion.startswith("```python"):
                suggestion = suggestion.split("```python\n", 1)[1].rsplit("\n```", 1)[0]
            return suggestion
        
        except Exception as e:
            return f"# AI Error: Could not get a suggestion.\n# Reason: {e}"

# --- Auto-Healing for Runtime Errors ---

class AutoHealer:
    """Diagnoses and proposes fixes for runtime errors by combining execution analysis, web research, and AI."""
    def __init__(self, file_path: str, api_key: str | None = None):
        self.file_path = file_path
        self.api_key = api_key or load_api_key()
        self.openai_client = OpenAI(api_key=self.api_key) if self.api_key else None

    def is_ready(self) -> bool:
        """Checks if the healer is configured with an API key."""
        return self.openai_client is not None

    def diagnose_runtime_error(self) -> tuple[str | None, str | None]:
        """Runs the script to capture stdout and, more importantly, any runtime errors (stderr)."""
        print(f"Diagnosing runtime error for: {self.file_path}")
        try:
            result = subprocess.run(
                [sys.executable, self.file_path],
                capture_output=True, text=True, check=False, encoding='utf-8'
            )
            return result.stdout, result.stderr
        except FileNotFoundError:
            return None, f"Healing Error: The script at {self.file_path} was not found."
        except Exception as e:
            return None, f"Healing Error: A system error occurred while running the script: {e}"

    def research_error_online(self, error_message: str) -> str:
        """Performs a web search for the error message, focusing on Stack Overflow for context."""
        if not error_message: return "No error message to research."

        # Extract the most descriptive line from the traceback for a better search query
        last_line = error_message.strip().split('\n')[-1]
        query = f"site:stackoverflow.com python {last_line}"
        print(f"Researching online for: '{query}'")

        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get("https://www.google.com/search", params={'q': query}, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            link_tag = soup.find('a', href=lambda href: href and "[stackoverflow.com/questions](https://stackoverflow.com/questions)" in href)
            if not link_tag: return "Could not find a relevant Stack Overflow page."

            url = link_tag['href'].split('&')[0].replace('/url?q=', '')
            print(f"Found potential solution: {url}")
            return f"Context found at: {url}" # For now, just return the URL, the AI can use the error message primarily.
        except requests.exceptions.RequestException as e:
            return f"Web search failed: {e}"

    def propose_fix(self, original_code: str, error_message: str, research_context: str) -> str:
        """Uses OpenAI to synthesize a fix based on the code, error, and online research."""
        if not self.is_ready():
            return "# Auto-Heal Error: AI is not configured. Please set your OpenAI API key."

        prompt = f"""
        You are an automated Python debugging assistant. Your task is to fix a script that failed with a runtime error.

        Here is the original code that caused the error:
        ---
        {original_code}
        ---

        When executed, it produced this error traceback:
        ---
        {error_message}
        ---

        An automated web search found this potentially related context:
        ---
        {research_context}
        ---

        Based on all this information, please generate the complete, corrected Python code.
        Your output must ONLY be the Python code itself, without any extra explanations or markdown formatting.
        """
        print("Generating AI-powered fix for runtime error...")
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            fixed_code = response.choices[0].message.content.strip()
            if fixed_code.startswith("```python"):
                fixed_code = fixed_code.split("```python\n", 1)[1].rsplit("\n```", 1)[0]
            return fixed_code
        except Exception as e:
            return f"# Auto-Heal Error: The AI could not generate a fix.\n# Reason: {e}"

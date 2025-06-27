"""
AI Assist: Simplified AI-powered code analysis and suggestions for Neural Weaver.
This module integrates tools like Pylint and OpenAI to provide actionable,
easy-to-understand feedback on the Python code inside the Blocks.
"""

import subprocess
import os
from openai import OpenAI
import json

# --- Configuration ---
# Store the API key in a more persistent way across sessions.
# This avoids asking the user for the key every time they start the app.
API_KEY_FILE = os.path.join(os.path.expanduser("~"), ".neural_weaver", "openai.key")

def save_api_key(api_key: str):
    """Saves the OpenAI API key to a local file."""
    os.makedirs(os.path.dirname(API_KEY_FILE), exist_ok=True)
    with open(API_KEY_FILE, 'w') as f:
        f.write(api_key)

def load_api_key() -> str | None:
    """Loads the OpenAI API key from a local file."""
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, 'r') as f:
            return f.read().strip()
    return None

class BlockAnalyzer:
    """
    Analyzes the code within a block for quality, style, and potential errors.
    This replaces the original 'CodeAnalyzer' with a focus on simplicity.
    """
    def __init__(self, code_content: str):
        """
        Initializes the analyzer with the code to be analyzed.
        It writes the code to a temporary file for tool processing.
        """
        # Create a temporary file to run analysis tools against.
        self.temp_file_path = "temp_block_code.py"
        with open(self.temp_file_path, "w", encoding="utf-8") as f:
            f.write(code_content)

    def analyze(self) -> dict:
        """
        Runs a suite of analysis tools and returns a structured,
        user-friendly summary of the findings.
        """
        print("Running static analysis on block code...")
        results = {
            "score": 10.0,
            "summary": "Excellent! No issues found.",
            "issues": []
        }
        
        try:
            # We use Pylint as it provides a score and detailed feedback.
            process = subprocess.run(
                ["pylint", self.temp_file_path],
                capture_output=True,
                text=True,
                check=False # Don't throw an error on non-zero exit code
            )
            
            pylint_output = process.stdout
            
            # --- Parse Pylint Output ---
            # Find the score
            score_line = [line for line in pylint_output.split('\n') if "Your code has been rated at" in line]
            if score_line:
                score_str = score_line[0].split('/')[0].split('at ')[-1].strip()
                try:
                    results["score"] = float(score_str)
                except ValueError:
                    pass # Keep default score if parsing fails
            
            # Find issues (warnings and errors)
            issues_found = []
            for line in pylint_output.split('\n'):
                if ":" in line and any(c in line for c in "WCREF"):
                    parts = line.split(':')
                    if len(parts) >= 4 and parts[1].strip().isdigit():
                        issue = {
                            "line": int(parts[1].strip()),
                            "type": parts[2].strip(),
                            "message": ":".join(parts[3:]).strip()
                        }
                        issues_found.append(issue)
            
            results["issues"] = issues_found
            
            # Update summary based on score and issues
            if not issues_found:
                 results["summary"] = "Excellent! No issues found."
            else:
                error_count = sum(1 for i in issues_found if i['type'].startswith('E'))
                warning_count = len(issues_found) - error_count
                results["summary"] = f"Found {error_count} error(s) and {warning_count} warning(s)."

        except FileNotFoundError:
            results["summary"] = "Pylint not found. Please install it to enable code analysis."
            results["issues"].append({"line": 0, "type": "Setup Error", "message": "The 'pylint' command was not found in your system's PATH."})
        except Exception as e:
            results["summary"] = f"An unexpected error occurred during analysis."
            results["issues"].append({"line": 0, "type": "System Error", "message": str(e)})
            
        finally:
            # Clean up the temporary file
            if os.path.exists(self.temp_file_path):
                os.remove(self.temp_file_path)
        
        return results

class AISuggestor:
    """
    Provides code suggestions and improvements using OpenAI's models.
    Manages the API key and provides a clean interface for getting suggestions.
    """
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or load_api_key()
        self.openai_client = None
        if self.api_key:
            try:
                self.openai_client = OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")

    def is_ready(self) -> bool:
        """Checks if the AI suggestor is configured and ready to use."""
        return self.openai_client is not None

    def get_suggestion(self, code: str, analysis_issues: list) -> str:
        """

        Gets a code improvement suggestion from OpenAI's GPT-4o-mini model.
        The prompt is context-aware, including the code and the issues found.
        """
        if not self.is_ready():
            return "AI Suggestor is not configured. Please provide an OpenAI API key in the settings."

        # Create a summary of the issues for the prompt
        issue_summary = "\n".join([f"- Line {i['line']}: {i['message']} ({i['type']})" for i in analysis_issues])

        prompt = f"""
        You are an expert Python programmer assisting in a visual programming environment.
        A user has written the following code for a 'block'. Please improve it.

        Your task is to:
        1. Fix the specific issues identified below.
        2. Improve the code's readability and efficiency.
        3. Adhere to Python best practices (PEP 8).
        4. Add comments where the logic is complex.
        
        ONLY return the complete, corrected Python code. Do not include any explanations, markdown, or intro/outro text.

        --- START OF CODE ---
        {code}
        --- END OF CODE ---

        --- ISSUES FOUND ---
        {issue_summary if issue_summary else "No specific issues were found, but please review for general improvements."}
        --- END OF ISSUES ---

        Corrected Code:
        """
        
        try:
            print("Sending request to OpenAI for code suggestion...")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            suggestion = response.choices[0].message.content
            # Clean up the response to ensure it's only code
            if suggestion.strip().startswith("```python"):
                suggestion = suggestion.split("```python\n")[1].split("\n```")[0]
            return suggestion
        
        except Exception as e:
            return f"# AI Error: Could not get a suggestion.\n# Reason: {e}"

# --- Example Usage ---
if __name__ == '__main__':
    # This demonstrates how the new AI Assist module works.
    
    # Example code with some issues
    bad_code = """
import sys
def   MyFunction( name,age):
     print("hello "+name)
     if age>18:
        return True
    else:
         return False
unused_variable = 5
"""
    
    print("--- Testing BlockAnalyzer ---")
    analyzer = BlockAnalyzer(bad_code)
    analysis_result = analyzer.analyze()
    print(json.dumps(analysis_result, indent=2))
    
    print("\n--- Testing AISuggestor ---")
    # To run this part, you need to set your OpenAI API key as an environment variable
    # or be prompted to enter it.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Skipping AISuggestor test: OPENAI_API_KEY environment variable not set.")
    else:
        suggester = AISuggestor(api_key=api_key)
        if suggester.is_ready():
            suggestion = suggester.get_suggestion(bad_code, analysis_result['issues'])
            print("\n--- Original Code ---")
            print(bad_code)
            print("\n--- AI Suggested Code ---")
            print(suggestion)
        else:
            print("Could not initialize AI Suggestor.")

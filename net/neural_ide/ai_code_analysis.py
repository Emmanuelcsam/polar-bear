import subprocess
import os
from openai import OpenAI
import json

class CodeAnalyzer:
    """
    A class to handle code analysis using various tools like Pylint, Flake8, etc.
    """
    def __init__(self, file_path):
        self.file_path = file_path

    def run_pylint(self):
        """Runs Pylint on the specified file."""
        try:
            result = subprocess.run(
                ["pylint", self.file_path],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            return f"Pylint analysis failed: {e}"

    def run_flake8(self):
        """Runs Flake8 on the specified file."""
        try:
            result = subprocess.run(
                ["flake8", self.file_path],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            return f"Flake8 analysis failed: {e}"

class AISuggestor:
    """
    A class to get suggestions from AI tools like GitHub Copilot and OpenAI.
    """
    def __init__(self, api_key=None):
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)

    def get_copilot_suggestion(self, prompt):
        """Gets a suggestion from GitHub Copilot CLI."""
        try:
            proc = subprocess.run(
                ["gh", "copilot", "suggest", "--stdin"],
                input=prompt,
                text=True,
                capture_output=True,
                check=True
            )
            return proc.stdout
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            return f"GitHub Copilot CLI error: {e}"

    def get_openai_suggestion(self, prompt):
        """Gets a suggestion from OpenAI's GPT-4o-mini model."""
        if not hasattr(self, 'openai_client'):
            return "OpenAI client not initialized. Please provide an API key."
            
        try:
            resp = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"OpenAI API error: {e}"

class AutoReviser:
    """
    A class to automatically revise code based on AI suggestions.
    """
    def __init__(self, file_path):
        self.file_path = file_path

    def revise_code(self, original_code, suggestion):
        """
        Revises the code by replacing the original with the suggestion.
        This is a simple implementation; a more advanced version could use diff-patch.
        """
        with open(self.file_path, 'w') as f:
            f.write(suggestion)
        return "Code has been revised."

def main():
    """
    Main function to demonstrate the usage of the new classes.
    """
    # Configuration
    USE_AI_CHECKER = True  # Set to False to disable AI features
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Securely get your API key

    # Example usage
    file_to_analyze = "your_script.py"  # Change this to the script you want to analyze
    
    # Create a dummy file for demonstration
    with open(file_to_analyze, "w") as f:
        f.write("def my_func( a, b):\n    return a+b")

    analyzer = CodeAnalyzer(file_to_analyze)
    print("--- Pylint Analysis ---")
    print(analyzer.run_pylint())
    print("\n--- Flake8 Analysis ---")
    print(analyzer.run_flake8())

    if USE_AI_CHECKER:
        ai_suggestor = AISuggestor(api_key=OPENAI_API_KEY)
        reviser = AutoReviser(file_to_analyze)

        with open(file_to_analyze, 'r') as f:
            original_code = f.read()

        # Get suggestion from Copilot
        copilot_prompt = f"Fix the following Python code:\n\n{original_code}"
        copilot_suggestion = ai_suggestor.get_copilot_suggestion(copilot_prompt)
        print("\n--- GitHub Copilot Suggestion ---")
        print(copilot_suggestion)

        # Get suggestion from OpenAI
        openai_prompt = f"Fix the following Python code to adhere to PEP 8 and best practices:\n\n{original_code}"
        openai_suggestion = ai_suggestor.get_openai_suggestion(openai_prompt)
        print("\n--- OpenAI Suggestion ---")
        print(openai_suggestion)

        # Automatically revise the code with OpenAI's suggestion
        print("\n--- Auto-Revision ---")
        revision_status = reviser.revise_code(original_code, openai_suggestion)
        print(revision_status)

        # Show the revised code
        with open(file_to_analyze, 'r') as f:
            revised_code = f.read()
        print("\n--- Revised Code ---")
        print(revised_code)

if __name__ == "__main__":
    main()


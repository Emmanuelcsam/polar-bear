"""
Auto Healer: An intelligent tool for automatically diagnosing and fixing
runtime errors in Neural Weaver blocks. This module simplifies the original
'AutoFixer' by focusing on a clear, AI-driven workflow.
"""

import subprocess
import sys
import os
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from ai_assist import load_api_key # Use the shared API key loader

class AutoHealer:
    """
    Diagnoses and proposes fixes for runtime errors in a Python script.
    It combines runtime analysis, web searching for context, and AI for a solution.
    """
    def __init__(self, file_path: str, api_key: str | None = None):
        """
        Initializes the AutoHealer.

        Args:
            file_path (str): The absolute path to the Python script to be healed.
            api_key (str, optional): OpenAI API key. If not provided, it will be loaded from the shared store.
        """
        self.file_path = file_path
        self.api_key = api_key or load_api_key()
        self.openai_client = OpenAI(api_key=self.api_key) if self.api_key else None

    def is_ready(self) -> bool:
        """Checks if the AutoHealer has the necessary API key to function."""
        return self.openai_client is not None

    def diagnose(self) -> tuple[str | None, str | None]:
        """
        Runs the script in a separate process to capture its output and, more
        importantly, any runtime errors.

        Returns:
            A tuple containing (stdout, stderr). If an error occurs, stderr
            will contain the traceback.
        """
        print(f"Diagnosing script: {self.file_path}")
        try:
            result = subprocess.run(
                [sys.executable, self.file_path],
                capture_output=True,
                text=True,
                check=False, # We want to capture errors, not crash the launcher
                encoding='utf-8'
            )
            return result.stdout, result.stderr
        except FileNotFoundError:
            return None, f"Healing Error: The script was not found at {self.file_path}."
        except Exception as e:
            return None, f"Healing Error: A system-level error occurred while running the script: {e}"

    def research_error(self, error_message: str) -> str:
        """
        Performs a targeted web search for the error message to find context,
        focusing on Stack Overflow for reliable solutions.

        Args:
            error_message (str): The stderr output from the script execution.

        Returns:
            A summary of the most relevant solution found, or a message indicating
            no solution was found.
        """
        if not error_message:
            return "No error message to research."

        # Extract the most relevant line from the error for a cleaner search
        last_line = error_message.strip().split('\n')[-1]
        query = f"site:stackoverflow.com python {last_line}"
        print(f"Researching online for: '{query}'")

        try:
            # Use a common user agent to avoid being blocked
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
            response = requests.get("[https://www.google.com/search](https://www.google.com/search)", params={'q': query}, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            link_tag = soup.find('a', href=lambda href: href and "[stackoverflow.com/questions](https://stackoverflow.com/questions)" in href)

            if not link_tag:
                return "Could not find a relevant Stack Overflow page from the search results."

            url = link_tag['href']
            if url.startswith('/url?q='):
                url = url.split('&')[0].replace('/url?q=', '')

            print(f"Found potential solution at: {url}")
            return self._scrape_solution(url)
        except requests.exceptions.RequestException as e:
            return f"Web search failed. Could not connect to the internet to find a solution. Error: {e}"

    def _scrape_solution(self, url: str) -> str:
        """Scrapes the highest-voted answer from a Stack Overflow page."""
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the answer with the highest vote count
            answer = soup.find('div', class_='answer', 'data-answerid'=True)
            if answer:
                # Extract the code blocks and text for context
                code_elements = answer.select('.s-prose js-post-body pre code')
                solution_text = "\n".join([code.get_text() for code in code_elements])
                return solution_text if solution_text else "Found a solution page, but could not extract code snippets."
            return "Could not find a clear answer on the Stack Overflow page."
        except requests.exceptions.RequestException as e:
            return f"Failed to access the solution page. Error: {e}"

    def propose_fix(self, original_code: str, error_message: str, research_summary: str) -> str:
        """
        Uses OpenAI's AI to synthesize a fix based on the code, the error, and
        the online research.

        Args:
            original_code (str): The full original code of the script.
            error_message (str): The error captured during diagnosis.
            research_summary (str): The context gathered from web research.

        Returns:
            The AI-generated, fully corrected code.
        """
        if not self.is_ready():
            return "# Auto-Heal Error: AI is not configured. Please set your OpenAI API key."

        prompt = f"""
        You are an automated Python debugging assistant. Your task is to fix a script that has a runtime error.

        Here is the original code:
        ---
        {original_code}
        ---

        When executed, it produced this error:
        ---
        {error_message}
        ---

        An automated search for a solution found this related information from Stack Overflow:
        ---
        {research_summary}
        ---

        Based on all the provided information, please generate the complete, corrected Python code.
        Your output should ONLY be the Python code itself, without any extra explanations or markdown formatting.
        """
        print("Generating AI-powered fix...")
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            fixed_code = response.choices[0].message.content.strip()
            # Clean up potential markdown code fences
            if fixed_code.startswith("```python"):
                fixed_code = fixed_code.split("```python\n", 1)[1].rsplit("\n```", 1)[0]
            return fixed_code
        except Exception as e:
            return f"# Auto-Heal Error: The AI could not generate a fix.\n# Reason: {e}"

    def apply_fix(self, fixed_code: str) -> str:
        """
        Applies the fix by replacing the original file's content.
        It creates a backup of the original file first.
        """
        try:
            backup_path = self.file_path + ".bak"
            if os.path.exists(self.file_path):
                 os.rename(self.file_path, backup_path)
                 print(f"Created backup of original file at: {backup_path}")
            else:
                 return "Error: Original file not found to apply fix."


            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_code)
            return "✅ Fix applied successfully. Please try running the flow again."
        except Exception as e:
            return f"❌ Error: Failed to write the fix to the file. Reason: {e}"


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Testing AutoHealer ---")
    
    # Create a dummy script with a common runtime error
    script_to_fix = "test_error_script.py"
    with open(script_to_fix, 'w') as f:
        # This will cause an AttributeError because os.get_login() doesn't exist on all platforms
        # and is often empty. A better function is os.getlogin().
        f.write("import os\nprint(f'User: {os.get_logon()}')") 

    # To run this part, you need to set your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nSkipping AutoHealer test: OPENAI_API_KEY environment variable not set.")
    else:
        healer = AutoHealer(script_to_fix, api_key=api_key)

        # 1. Diagnose the script
        stdout, stderr = healer.diagnose()

        if stderr:
            print("\n--- ❗ Error Detected ---")
            print(stderr)

            # 2. Research the error
            print("\n--- 🔍 Researching Error ---")
            solution = healer.research_error(stderr)
            print(f"Research result:\n{solution[:300]}...") # Print first 300 chars

            # 3. Propose a fix
            print("\n--- 🤖 Proposing AI Fix ---")
            with open(script_to_fix, 'r') as f:
                original_code = f.read()
            
            ai_fix = healer.propose_fix(original_code, stderr, solution)
            print("AI's proposed code:\n---")
            print(ai_fix)
            print("---\n")
            
            # 4. Apply the fix
            if "Auto-Heal Error" not in ai_fix:
                status = healer.apply_fix(ai_fix)
                print(status)
                
                # 5. Re-diagnose to verify
                print("\n--- ✅ Verifying Fix ---")
                new_stdout, new_stderr = healer.diagnose()
                if new_stderr:
                    print("Verification failed. The script still has an error:")
                    print(new_stderr)
                else:
                    print("Verification successful! Script now runs without errors.")
                    print(f"New output: {new_stdout}")
        else:
            print("\n--- ✅ No Errors Detected ---")
            print(f"Script output: {stdout}")

    # Clean up test files
    if os.path.exists(script_to_fix):
        os.remove(script_to_fix)
    if os.path.exists(script_to_fix + ".bak"):
        os.remove(script_to_fix + ".bak")

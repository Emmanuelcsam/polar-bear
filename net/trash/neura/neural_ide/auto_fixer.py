import subprocess
import os
import sys
from openai import OpenAI
import requests
from bs4 import BeautifulSoup

class AutoFixer:
    """
    A class to automatically diagnose and fix errors in Python scripts.
    """

    def __init__(self, file_path, api_key=None):
        self.file_path = file_path
        self.openai_client = OpenAI(api_key=api_key) if api_key else None

    def run_script(self):
        """Runs the script and captures its output and errors."""
        try:
            result = subprocess.run(
                [sys.executable, self.file_path],
                capture_output=True,
                text=True,
                check=False  # We want to handle non-zero exit codes ourselves
            )
            return result.stdout, result.stderr
        except FileNotFoundError:
            return None, f"Error: Script not found at {self.file_path}"

    def search_for_fix(self, error_message):
        """Searches for a fix online, focusing on Stack Overflow."""
        if not error_message:
            return "No error message provided."

        query = f"site:stackoverflow.com python {error_message.strip()}"
        print(f"Searching online for: '{query}'")
        
        try:
            response = requests.get("https://www.google.com/search", params={'q': query})
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the first Stack Overflow link
            link_tag = soup.find('a', href=lambda href: href and "stackoverflow.com/questions" in href)
            
            if not link_tag:
                return "Could not find a relevant Stack Overflow link."

            url = link_tag['href']
            # Clean up the URL from Google's redirect
            if url.startswith('/url?q='):
                url = url.split('&')[0].replace('/url?q=', '')

            print(f"Found potential solution at: {url}")
            return self.scrape_stack_overflow_solution(url)
            
        except requests.exceptions.RequestException as e:
            return f"Failed to search for a fix: {e}"

    def scrape_stack_overflow_solution(self, url):
        """Scrapes the highest-voted answer from a Stack Overflow page."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the highest-voted answer
            answer = soup.find('div', class_='answer', 'data-answerid'=True)
            if answer:
                return answer.get_text(separator='\n', strip=True)
            return "Could not find a suitable answer on the page."

        except requests.exceptions.RequestException as e:
            return f"Failed to scrape Stack Overflow page: {e}"

    def get_ai_fix(self, original_code, error_message, online_solution):
        """Uses AI to generate a fix based on the error and online solution."""
        if not self.openai_client:
            return "OpenAI client not initialized. Cannot generate AI fix."

        prompt = f"""
        The following Python code has an error.
        
        Original Code:
        ---
        {original_code}
        ---

        Error Message:
        ---
        {error_message}
        ---

        A potential solution found online suggests:
        ---
        {online_solution}
        ---

        Based on all this information, please provide the fully corrected Python code.
        Only output the code, with no additional explanation.
        """
        
        try:
            resp = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"OpenAI API error during fix generation: {e}"

    def apply_fix(self, fixed_code):
        """Applies the fixed code to the original file."""
        try:
            # Create a backup of the original file
            backup_path = self.file_path + ".bak"
            os.rename(self.file_path, backup_path)
            print(f"Backup of original file created at: {backup_path}")

            with open(self.file_path, 'w') as f:
                f.write(fixed_code)
            return "Fix applied successfully."
        except Exception as e:
            return f"Failed to apply fix: {e}"

def main():
    """Main function to demonstrate the AutoFixer."""
    
    # --- Configuration ---
    # !! IMPORTANT !!
    # This feature uses AI and web scraping and can modify your files.
    # Always review changes and ensure you have backups.
    
    file_to_fix = "test_script.py"  # The script to be fixed
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Create a dummy script with an error for demonstration
    with open(file_to_fix, 'w') as f:
        f.write("import os\n\ndef my_function():\n    print(os.get_login())\n\nmy_function()")

    fixer = AutoFixer(file_to_fix, api_key=OPENAI_API_KEY)

    print(f"--- Running initial script: {file_to_fix} ---")
    stdout, stderr = fixer.run_script()

    if stderr:
        print("\n--- Error Detected ---")
        print(stderr)

        print("\n--- Searching for a fix online ---")
        online_solution = fixer.search_for_fix(stderr)
        print(online_solution)

        if online_solution:
            print("\n--- Generating AI-powered fix ---")
            with open(file_to_fix + ".bak", 'r') if os.path.exists(file_to_fix + ".bak") else open(file_to_fix, 'r') as f:
                original_code = f.read()
                
            ai_generated_fix = fixer.get_ai_fix(original_code, stderr, online_solution)
            print("AI Suggestion:\n", ai_generated_fix)

            print("\n--- Applying the fix ---")
            status = fixer.apply_fix(ai_generated_fix)
            print(status)
            
            if "success" in status.lower():
                print("\n--- Verifying the fix by re-running the script ---")
                new_stdout, new_stderr = fixer.run_script()

                if new_stderr:
                    print("\n--- The script still has errors after the fix ---")
                    print(new_stderr)
                else:
                    print("\n--- Script ran successfully after the fix! ---")
                    print("Output:")
                    print(new_stdout)
    else:
        print("\n--- Script ran successfully with no errors ---")
        print("Output:")
        print(stdout)

if __name__ == "__main__":
    main()

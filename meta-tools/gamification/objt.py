#!/usr/bin/env python3
"""
HabiticaObjectiveTracker - A terminal-based UI for tracking objectives with Habitica integration.
Combines functionality from ScriptFinder and objective_tracker.sh.
"""

import os
import sys
import curses
import subprocess
import json
import re
import time
import datetime
from pathlib import Path
import requests
from urllib.parse import quote
import tempfile
import math
import signal

class HabiticaObjectiveTracker:
    """Terminal UI for tracking objectives with Habitica integration."""
    
    def __init__(self, stdscr):
        self.stdscr = stdscr
        
        # Configuration
        self.save_dir = os.path.expanduser("~/Documents/Writer's Den/Tasks")
        # Consider loading these from environment variables or a config file
        self.habitica_user_id = os.environ.get("HABITICA_USER_ID", "8a9e85dc-d8f5-4c60-86ef-d70a19bf225e")
        self.habitica_api_token = os.environ.get("HABITICA_API_TOKEN", "a4375d21-0a50-4ceb-a412-ebb70e927349")
        self.use_habitica = True
        
        # UI state
        self.todos = []
        self.filtered_todos = []
        self.current_index = 0
        self.search_term = ""
        self.is_searching = False
        self.search_buffer = ""
        self.scroll_offset = 0
        self.status_message = "Starting..."
        self.terminal_output = ["Welcome to HabiticaObjectiveTracker"]
        self.view_mode = "list"  # can be "list", "create", "run"
        
        # For create and run modes
        self.form_fields = []
        self.form_field_index = 0
        self.form_task_list = []
        self.form_subtask_list = []
        self.task_subtask_pairs = []  # Store pairs of tasks and their subtasks
        self.current_task_buffer = ""
        self.current_subtask_buffer = ""
        self.adding_tasks = False
        self.adding_subtasks = False
        self.in_task_subtask_flow = False  # Flag for the interleaved task-subtask flow
        self.current_pair_index = -1  # Index for current task-subtask pair
        
        # For timer functionality
        self.timer_running = False
        self.timer_start = 0
        self.timer_elapsed = 0
        self.timer_paused = False
        self.timer_pause_start = 0
        self.timer_break_time = 0
        
        # Ensure the save directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize curses
        curses.curs_set(0)  # Hide cursor
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)  # Regular text
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_GREEN)  # Selected item
        curses.init_pair(3, curses.COLOR_YELLOW, -1)  # High priority
        curses.init_pair(4, curses.COLOR_BLUE, -1)  # Medium priority
        curses.init_pair(5, curses.COLOR_RED, -1)  # Urgent priority
        curses.init_pair(6, curses.COLOR_CYAN, -1)  # Input fields
        curses.init_pair(7, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Input active
        
        # Get screen dimensions
        self.height, self.width = self.stdscr.getmaxyx()
        
        # Setup resize handler
        signal.signal(signal.SIGWINCH, self.handle_resize)
        
        # Initialize the screen
        self.stdscr.clear()
        self.stdscr.timeout(100)  # Non-blocking input
        
        # Init form fields for create mode
        self.init_form_fields()
        
        # Fetch Habitica todos
        # Validate Habitica credentials
        self.terminal_output.append("Validating Habitica credentials...")
        try:
            headers = {
                "x-api-user": self.habitica_user_id,
                "x-api-key": self.habitica_api_token,
                "Content-Type": "application/json"
            }
            
            test_response = requests.get("https://habitica.com/api/v3/user", headers=headers)
            if test_response.status_code == 200:
                self.terminal_output.append("Habitica credentials valid!")
            else:
                self.terminal_output.append(f"WARNING: Habitica credential test failed with status {test_response.status_code}")
                self.terminal_output.append(f"Response: {test_response.text[:100]}...")
        except Exception as e:
            self.terminal_output.append(f"WARNING: Habitica validation error: {str(e)}")
        try:
            test_response = requests.get(
                "https://habitica.com/api/v3/user", 
                headers=headers,
                timeout=10
            )
            
            if test_response.status_code != 200:
                self.terminal_output.append("ERROR: Cannot proceed with Habitica features")
                self.terminal_output.append("The application will continue in offline mode only")
                self.terminal_output.append(f"Full error: {test_response.text[:200]}")
                self.use_habitica = False  # Disable Habitica integration
                self.status_message = "WARNING: Running in offline mode - Habitica connection failed"
            else:
                user_data = test_response.json().get("data", {})
                username = user_data.get("profile", {}).get("name", "Unknown User")
                self.terminal_output.append(f"Connected to Habitica as: {username}")
        except Exception as e:
            self.terminal_output.append(f"ERROR: Could not validate Habitica connection: {str(e)}")
            self.terminal_output.append("Continuing in offline mode")
            self.use_habitica = False
        # Fetch Habitica todos
        self.fetch_habitica_todos()

    def handle_resize(self, signum, frame):
        """Handle terminal resize events."""
        # Update screen dimensions
        curses.endwin()
        curses.initscr()
        self.height, self.width = self.stdscr.getmaxyx()
        self.stdscr.clear()
        self.stdscr.refresh()

    def log_api_error(self, response, context="API Error"):
        """Log detailed API error information"""
        try:
            error_data = response.json()
            error_msg = error_data.get("message", "Unknown error")
            self.terminal_output.append(f"{context}: {response.status_code} - {error_msg}")
            self.terminal_output.append(f"Full response: {response.text[:300]}...")
        except:
            self.terminal_output.append(f"{context}: {response.status_code} - Could not parse response")
            self.terminal_output.append(f"Raw response: {response.text[:300]}")
    
    def init_form_fields(self):
        """Initialize form fields for objective creation."""
        self.form_fields = [
            {"name": "objective", "label": "What's your objective?", "value": "", "required": True},
            {"name": "difficulty", "label": "What's the difficulty (trivial, easy, medium, hard)?", "value": "medium", "required": True},
            {"name": "priority", "label": "What's the priority (low, medium, high, urgent)?", "value": "medium", "required": True},
            {"name": "due_date", "label": "What's the due date (YYYY-MM-DD, optional)?", "value": "", "required": False},
            {"name": "task_subtask", "label": "Add tasks and subtasks", "value": "", "required": False},
            {"name": "reward", "label": "What's your reward (optional)?", "value": "", "required": False},
            {"name": "worth", "label": "What does this objective mean to you?", "value": "", "required": False},
            {"name": "planned_min", "label": "How long do you plan to work on this (minutes)?", "value": "25", "required": True}
        ]
        self.task_subtask_pairs = []  # Clear existing pairs
        self.current_pair_index = -1
    
    def fetch_habitica_todos(self):
        """Fetch todos from Habitica API with improved error handling."""
        self.todos = []
        self.status_message = "Fetching Habitica todos..."
        self.draw_screen()
        
        # Check if Habitica integration is disabled
        if not self.use_habitica:
            self.status_message = "Habitica integration disabled - working offline"
            self.terminal_output.append("Skipping Habitica API call - offline mode")
            self.filtered_todos = []  # Initialize empty todos list
            return
        
        try:
            headers = {
                "x-api-user": self.habitica_user_id,
                "x-api-key": self.habitica_api_token,
                "Content-Type": "application/json"
            }
            
            self.terminal_output.append(f"Using Habitica credentials - User ID: {self.habitica_user_id[:5]}..., Token: {self.habitica_api_token[:5]}...")
            
            # Use params instead of embedding in URL
            response = requests.get(
                "https://habitica.com/api/v3/tasks/user", 
                params={"type": "todos"},
                headers=headers,
                timeout=10
            )
            
            self.terminal_output.append(f"Response status: {response.status_code}")
            self.terminal_output.append(f"Response preview: {response.text[:100]}...")
                        
            if response.status_code == 200:
                data = response.json()
                self.todos = data.get("data", [])
                
                # Sort todos: first by completion status, then by creation date (newest first)
                priority_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
                
                # Make sure each todo has a creation date, using a default if not present
                for todo in self.todos:
                    if not todo.get("createdAt"):
                        todo["createdAt"] = "1970-01-01T00:00:00.000Z"
                    
                    # Map priority to difficulty for consistent representation
                    priority_value = todo.get("priority")
                    difficulty_map_reverse = {0.1: "trivial", 1: "easy", 1.5: "medium", 2: "hard"}
                    todo["difficulty"] = difficulty_map_reverse.get(priority_value, "medium")
                
                # Sort tasks: first by completion, then by creation date (newer first), then by priority
                try:
                    self.todos.sort(
                        key=lambda x: (
                            x.get("completed", False),
                            # Reverse the creation date sorting to get newer tasks at the top
                            -datetime.datetime.fromisoformat(x.get("createdAt", "1970-01-01T00:00:00.000Z").replace("Z", "+00:00")).timestamp(),
                            priority_order.get(x.get("priority", "low"), 4)
                        )
                    )
                except Exception as e:
                    self.terminal_output.append(f"Error sorting todos: {str(e)}")
                    # Fallback to a simpler sort if date parsing fails
                    self.todos.sort(key=lambda x: x.get("completed", False))
                
                self.status_message = f"Found {len(self.todos)} Habitica todos"
            else:
                self.log_api_error(response, "Error fetching todos")
                self.status_message = f"Error fetching todos: {response.status_code}"
                    
        except requests.exceptions.Timeout:
            self.status_message = "Error: Request to Habitica timed out"
            self.terminal_output.append("Connection timeout - check internet connection")
        except requests.exceptions.ConnectionError:
            self.status_message = "Error: Could not connect to Habitica"
            self.terminal_output.append("Connection error - check internet connection")
        except Exception as e:
            self.status_message = f"Error: {str(e)}"
            self.terminal_output.append(f"Habitica API error: {str(e)}")
        
        self.filtered_todos = self.todos.copy()  # Initialize filtered todos list

    def update_task_difficulty(self, task_id, difficulty_str):
        """Update a task's difficulty in Habitica."""
        if not self.use_habitica:
            self.terminal_output.append("Cannot update difficulty: Habitica integration is disabled")
            return False
            
        if not task_id:
            self.terminal_output.append("Cannot update difficulty: No task ID provided")
            return False
        
        # Map difficulty string to Habitica's numeric value
        difficulty_map = {"trivial": 0.1, "easy": 1, "medium": 1.5, "hard": 2}
        
        # Default to medium if invalid difficulty is provided
        difficulty = difficulty_map.get(difficulty_str.lower(), 1.5)
        
        try:
            headers = {
                "x-api-user": self.habitica_user_id,
                "x-api-key": self.habitica_api_token,
                "Content-Type": "application/json"
            }
            
            # In Habitica's API, the difficulty is represented by the 'priority' field
            data = {
                "priority": difficulty
            }
            
            self.terminal_output.append(f"Updating task {task_id} difficulty to {difficulty_str}")
            
            # PUT request to update the task
            response = requests.put(
                f"https://habitica.com/api/v3/tasks/{task_id}",
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                self.terminal_output.append(f"Successfully updated task difficulty to {difficulty_str}")
                
                # Update the local todo object
                for todo in self.todos:
                    if todo.get("id") == task_id:
                        todo["difficulty"] = difficulty_str
                        todo["priority"] = difficulty  # Update the priority value as well
                        break
                        
                # Update filtered todos too
                for todo in self.filtered_todos:
                    if todo.get("id") == task_id:
                        todo["difficulty"] = difficulty_str
                        todo["priority"] = difficulty
                        break
                        
                return True
            else:
                self.log_api_error(response, "Failed to update task difficulty")
                return False
                    
        except Exception as e:
            self.terminal_output.append(f"Error updating task difficulty: {str(e)}")
            return False
        
    def filter_todos(self):
        """Filter todos based on the search term."""
        if not self.search_term:
            self.filtered_todos = self.todos.copy()
        else:
            self.filtered_todos = []
            search_term_lower = self.search_term.lower()
            
            for todo in self.todos:
                if search_term_lower in todo.get("text", "").lower():
                    self.filtered_todos.append(todo)
        
        # Reset index and scroll position
        self.current_index = 0
        self.scroll_offset = 0
        
        # Update status message - more informative when no todos are found
        if len(self.filtered_todos) == 0:
            self.status_message = "No matching todos found"
        else:
            self.status_message = f"Found {len(self.filtered_todos)} matching todos"
    
    def create_new_objective(self):
        """Create a new Habitica objective."""
        # Reset form fields
        self.init_form_fields()
        self.form_field_index = 0
        self.form_task_list = []
        self.form_subtask_list = []
        self.task_subtask_pairs = []
        self.current_task_buffer = ""
        self.current_subtask_buffer = ""
        self.adding_tasks = False
        self.adding_subtasks = False
        self.in_task_subtask_flow = False
        self.current_pair_index = -1
        self.view_mode = "create"

    def run_objective(self, todo):
        """Run the selected objective with timer."""
        if not todo:
            self.status_message = "No objective selected."
            return
        
        objective = todo.get("text", "Unknown Objective")
        todo_id = todo.get("id", "")
        
        # Store the original difficulty for comparison later
        original_difficulty = todo.get("difficulty", "medium")
        if isinstance(original_difficulty, (int, float)):
            # Convert numeric difficulty back to string
            difficulty_map_reverse = {0.1: "trivial", 1: "easy", 1.5: "medium", 2: "hard"}
            original_difficulty = difficulty_map_reverse.get(original_difficulty, "medium")
        
        # Log the attempt to run this task
        self.terminal_output.append(f"Starting objective: {objective} (ID: {todo_id})")
        
        # Initialize timer values
        self.timer_running = True
        self.timer_start = time.time()
        self.timer_elapsed = 0
        self.timer_paused = False
        self.timer_pause_start = 0
        self.timer_break_time = 0
        
        # Extract planned_min from notes if available
        notes = todo.get("notes", "")
        planned_min = "25"  # Default value
        
        # First try: Use regex to find the planned duration section
        planned_min_match = re.search(r"## Planned Duration\s+(\d+)\s+minutes", notes, re.DOTALL)
        if planned_min_match:
            planned_min = planned_min_match.group(1)
            self.terminal_output.append(f"Found planned duration in notes: {planned_min} minutes")
        
        # If that fails, try a more general approach
        if planned_min == "25" and notes:
            # Look for any line with numbers followed by "minutes"
            min_match = re.search(r"(\d+)\s+minutes", notes)
            if min_match:
                planned_min = min_match.group(1)
                self.terminal_output.append(f"Found minutes using fallback method: {planned_min}")
        
        # Debug logging
        self.terminal_output.append(f"Using planned minutes: {planned_min}")
        
        # Set up basic objective info from todo
        self.form_fields = [
            {"name": "objective", "label": "Objective", "value": objective, "required": True},
            {"name": "difficulty", "label": "Difficulty", "value": todo.get("difficulty", "medium"), "required": True},
            {"name": "priority", "label": "Priority", "value": todo.get("priority", "medium"), "required": True},
            {"name": "due_date", "label": "Due Date", "value": todo.get("date", ""), "required": False},
            {"name": "reward", "label": "Reward", "value": "", "required": False},
            {"name": "worth", "label": "Worth", "value": "", "required": False},
            {"name": "planned_min", "label": "Planned duration (min)", "value": planned_min, "required": True}
        ]
        
        # Extract tasks and subtasks from notes if available
        self.form_task_list = []
        self.form_subtask_list = []
        self.task_subtask_pairs = []
        
        if notes:
            # Try to parse tasks and subtasks from notes
            task_match = re.search(r"## Tasks\s+(.+?)(?=##|$)", notes, re.DOTALL)
            if task_match:
                tasks_section = task_match.group(1).strip()
                for line in tasks_section.split("\n"):
                    # Handle numbered list items and clean them up
                    stripped = re.sub(r"^\d+\.\s+", "", line.strip())
                    if stripped:
                        self.form_task_list.append(stripped)
            
            subtask_match = re.search(r"## Subtasks\s+(.+?)(?=##|$)", notes, re.DOTALL)
            if subtask_match:
                subtasks_section = subtask_match.group(1).strip()
                for line in subtasks_section.split("\n"):
                    # Handle both checked and unchecked box formats
                    stripped = re.sub(r"^-\s+\[\s*[xX]?\s*\]\s*", "", line.strip())
                    if stripped:
                        self.form_subtask_list.append(stripped)
            
            # If parsing failed, try a more lenient approach for older tasks
            if not self.form_task_list and not self.form_subtask_list:
                self.terminal_output.append("Using fallback parsing for older task format")
                # Look for any bullet points or numbered lists
                lines = notes.split("\n")
                for line in lines:
                    stripped = line.strip()
                    # Check if it's a bullet point or number
                    if re.match(r"^(\d+\.|\-|\*)\s+", stripped):
                        item = re.sub(r"^(\d+\.|\-|\*)\s+", "", stripped)
                        item = re.sub(r"\[\s*[xX]?\s*\]", "", item).strip()
                        if item:
                            self.form_task_list.append(item)
        
        # Create task-subtask pairs from the parsed lists
        for i, task in enumerate(self.form_task_list):
            pair = {"task": task}
            # Only add a subtask if one exists for this index
            if i < len(self.form_subtask_list):
                pair["subtask"] = self.form_subtask_list[i]
            self.task_subtask_pairs.append(pair)
        
        # Switch to run mode
        self.view_mode = "run"
        self.terminal_output.append(f"Timer started for: {objective} with planned duration: {planned_min} minutes")

    def save_objective_to_obsidian(self, objective_data):
        """Save the objective to Obsidian as a markdown file, even if Habitica fails."""
        try:
            # Get the objective title and create a safe filename
            objective = objective_data.get("objective", "Untitled Objective")
            # Improved filename sanitization to handle special characters better
            safe_obj = re.sub(r'[^\w\s-]', '', objective).replace(' ', '_')
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            file_name = f"{today}_{safe_obj}.md"
            file_path = os.path.join(self.save_dir, file_name)
            
            # Ensure the save directory exists
            os.makedirs(self.save_dir, exist_ok=True)
            
            # Create markdown content
            with open(file_path, 'w') as f:
                f.write(f"# {objective}\n\n")
                
                # Write difficulty and priority
                difficulty = objective_data.get("difficulty", "medium")
                priority = objective_data.get("priority", "medium")
                f.write(f"**Difficulty:** {difficulty.upper()}\n\n")
                f.write(f"**Priority:** {priority.upper()}\n")
                
                # Write due date if provided
                due_date = objective_data.get("due_date", "")
                if due_date:
                    f.write(f"**Due Date:** {due_date}\n\n")
                
                # Write tasks
                if self.form_task_list:
                    f.write("## Tasks\n\n")
                    for i, task in enumerate(self.form_task_list):
                        f.write(f"{i+1}. {task}\n")
                    f.write("\n")
                
                # Write subtasks
                if self.form_subtask_list:
                    f.write("## Subtasks\n\n")
                    for subtask in self.form_subtask_list:
                        f.write(f"- [ ] {subtask}\n")
                    f.write("\n")
                
                # Write reward if provided
                reward = objective_data.get("reward", "")
                if reward:
                    f.write("## Reward\n\n")
                    f.write(f"{reward}\n\n")
                
                # Write worth if provided
                worth = objective_data.get("worth", "")
                if worth:
                    f.write("## Worth\n\n")
                    f.write(f"{worth}\n\n")
                
                # Write planned duration
                planned_min = objective_data.get("planned_min", "25")
                f.write("## Planned Duration\n\n")
                f.write(f"{planned_min} minutes\n\n")
                
                # Write creation timestamp
                f.write("\n## Created\n\n")
                f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                # Write Habitica integration status
                f.write("\n## Habitica Integration\n\n")
                f.write("Status: Not synchronized with Habitica\n")
            
            self.terminal_output.append(f"Objective saved to Obsidian: {file_path}")
            return file_path
        
        except Exception as e:
            error_msg = f"Error saving to Obsidian: {str(e)}"
            self.terminal_output.append(error_msg)
            self.status_message = error_msg
            
            # Try to save to a temp file as fallback
            try:
                temp_dir = tempfile.gettempdir()
                backup_path = os.path.join(temp_dir, file_name)
                with open(backup_path, 'w') as f:
                    f.write(f"# {objective} (BACKUP - SAVE FAILED)\n\n")
                    # Write minimal content
                    f.write(f"Created on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.terminal_output.append(f"Backup file saved to: {backup_path}")
                return backup_path
            except Exception as backup_error:
                self.terminal_output.append(f"Failed to save backup file: {str(backup_error)}")
                return None

    def submit_new_objective(self):
        """Submit the new objective to Habitica and save to Obsidian with improved error handling."""
        # Validate required fields
        for field in self.form_fields:
            if field.get("required", False) and not field.get("value", "").strip():
                self.status_message = f"Error: {field['label']} is required"
                return
        
        # Get objective data
        objective_data = {}
        for field in self.form_fields:
            if field["name"] != "task_subtask":  # Skip the special task_subtask field
                objective_data[field["name"]] = field["value"]
        
        # Extract task and subtask lists from pairs if using the interleaved flow
        if self.task_subtask_pairs:
            self.form_task_list = []
            self.form_subtask_list = []
            for pair in self.task_subtask_pairs:
                if pair.get("task"):
                    self.form_task_list.append(pair["task"])
                    if pair.get("subtask"):
                        self.form_subtask_list.append(pair["subtask"])
        
        # Add tasks and subtasks
        objective_data["tasks"] = self.form_task_list
        objective_data["subtasks"] = self.form_subtask_list
        
        # First, always save to Obsidian
        obsidian_file_path = self.save_objective_to_obsidian(objective_data)
        if obsidian_file_path:
            self.status_message = f"Objective saved to Obsidian: {os.path.basename(obsidian_file_path)}"
        
        # Habitica integration - try to create the task but don't fail completely if it doesn't work
        habitica_success = False
        if self.use_habitica:
            try:
                # Validate objective text exists
                if not objective_data.get("objective", "").strip():
                    self.terminal_output.append("Error: Objective text is required for Habitica")
                    self.status_message = "Saved to Obsidian only - missing objective text for Habitica"
                    self.view_mode = "list"
                    return
                    
                headers = {
                    "x-api-user": self.habitica_user_id,
                    "x-api-key": self.habitica_api_token,
                    "Content-Type": "application/json"
                }
                
                # Prepare notes field with tasks and subtasks
                notes = ""
                if self.form_task_list:
                    notes += "## Tasks\n\n"
                    for i, task in enumerate(self.form_task_list):
                        notes += f"{i+1}. {task}\n"
                    notes += "\n"
                
                if self.form_subtask_list:
                    notes += "## Subtasks\n\n"
                    for subtask in self.form_subtask_list:
                        notes += f"- [ ] {subtask}\n"
                    notes += "\n"
                
                # Add worth and reward to notes if provided
                if objective_data.get("worth", "").strip():
                    notes += f"## Worth\n\n{objective_data['worth']}\n\n"
                
                if objective_data.get("reward", "").strip():
                    notes += f"## Reward\n\n{objective_data['reward']}\n\n"
                
                # ADD THIS SECTION TO INCLUDE PLANNED DURATION IN NOTES
                planned_min = objective_data.get("planned_min", "25")
                notes += f"## Planned Duration\n\n{planned_min} minutes\n\n"
                
                # Fix validation issue: Ensure priority and difficulty are valid values
                # Convert string values to the numeric values expected by the API
                priority_map = {"low": 0.1, "medium": 1, "high": 1.5, "urgent": 2}
                difficulty_map = {"trivial": 0.1, "easy": 1, "medium": 1.5, "hard": 2}
                
                priority_str = objective_data.get("priority", "medium").lower()
                difficulty_str = objective_data.get("difficulty", "medium").lower()
                
                # Use numeric values for priority and difficulty
                priority = priority_map.get(priority_str, 1)
                difficulty = difficulty_map.get(difficulty_str, 1.5)
                
                if priority_str not in priority_map:
                    priority_str = "medium"
                    self.terminal_output.append(f"Warning: Invalid priority value, using 'medium' instead")
                
                if difficulty_str not in difficulty_map:
                    difficulty_str = "medium"
                    self.terminal_output.append(f"Warning: Invalid difficulty value, using 'medium' instead")
                
                # Start with a minimal payload with required fields
                data = {
                    "text": objective_data["objective"],
                    "type": "todo",
                    "priority": difficulty  # Use the difficulty value for the priority field in Habitica API
                }

                # Only add notes if not empty
                if notes.strip():
                    data["notes"] = notes
                    
                # Add due date only if provided and valid
                if objective_data.get("due_date", ""):
                    try:
                        due_date = datetime.datetime.strptime(objective_data["due_date"], "%Y-%m-%d")
                        # Use the correct date format expected by Habitica
                        data["date"] = due_date.strftime("%Y-%m-%dT12:00:00.000Z")
                    except ValueError:
                        # Invalid date format, log the error but continue
                        self.terminal_output.append(f"Warning: Invalid due date format - {objective_data['due_date']}")
                
                # Log the data being sent for debugging
                self.terminal_output.append(f"Sending request to Habitica API...")
                self.terminal_output.append(f"Request data: {json.dumps(data)[:100]}...")
                
                # Use a timeout to prevent hanging
                response = requests.post(
                    "https://habitica.com/api/v3/tasks/user", 
                    headers=headers, 
                    json=data,
                    timeout=10  # 10 second timeout
                )
                
                # Log the complete response for debugging
                self.terminal_output.append(f"Habitica API response: Status {response.status_code}")
                self.terminal_output.append(f"Full response: {response.text[:200]}...")
                
                if response.status_code == 201:
                    response_data = response.json()
                    new_todo = response_data.get("data", {})
                    todo_id = new_todo.get("id", "")
                    
                    self.status_message = f"Objective created in Habitica (ID: {todo_id}) and saved to Obsidian"
                    self.terminal_output.append(f"Objective created in Habitica: {objective_data['objective']}")
                    
                    # Update the Obsidian file with the Habitica ID if possible
                    if obsidian_file_path and os.path.exists(obsidian_file_path):
                        try:
                            with open(obsidian_file_path, 'a') as f:
                                f.write("\n## Habitica Integration\n\n")
                                f.write(f"Status: Synchronized\n")
                                f.write(f"Task ID: {todo_id}\n")
                        except Exception as file_update_error:
                            self.terminal_output.append(f"Note: Could not update the Obsidian file with Habitica ID: {str(file_update_error)}")
                    
                    habitica_success = True
                else:
                    self.log_api_error(response, "Habitica task creation failed")
                    self.status_message = f"Saved to Obsidian but Habitica integration failed"
                    
            except requests.exceptions.Timeout:
                self.terminal_output.append("Error: Request to Habitica timed out.")
                self.status_message = "Saved to Obsidian but Habitica API timed out"
            except requests.exceptions.ConnectionError:
                self.terminal_output.append("Error: Could not connect to Habitica. Check your internet connection.")
                self.status_message = "Saved to Obsidian but couldn't connect to Habitica"
            except Exception as e:
                self.terminal_output.append(f"Habitica integration exception: {str(e)}")
                self.status_message = "Saved to Obsidian but Habitica integration failed"
        
        # Refresh todos if Habitica integration worked, otherwise just return to list view
        if habitica_success:
            # Refresh todos and return to list view
            self.fetch_habitica_todos()
        
        # Always return to list view after saving, regardless of Habitica success
        self.view_mode = "list"


    def complete_objective(self, todo_id):
        """Mark a Habitica todo as complete with proper API call."""
        if not todo_id:
            self.status_message = "Error: No task ID provided"
            self.terminal_output.append("Failed to complete objective: No task ID")
            return False
            
        try:
            headers = {
                "x-api-user": self.habitica_user_id,
                "x-api-key": self.habitica_api_token,
                "Content-Type": "application/json"
            }
            
            # First try to check if the task exists
            check_response = requests.get(
                f"https://habitica.com/api/v3/tasks/{todo_id}", 
                headers=headers,
                timeout=10
            )
            
            if check_response.status_code != 200:
                # Task might not exist or might be inaccessible
                self.status_message = f"Error: Task with ID {todo_id} not found"
                self.terminal_output.append(f"Task with ID {todo_id} not found. It may have been deleted or completed already.")
                return False
                
            # Now try to complete it using the correct API endpoint
            response = requests.post(
                f"https://habitica.com/api/v3/tasks/{todo_id}/score/up", 
                headers=headers,
                timeout=10
            )
            
            self.terminal_output.append(f"Completion response status: {response.status_code}")
            if response.status_code == 200:
                self.status_message = "Objective marked as complete in Habitica"
                self.terminal_output.append("Objective completed successfully!")
                return True
            else:
                self.log_api_error(response, "Task completion failed")
                self.status_message = "Error completing objective"
                return False
                    
        except requests.exceptions.RequestException as e:
            self.status_message = f"Network error: {str(e)}"
            self.terminal_output.append(f"Network error when completing objective: {str(e)}")
            return False
        except Exception as e:
            self.status_message = f"Error: {str(e)}"
            self.terminal_output.append(f"Error when completing objective: {str(e)}")
            return False

    def complete_current_objective(self):
        """Complete the current objective and generate report."""
        # Stop the timer
        self.timer_running = False
        end_time = time.time()
        
        # Calculate time values
        total_sec = int(end_time - self.timer_start)
        break_sec = int(self.timer_break_time)
        active_sec = total_sec - break_sec
        active_min = active_sec // 60
        
        # Get the current todo
        if self.current_index < len(self.filtered_todos):
            todo = self.filtered_todos[self.current_index]
            todo_id = todo.get("id", "")
            objective = todo.get("text", "Unknown Objective")
            
            # Get planned minutes to calculate productivity
            planned_min = next((f["value"] for f in self.form_fields if f["name"] == "planned_min"), "25")
            try:
                planned_min_int = int(float(planned_min))
            except (ValueError, TypeError):
                planned_min_int = 25
                self.terminal_output.append(f"Warning: Using default planned minutes (25) due to conversion error")
            
            # Calculate productivity
            planned_sec = planned_min_int * 60
            prod_str = "N/A"
            try:
                if active_sec > 0 and planned_sec > 0:
                    prod = min(200, (planned_sec / active_sec) * 100)  # Cap at 200%
                    prod_str = f"{prod:.1f}%"
                else:
                    prod_str = "0.0%"
                    self.terminal_output.append("Warning: Active time was zero, productivity calculation defaulted to 0%")
            except Exception as e:
                self.terminal_output.append(f"Productivity calculation error: {str(e)}")
                prod_str = "Error%"

            # Generate markdown file
            try:
                report_path = self.generate_markdown_report(objective, total_sec, active_sec, break_sec, todo_id)
                self.terminal_output.append(f"Report generated at: {report_path}")
            except Exception as e:
                self.terminal_output.append(f"Error generating report: {str(e)}")
                report_path = None
            
            # Mark as complete in Habitica
            habitica_success = False
            if todo_id:
                try:
                    habitica_success = self.complete_objective(todo_id)
                except Exception as e:
                    self.terminal_output.append(f"Error completing objective in Habitica: {str(e)}")
                
                if not habitica_success and report_path:
                    self.terminal_output.append(f"Note: Objective report was saved locally at {report_path}")
            
            # IMPORTANT: Set view mode to "list" BEFORE displaying productivity
            # This ensures we don't get stuck in a loop
            original_view_mode = self.view_mode
            self.view_mode = "list"
            
            # Display productivity and wait for acknowledgment
            display_result = self.display_productivity(prod_str, active_min)
            
            # Only if display was successful and acknowledged
            if display_result:
                # Refresh todos only if we successfully showed the productivity screen
                try:
                    self.fetch_habitica_todos()
                except Exception as e:
                    self.terminal_output.append(f"Error refreshing todos: {str(e)}")
            else:
                # If display failed, make sure we're still in list view
                self.view_mode = "list"
                self.terminal_output.append("Warning: Could not display productivity screen")
                
        else:
            self.status_message = "Error: No objective selected"
            self.terminal_output.append("Error: Attempted to complete objective but none was selected")

    def format_time(self, seconds):
        """Format seconds as HH:MM:SS."""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def generate_markdown_report(self, objective, total_sec, active_sec, break_sec, habitica_task_id=""):
        """Generate a markdown report for the objective."""
        # Get form field values
        difficulty = next((f["value"] for f in self.form_fields if f["name"] == "difficulty"), "medium")
        priority = next((f["value"] for f in self.form_fields if f["name"] == "priority"), "medium")
        due_date = next((f["value"] for f in self.form_fields if f["name"] == "due_date"), "")
        reward = next((f["value"] for f in self.form_fields if f["name"] == "reward"), "")
        worth = next((f["value"] for f in self.form_fields if f["name"] == "worth"), "")
        planned_min = next((f["value"] for f in self.form_fields if f["name"] == "planned_min"), "25")
        
        # Ensure values are strings before calling upper()
        difficulty = str(difficulty) if difficulty is not None else "medium"
        priority = str(priority) if priority is not None else "medium"
        
        try:
            # Try to convert planned_min to integer
            planned_min_int = int(float(planned_min))
        except (ValueError, TypeError):
            # Default to 25 if conversion fails
            planned_min_int = 25
            planned_min = "25"
        
        # Calculate productivity
        planned_sec = planned_min_int * 60
        prod_str = "N/A"
        try:
            if active_sec > 0 and planned_sec > 0:
                prod = min(200, (planned_sec / active_sec) * 100)  # Cap at 200%
                prod_str = f"{prod:.1f}%"
        except (ZeroDivisionError, TypeError):
            prod_str = "N/A"
        
        # Generate markdown content
        # Improved filename sanitization
        safe_obj = re.sub(r'[^\w\s-]', '', objective).replace(' ', '_')
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        file_name = f"{today}_{safe_obj}.md"
        file_path = os.path.join(self.save_dir, file_name)
        
        # Ensure the save directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        
        try:
            with open(file_path, 'w') as f:
                f.write(f"# {objective}\n\n")
                # Use str.upper() safely after ensuring it's a string
                f.write(f"**Difficulty:** {difficulty.upper()}\n\n")
                f.write(f"**Priority:** {priority.upper()}\n")
                if due_date:
                    f.write(f"**Due Date:** {due_date}\n\n")
                
                if self.form_task_list:
                    f.write("## Tasks\n\n")
                    for i, task in enumerate(self.form_task_list):
                        f.write(f"{i+1}. {task}\n")
                    f.write("\n")
                
                if self.form_subtask_list:
                    f.write("## Subtasks\n\n")
                    for subtask in self.form_subtask_list:
                        f.write(f"- [ ] {subtask}\n")
                    f.write("\n")
                
                if reward:
                    f.write("## Reward\n\n")
                    f.write(f"{reward}\n\n")
                
                if worth:
                    f.write("## Worth\n\n")
                    f.write(f"{worth}\n\n")
                
                f.write("## Planned Duration\n\n")
                f.write(f"{planned_min} minutes\n\n")
                
                f.write("## Actual Duration\n\n")
                f.write(f"{active_sec // 60} minutes (Active work time)\n\n")
                
                f.write("## Break Time\n\n")
                f.write(f"{break_sec // 60} minutes\n\n")
                
                f.write("## Productivity\n\n")
                f.write(f"{prod_str}\n\n")
                
                if habitica_task_id:
                    f.write("## Habitica Task\n\n")
                    f.write(f"Task ID: {habitica_task_id}\n")
                    
                # Add completion timestamp
                f.write("\n## Completed\n\n")
                f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
            self.terminal_output.append(f"Objective saved to: {file_path}")
            self.status_message = f"Report generated: {file_name}"
            
        except Exception as e:
            error_msg = f"Error saving report: {str(e)}"
            self.terminal_output.append(error_msg)
            self.status_message = error_msg
            
            # Try to save to a temp file as fallback
            try:
                temp_dir = tempfile.gettempdir()
                backup_path = os.path.join(temp_dir, file_name)
                with open(backup_path, 'w') as f:
                    f.write(f"# {objective} (BACKUP - SAVE FAILED)\n\n")
                    # Write minimal content
                    f.write(f"Completed on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.terminal_output.append(f"Backup report saved to: {backup_path}")
                return backup_path
            except Exception as backup_error:
                self.terminal_output.append(f"Failed to save backup report: {str(backup_error)}")
        
        return file_path

    def toggle_timer(self):
        """Pause or resume the timer."""
        if self.timer_running:
            current_time = time.time()
            if self.timer_paused:
                # Resume timer - calculate break duration and add to total
                pause_duration = current_time - self.timer_pause_start
                self.timer_break_time += pause_duration
                self.timer_paused = False
                self.terminal_output.append(f"Timer resumed. Break: {self.format_time(int(pause_duration))}, Total break: {self.format_time(int(self.timer_break_time))}")
            else:
                # Pause timer
                self.timer_pause_start = current_time
                self.timer_paused = True
                self.terminal_output.append("Timer paused. Press TAB to resume.")

    def draw_screen(self):
        """Draw the UI screen based on the current view mode."""
        # Instead of clear(), use erase() which is less disruptive visually
        self.stdscr.erase()
        self.height, self.width = self.stdscr.getmaxyx()
        
        # Calculate layout dimensions
        header_height = 3
        terminal_height = 6
        content_height = self.height - header_height - terminal_height - 2
        
        # Draw header
        self.stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
        header = "╔══ Habitica Objective Tracker ══╗"
        self.stdscr.addstr(0, (self.width - len(header)) // 2, header)
        self.stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)
        
        # Draw search bar (in list mode only)
        if self.view_mode == "list":
            if self.is_searching:
                self.stdscr.attron(curses.A_REVERSE)
                search_prompt = "Search: " + self.search_buffer + "_"
            else:
                search_prompt = "Search: " + self.search_term
            self.stdscr.addstr(1, 2, search_prompt[:self.width-4])
            if self.is_searching:
                self.stdscr.attroff(curses.A_REVERSE)
        
        # Draw view mode indicator
        mode_text = f"Mode: {self.view_mode.capitalize()}"
        self.stdscr.addstr(1, self.width - len(mode_text) - 2, mode_text)
        
        # Draw status message
        self.stdscr.attron(curses.color_pair(5))
        self.stdscr.addstr(2, 2, self.status_message[:self.width-4])
        self.stdscr.attroff(curses.color_pair(5))
        
        # Draw separator
        self.stdscr.addstr(header_height - 1, 0, "─" * self.width)
        
        # Draw content based on view mode
        if self.view_mode == "list":
            self.draw_list_view(header_height, content_height)
        elif self.view_mode == "create":
            self.draw_create_view(header_height, content_height)
        elif self.view_mode == "run":
            self.draw_run_view(header_height, content_height)
        
        # Draw separator before terminal
        term_start = self.height - terminal_height - 1
        self.stdscr.addstr(term_start, 0, "─" * self.width)
        
        # Draw terminal header
        self.stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
        self.stdscr.addstr(term_start + 1, 2, "Terminal Output")
        self.stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)
        
        # Draw terminal output
        for i in range(terminal_height - 2):
            line_idx = len(self.terminal_output) - terminal_height + 2 + i
            if line_idx >= 0 and line_idx < len(self.terminal_output):
                line = self.terminal_output[line_idx]
                if len(line) > self.width - 4:
                    line = line[:self.width - 7] + "..."
                self.stdscr.addstr(term_start + 2 + i, 2, line)
        
        # Draw help text based on view mode
        if self.view_mode == "list":
            help_text = "↑/↓: Navigate | Enter: Run | N: New | R: Refresh | Q: Quit | /: Search"
        elif self.view_mode == "create":
            help_text = "Enter: Submit | ESC: Cancel | TAB: Next Field | ←/→: Navigate fields"
        elif self.view_mode == "run":
            help_text = "Enter: Complete | ESC: Cancel | TAB: Pause/Resume Timer"
        
        self.stdscr.attron(curses.color_pair(1))
        self.stdscr.addstr(self.height - 1, 0, help_text[:self.width-1])
        self.stdscr.attroff(curses.color_pair(1))
        
        self.stdscr.refresh()


    def display_productivity(self, productivity, active_min):
        """Display productivity percentage in terminal after completing an objective."""
        # Save current timeout setting and switch to blocking mode
        old_timeout = self.stdscr.timeout(-1)  # -1 sets blocking mode
        
        # Clear any pending input to avoid skipping the display
        curses.flushinp()
        
        # Clear the current screen
        self.stdscr.clear()
        
        # Get screen dimensions
        height, width = self.stdscr.getmaxyx()
        
        # Calculate center position
        center_y = height // 2
        
        # Draw a box to highlight the productivity
        box_width = 50
        box_start_x = (width - box_width) // 2
        
        # Draw top border
        self.stdscr.addstr(center_y - 3, box_start_x, "╭" + "─" * (box_width - 2) + "╮")
        
        # Draw side borders
        for i in range(5):
            self.stdscr.addstr(center_y - 2 + i, box_start_x, "│")
            self.stdscr.addstr(center_y - 2 + i, box_start_x + box_width - 1, "│")
        
        # Draw bottom border
        self.stdscr.addstr(center_y + 3, box_start_x, "╰" + "─" * (box_width - 2) + "╯")
        
        # Display objective completion message
        message = "Objective Completed!"
        self.stdscr.addstr(center_y - 2, (width - len(message)) // 2, message, curses.A_BOLD)
        
        # Display productivity percentage
        prod_title = "PRODUCTIVITY"
        self.stdscr.addstr(center_y, (width - len(prod_title)) // 2, prod_title)
        
        # Use different formatting based on productivity value
        try:
            prod_value = float(productivity.strip('%'))
            if prod_value >= 100:
                self.stdscr.attron(curses.color_pair(1) | curses.A_BOLD)  # Green for good productivity
            elif prod_value >= 75:
                self.stdscr.attron(curses.color_pair(3))  # Yellow for moderate productivity
            else:
                self.stdscr.attron(curses.color_pair(5))  # Red for low productivity
        except (ValueError, AttributeError):
            self.stdscr.attron(curses.color_pair(1))  # Default to green
        
        self.stdscr.addstr(center_y + 1, (width - len(productivity)) // 2, productivity)
        self.stdscr.attroff(curses.A_BOLD | curses.color_pair(1) | curses.color_pair(3) | curses.color_pair(5))
        
        # Show active time that was used for calculation
        active_time_msg = f"Active time: {active_min} minutes"
        self.stdscr.addstr(center_y + 2, (width - len(active_time_msg)) // 2, active_time_msg)
        
        # Display instruction to continue - be very explicit
        continue_msg = "Press ENTER to continue..."
        self.stdscr.addstr(center_y + 5, (width - len(continue_msg)) // 2, continue_msg)
        
        # Refresh the screen
        self.stdscr.refresh()
        
        # Wait for specific input - only ENTER will continue
        while True:
            key = self.stdscr.getch()
            if key == 10:  # ENTER key
                break
        
        # Restore original timeout
        self.stdscr.timeout(old_timeout)
        
        # Return True to indicate completion
        return True

    def draw_list_view(self, header_height, content_height):
        """Draw the todo list view."""
        # Show list count
        if len(self.filtered_todos) > 0:
            status = f"{len(self.filtered_todos)} todos | {self.current_index + 1}/{len(self.filtered_todos)}"
        else:
            status = "No todos found"
        self.stdscr.addstr(header_height, 2, status)
        
        # Draw todo list
        list_area_height = content_height - 1
        visible_items = min(list_area_height, len(self.filtered_todos))
        
        for i in range(visible_items):
            todo_idx = i + self.scroll_offset
            if todo_idx >= len(self.filtered_todos):
                break
            
            todo = self.filtered_todos[todo_idx]
            todo_text = todo.get("text", "Unknown")
            priority = todo.get("priority", "medium")
            due_date = todo.get("date", "")
            
            # Format due date
            date_str = ""
            if due_date:
                try:
                    date_obj = datetime.datetime.fromisoformat(due_date.replace("Z", "+00:00"))
                    date_str = f" (Due: {date_obj.strftime('%Y-%m-%d')})"
                except Exception as e:
                    # Handle date parsing errors gracefully
                    self.terminal_output.append(f"Date parsing error: {str(e)}")
                    date_str = " (Invalid due date)"
            
            display_text = f"{todo_text}{date_str}"
            
            # Truncate if too long
            if len(display_text) > self.width - 4:
                display_text = display_text[:self.width - 7] + "..."
            
            # Highlight selected item
            if todo_idx == self.current_index:
                self.stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
                self.stdscr.addstr(i + header_height + 1, 0, " " * self.width)
                self.stdscr.addstr(i + header_height + 1, 2, display_text)
                self.stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
            else:
                # Color based on priority
                if priority == "high":
                    self.stdscr.attron(curses.color_pair(3))
                elif priority == "medium":
                    self.stdscr.attron(curses.color_pair(4))
                elif priority == "urgent":
                    self.stdscr.attron(curses.color_pair(5))
                else:  # low priority
                    self.stdscr.attron(curses.color_pair(1))
                
                self.stdscr.addstr(i + header_height + 1, 2, display_text)
                
                if priority == "high":
                    self.stdscr.attroff(curses.color_pair(3))
                elif priority == "medium":
                    self.stdscr.attroff(curses.color_pair(4))
                elif priority == "urgent":
                    self.stdscr.attroff(curses.color_pair(5))
                else:  # low priority
                    self.stdscr.attroff(curses.color_pair(1))
    
    def draw_create_view(self, header_height, content_height):
        """Draw the objective creation form with interleaved task-subtask flow."""
        # Display a different, more conversational header
        self.stdscr.addstr(header_height, 2, "Let's Create a New Objective")
        
        y_offset = header_height + 2
        
        # Handle regular form fields
        if self.form_field_index < len(self.form_fields):
            field = self.form_fields[self.form_field_index]
            
            # Special handling for the task-subtask field
            if field["name"] == "task_subtask":
                # Draw the task-subtask section header
                self.stdscr.addstr(y_offset, 2, "Enter tasks and subtasks:")
                y_offset += 2
                
                # Show existing task-subtask pairs
                for i, pair in enumerate(self.task_subtask_pairs):
                    if y_offset >= header_height + content_height - 4:
                        break
                    
                    # Display task
                    task = pair.get("task", "")
                    if task:
                        self.stdscr.addstr(y_offset, 4, f"Task {i+1}: {task}")
                        y_offset += 1
                    
                    # Display subtask
                    subtask = pair.get("subtask", "")
                    if subtask:
                        self.stdscr.addstr(y_offset, 6, f"↳ Subtask: {subtask}")
                        y_offset += 1
                    
                    y_offset += 1
                
                # Show current input field based on the state
                if self.in_task_subtask_flow:
                    if self.adding_tasks:
                        # Task input field
                        self.stdscr.addstr(y_offset, 4, f"Task {len(self.task_subtask_pairs)+1}: ")
                        y_offset += 1
                        self.stdscr.attron(curses.color_pair(7))
                        self.stdscr.addstr(y_offset, 6, f"{self.current_task_buffer}_")
                        self.stdscr.attroff(curses.color_pair(7))
                        y_offset += 2
                    
                    elif self.adding_subtasks:
                        # Subtask input field for the current task
                        task_idx = len(self.task_subtask_pairs)
                        if task_idx > 0:
                            current_task = self.task_subtask_pairs[task_idx-1].get("task", "")
                            self.stdscr.addstr(y_offset, 4, f"Task {task_idx}: {current_task}")
                            y_offset += 1
                            
                            self.stdscr.addstr(y_offset, 6, "↳ Subtask: ")
                            y_offset += 1
                            self.stdscr.attron(curses.color_pair(7))
                            self.stdscr.addstr(y_offset, 8, f"{self.current_subtask_buffer}_")
                            self.stdscr.attroff(curses.color_pair(7))
                            y_offset += 2
                    
                # Show instruction for starting task-subtask flow
                else:
                    self.stdscr.attron(curses.A_REVERSE)
                    self.stdscr.addstr(y_offset, 4, "[Press Enter to add tasks and subtasks, Right arrow to continue]")
                    self.stdscr.attroff(curses.A_REVERSE)
                    y_offset += 2
                
                # Navigation instructions
                if self.in_task_subtask_flow:
                    self.stdscr.addstr(y_offset, 4, "Press Right → to continue, Left ← to edit previous")
                    y_offset += 1
                    
                    if not self.adding_tasks and not self.adding_subtasks:
                        self.stdscr.addstr(y_offset, 4, "Press Enter to start adding another task")
                        y_offset += 1
                
            else:
                # Regular field
                label = field["label"]
                value = field["value"]
                
                # Display the field question
                self.stdscr.addstr(y_offset, 2, label)
                y_offset += 2
                
                # Display the input field
                self.stdscr.attron(curses.color_pair(7))
                self.stdscr.addstr(y_offset, 4, f"{value}_")
                self.stdscr.attroff(curses.color_pair(7))
        
        # If we've completed all fields, show summary
        else:
            self.stdscr.addstr(y_offset, 2, "All fields completed. Press ENTER to submit.")
    
    def draw_run_view(self, header_height, content_height):
        """Draw the objective running view with timer."""
        # Get current objective
        if self.current_index < len(self.filtered_todos):
            todo = self.filtered_todos[self.current_index]
            objective = todo.get("text", "Unknown Objective")
            
            # Display objective title
            self.stdscr.attron(curses.A_BOLD)
            self.stdscr.addstr(header_height, 2, objective[:self.width-4])
            self.stdscr.attroff(curses.A_BOLD)
            
            # Display timer
            if self.timer_running:
                current_time = time.time()
                if self.timer_paused:
                    elapsed = self.timer_pause_start - self.timer_start - self.timer_break_time
                else:
                    elapsed = current_time - self.timer_start - self.timer_break_time
                
                # Format timer display
                timer_str = self.format_time(max(0, int(elapsed)))  # Ensure non-negative
                timer_status = "PAUSED" if self.timer_paused else "RUNNING"
                break_time = self.format_time(int(self.timer_break_time))
                total_time = self.format_time(int(current_time - self.timer_start))
                
                # Calculate progress based on planned duration
                planned_min = next((f["value"] for f in self.form_fields if f["name"] == "planned_min"), "25")
                try:
                    planned_sec = int(float(planned_min)) * 60
                    progress_pct = min(100, (elapsed / planned_sec) * 100) if planned_sec > 0 else 0
                    progress_bar = f"[{'#' * int(progress_pct / 5)}{' ' * (20 - int(progress_pct / 5))}]"
                    progress_text = f"{progress_pct:.1f}% of planned {planned_min} min"
                except (ValueError, ZeroDivisionError, TypeError):
                    progress_bar = "[                    ]"
                    progress_text = "Progress calculation error"
                
                self.stdscr.attron(curses.color_pair(5) | curses.A_BOLD)
                self.stdscr.addstr(header_height + 2, 2, f"Timer: {timer_str} [{timer_status}]")
                self.stdscr.attroff(curses.color_pair(5) | curses.A_BOLD)
                
                self.stdscr.addstr(header_height + 3, 2, f"Break time: {break_time}")
                self.stdscr.addstr(header_height + 4, 2, f"Total session: {total_time}")
                
                # Show progress bar
                self.stdscr.attron(curses.color_pair(4))
                self.stdscr.addstr(header_height + 5, 2, f"Progress: {progress_bar} {progress_text}")
                self.stdscr.attroff(curses.color_pair(4))
            
            # Display tasks and subtasks
            y_offset = header_height + 7
            
            # Show tasks
            if self.form_task_list:
                self.stdscr.attron(curses.A_BOLD)
                self.stdscr.addstr(y_offset, 2, "Tasks:")
                self.stdscr.attroff(curses.A_BOLD)
                y_offset += 1
                
                for i, task in enumerate(self.form_task_list):
                    if y_offset >= header_height + content_height:
                        break
                    task_display = task[:self.width-8]  # Truncate if too long
                    self.stdscr.addstr(y_offset, 4, f"{i+1}. {task_display}")
                    y_offset += 1
                
                y_offset += 1
            
            # Show subtasks
            if self.form_subtask_list and y_offset < header_height + content_height:
                self.stdscr.attron(curses.A_BOLD)
                self.stdscr.addstr(y_offset, 2, "Subtasks:")
                self.stdscr.attroff(curses.A_BOLD)
                y_offset += 1
                
                for subtask in self.form_subtask_list:
                    if y_offset >= header_height + content_height:
                        break
                    subtask_display = subtask[:self.width-10]  # Truncate if too long
                    self.stdscr.addstr(y_offset, 4, f"- [ ] {subtask_display}")
                    y_offset += 1

    def handle_input(self):
        """Handle user input based on the current view mode."""
        try:
            key = self.stdscr.getch()
            
            # Handle search mode (applicable in list view)
            if self.is_searching:
                if key == 27:  # ESC
                    self.is_searching = False
                    self.search_buffer = ""
                elif key == 10:  # Enter
                    self.search_term = self.search_buffer
                    self.search_buffer = ""
                    self.is_searching = False
                    self.filter_todos()
                elif key == curses.KEY_BACKSPACE or key == 127 or key == 8:
                    self.search_buffer = self.search_buffer[:-1]
                    self.search_term = self.search_buffer
                    self.filter_todos()
                elif key < 256 and chr(key).isprintable():
                    self.search_buffer += chr(key)
                    self.search_term = self.search_buffer
                    self.filter_todos()
                
                return True
            
            # Handle input based on view mode
            if self.view_mode == "list":
                return self.handle_list_input(key)
            elif self.view_mode == "create":
                return self.handle_create_input(key)
            elif self.view_mode == "run":
                return self.handle_run_input(key)
            
            return True
            
        except Exception as e:
            self.status_message = f"Error handling input: {str(e)}"
            self.terminal_output.append(f"Input error: {str(e)}")
            return True
    
    def handle_list_input(self, key):
        """Handle input in list view mode."""
        if key == ord('q') or key == ord('Q'):
            return False
        elif key == curses.KEY_UP:
            self.current_index = max(0, self.current_index - 1)
            if self.current_index < self.scroll_offset:
                self.scroll_offset = self.current_index
        elif key == curses.KEY_DOWN:
            self.current_index = min(len(self.filtered_todos) - 1, self.current_index + 1)
            if self.current_index >= self.scroll_offset + (self.height - 11):
                self.scroll_offset = self.current_index - (self.height - 11) + 1
        elif key == ord('/'):
            # Enable search mode
            self.is_searching = True
            self.search_buffer = ""
            self.terminal_output.append("Search mode: type to search, press Enter to confirm, ESC to cancel")
        elif key == 10:  # Enter
            if self.filtered_todos and self.current_index < len(self.filtered_todos):
                self.run_objective(self.filtered_todos[self.current_index])
        elif key == ord('n') or key == ord('N'):
            self.create_new_objective()
        elif key == ord('r') or key == ord('R'):
            self.fetch_habitica_todos()
        
        return True
    

    def handle_create_input(self, key):
        """Handle input in create view mode with interleaved task-subtask flow."""
        if key == 27:  # ESC
            # Return to list view without saving
            self.view_mode = "list"
            self.terminal_output.append("Objective creation cancelled")
        elif key == 9:  # TAB - move to next field
            if not self.in_task_subtask_flow:
                old_index = self.form_field_index
                self.form_field_index = (self.form_field_index + 1) % len(self.form_fields)
                self.terminal_output.append(f"Moved from field '{self.form_fields[old_index]['name']}' to '{self.form_fields[self.form_field_index]['name']}'")
        elif key == curses.KEY_LEFT:  # Left Arrow - go back to previous field or entry
            # Handle going back in task-subtask flow
            if self.form_fields[self.form_field_index]["name"] == "task_subtask" and self.in_task_subtask_flow:
                if self.adding_subtasks:
                    # Go back to task input
                    self.adding_subtasks = False
                    self.adding_tasks = True
                    # Get the last task entered
                    if self.task_subtask_pairs and len(self.task_subtask_pairs) > 0:
                        last_pair = self.task_subtask_pairs.pop()  # Remove the last pair
                        self.current_task_buffer = last_pair.get("task", "")
                        self.terminal_output.append("Returned to editing the previous task")
                elif not self.adding_tasks and not self.adding_subtasks:
                    # Go back to the last subtask
                    if self.task_subtask_pairs and len(self.task_subtask_pairs) > 0:
                        last_pair = self.task_subtask_pairs[-1]
                        if "subtask" in last_pair:
                            self.adding_subtasks = True
                            self.current_subtask_buffer = last_pair.get("subtask", "")
                            self.task_subtask_pairs.pop()  # Remove the last pair
                            self.terminal_output.append("Returned to editing the previous subtask")
                        else:
                            self.adding_tasks = True
                            self.current_task_buffer = last_pair.get("task", "")
                            self.task_subtask_pairs.pop()  # Remove the last pair
                            self.terminal_output.append("Returned to editing the previous task")
            else:
                # For regular fields, move to previous field
                old_index = self.form_field_index
                self.form_field_index = (self.form_field_index - 1) % len(self.form_fields)
                self.terminal_output.append(f"Moved from field '{self.form_fields[old_index]['name']}' to '{self.form_fields[self.form_field_index]['name']}'")
                
        elif key == curses.KEY_RIGHT:  # Right Arrow - continue to next field 
            current_field = self.form_fields[self.form_field_index]
            
            # Handle exiting task-subtask flow and moving to next field
            if current_field["name"] == "task_subtask" and self.in_task_subtask_flow:
                if self.adding_tasks:
                    # Add this task if not empty and transition to subtask entry
                    if self.current_task_buffer.strip():
                        self.task_subtask_pairs.append({"task": self.current_task_buffer.strip()})
                        self.terminal_output.append(f"Added task: {self.current_task_buffer.strip()}")
                    self.current_task_buffer = ""
                    self.adding_tasks = False
                    self.adding_subtasks = True
                    
                elif self.adding_subtasks:
                    # Add this subtask if not empty
                    if len(self.task_subtask_pairs) > 0:
                        last_pair = self.task_subtask_pairs[-1]
                        if self.current_subtask_buffer.strip():
                            last_pair["subtask"] = self.current_subtask_buffer.strip()
                            self.terminal_output.append(f"Added subtask: {self.current_subtask_buffer.strip()}")
                    
                    # Reset and prepare for next task
                    self.current_subtask_buffer = ""
                    self.adding_subtasks = False
                    self.adding_tasks = True
                    
                elif not self.adding_tasks and not self.adding_subtasks:
                    # Exit task-subtask flow and move to next field
                    self.in_task_subtask_flow = False
                    old_index = self.form_field_index
                    self.form_field_index = (self.form_field_index + 1) % len(self.form_fields)
                    self.terminal_output.append(f"Moved from task-subtask section to field '{self.form_fields[self.form_field_index]['name']}'")
            else:
                # For regular fields, move to next field
                old_index = self.form_field_index
                self.form_field_index = (self.form_field_index + 1) % len(self.form_fields)
                self.terminal_output.append(f"Moved from field '{self.form_fields[old_index]['name']}' to '{self.form_fields[self.form_field_index]['name']}'")
                
        elif key == 10:  # Enter
            current_field = self.form_fields[self.form_field_index]
            
            # Handle special case for task-subtask field
            if current_field["name"] == "task_subtask":
                if not self.in_task_subtask_flow:
                    # Start the task-subtask flow
                    self.in_task_subtask_flow = True
                    self.adding_tasks = True
                    self.current_task_buffer = ""
                    self.terminal_output.append("Started adding tasks and subtasks")
                    
                elif self.in_task_subtask_flow and self.adding_tasks:
                    # Add this task if not empty and transition to subtask entry
                    if self.current_task_buffer.strip():
                        self.task_subtask_pairs.append({"task": self.current_task_buffer.strip()})
                        self.terminal_output.append(f"Added task: {self.current_task_buffer.strip()}")
                        self.current_task_buffer = ""
                        self.adding_tasks = False
                        self.adding_subtasks = True
                    # If empty, exit task entry mode
                    else:
                        self.adding_tasks = False
                        self.in_task_subtask_flow = False
                        old_index = self.form_field_index
                        self.form_field_index = (self.form_field_index + 1) % len(self.form_fields)
                        self.terminal_output.append(f"Finished adding tasks and subtasks. Moved to field '{self.form_fields[self.form_field_index]['name']}'")
                        
                elif self.in_task_subtask_flow and self.adding_subtasks:
                    # Add this subtask if not empty
                    if len(self.task_subtask_pairs) > 0:
                        last_pair = self.task_subtask_pairs[-1]
                        if self.current_subtask_buffer.strip():
                            last_pair["subtask"] = self.current_subtask_buffer.strip()
                            self.terminal_output.append(f"Added subtask: {self.current_subtask_buffer.strip()}")
                    
                    # Reset and prepare for next task
                    self.current_subtask_buffer = ""
                    self.adding_subtasks = False
                    self.adding_tasks = True
                    
                elif not self.adding_tasks and not self.adding_subtasks:
                    # Start adding a new task
                    self.adding_tasks = True
                    self.current_task_buffer = ""
                    self.terminal_output.append("Started adding a new task")
            
            elif self.form_field_index == len(self.form_fields) - 1:
                # Submit the form if on the last field
                self.terminal_output.append("Submitting objective to Habitica...")
                self.submit_new_objective()
            else:
                # Move to next field
                old_index = self.form_field_index
                self.form_field_index = (self.form_field_index + 1) % len(self.form_fields)
                self.terminal_output.append(f"Moved from field '{self.form_fields[old_index]['name']}' to '{self.form_fields[self.form_field_index]['name']}'")
        
        elif key == 32:  # Space - now ONLY adds a space character
            current_field = self.form_fields[self.form_field_index]
            
            if current_field["name"] == "task_subtask":
                if self.adding_tasks:
                    self.current_task_buffer += " "
                elif self.adding_subtasks:
                    self.current_subtask_buffer += " "
            else:
                # For regular fields, add a space
                current_field["value"] += " "
        
        elif key == curses.KEY_BACKSPACE or key == 127 or key == 8:
            # Handle backspace
            current_field = self.form_fields[self.form_field_index]
            
            if current_field["name"] == "task_subtask":
                if self.adding_tasks and self.current_task_buffer:
                    self.current_task_buffer = self.current_task_buffer[:-1]
                elif self.adding_subtasks and self.current_subtask_buffer:
                    self.current_subtask_buffer = self.current_subtask_buffer[:-1]
            else:
                if current_field["value"]:
                    current_field["value"] = current_field["value"][:-1]
        
        elif key < 256 and chr(key).isprintable():
            # Handle printable characters
            current_field = self.form_fields[self.form_field_index]
            
            if current_field["name"] == "task_subtask":
                if self.adding_tasks:
                    self.current_task_buffer += chr(key)
                elif self.adding_subtasks:
                    self.current_subtask_buffer += chr(key)
            else:
                current_field["value"] += chr(key)
        
        return True


    def handle_run_input(self, key):
        """Handle input in run view mode."""
        if key == 27:  # ESC
            # Return to list view without completing
            
            # Check if difficulty was changed before exiting
            if self.current_index < len(self.filtered_todos):
                todo = self.filtered_todos[self.current_index]
                todo_id = todo.get("id", "")
                
                # Get the original difficulty (stored as field's original value)
                original_difficulty = todo.get("difficulty", "medium")
                if isinstance(original_difficulty, (int, float)):
                    # Convert numeric difficulty back to string
                    difficulty_map_reverse = {0.1: "trivial", 1: "easy", 1.5: "medium", 2: "hard"}
                    original_difficulty = difficulty_map_reverse.get(original_difficulty, "medium")
                
                # Get current difficulty from form
                current_difficulty = next((f["value"] for f in self.form_fields if f["name"] == "difficulty"), "medium")
                
                # If difficulty changed, update it in Habitica
                if current_difficulty != original_difficulty and todo_id:
                    self.terminal_output.append(f"Difficulty changed from {original_difficulty} to {current_difficulty}")
                    self.update_task_difficulty(todo_id, current_difficulty)
            
            self.view_mode = "list"
            self.timer_running = False
            
        elif key == 10:  # Enter
            # Complete the objective
            
            # Also check for difficulty changes here
            if self.current_index < len(self.filtered_todos):
                todo = self.filtered_todos[self.current_index]
                todo_id = todo.get("id", "")
                
                # Get the original difficulty
                original_difficulty = todo.get("difficulty", "medium")
                if isinstance(original_difficulty, (int, float)):
                    # Convert numeric difficulty back to string
                    difficulty_map_reverse = {0.1: "trivial", 1: "easy", 1.5: "medium", 2: "hard"}
                    original_difficulty = difficulty_map_reverse.get(original_difficulty, "medium")
                
                # Get current difficulty from form
                current_difficulty = next((f["value"] for f in self.form_fields if f["name"] == "difficulty"), "medium")
                
                # If difficulty changed, update it in Habitica before completing
                if current_difficulty != original_difficulty and todo_id:
                    self.terminal_output.append(f"Difficulty changed from {original_difficulty} to {current_difficulty}")
                    success = self.update_task_difficulty(todo_id, current_difficulty)
                    if success:
                        self.terminal_output.append("Successfully updated difficulty in Habitica")
                    else:
                        self.terminal_output.append("Failed to update difficulty in Habitica")
            
            self.complete_current_objective()
            
        elif key == 9:  # TAB
            # Pause/resume timer
            self.toggle_timer()
        
        return True

    def run(self):
        """Main application loop."""
        running = True
        last_refresh = 0
        refresh_rate = 0.1  # 100ms minimum between refreshes
        
        while running:
            try:
                current_time = time.time()
                
                # Only refresh the screen if enough time has passed
                if current_time - last_refresh >= refresh_rate:
                    # Update timer display if running
                    if self.timer_running and self.view_mode == "run":
                        self.draw_screen()
                    else:
                        self.draw_screen()
                        
                    last_refresh = current_time
                
                # Process input without blocking
                running = self.handle_input()
                
                # Add a small sleep to prevent CPU hogging
                time.sleep(0.01)
                
            except Exception as e:
                # Prevent unexpected crashes
                self.terminal_output.append(f"Error in main loop: {str(e)}")
                self.stdscr.refresh()
                curses.napms(3000)  # Show error for 3 seconds

def test_habitica_api():
    """Simple test function to verify Habitica API credentials"""
    import requests
    import json
    import sys
    
    print("Testing Habitica API connection...")
    habitica_user_id = "8a9e85dc-d8f5-4c60-86ef-d70a19bf225e"
    habitica_api_token = "a4375d21-0a50-4ceb-a412-ebb70e927349"
    
    headers = {
        "x-api-user": habitica_user_id,
        "x-api-key": habitica_api_token,
        "Content-Type": "application/json"
    }
    
    try:
        print("Checking user account...")
        response = requests.get("https://habitica.com/api/v3/user", headers=headers)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}...")
        
        if response.status_code == 200:
            print("✓ API credentials are valid!")
            
            print("\nTesting task list retrieval...")
            todo_response = requests.get(
                "https://habitica.com/api/v3/tasks/user",
                params={"type": "todos"},  # FIXED: Changed from "todo" to "todos"
                headers=headers
            )
            print(f"Status: {todo_response.status_code}")
            
            if todo_response.status_code == 200:
                print("✓ Task retrieval successful!")
                data = todo_response.json()
                todos = data.get("data", [])
                print(f"Found {len(todos)} todo tasks")
            else:
                print("✗ Task retrieval failed")
                print(f"Error: {todo_response.text}")
        else:
            print("✗ API credentials are invalid or expired")
            print(f"Full error: {response.text}")
            return False
            
        return True
    except Exception as e:
        print(f"Error connecting to Habitica: {str(e)}")
        return False
    
def main(stdscr):
    """Initialize and run the application."""
    # Create and run the application
    app = HabiticaObjectiveTracker(stdscr)
    app.run()

if __name__ == "__main__":
    # Add command line argument parsing
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test-api":
        # Run API test only
        success = test_habitica_api()
        sys.exit(0 if success else 1)
    
    # Initialize curses wrapper
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)

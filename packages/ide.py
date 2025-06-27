#!/usr/bin/env python3
"""
Enhanced Multi-Script Runner
Features:
- Multiple script tabs
- Inter-script communication
- Syntax checking
- Real-time output monitoring
- Script dependency management
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import threading
import queue
import platform
import os
import sys
import json
import re
import ast
import tempfile
import time
from pathlib import Path
from datetime import datetime
import pylint.lint
import pylint.reporters.text
from io import StringIO
import tokenize
import keyword

class ScriptTab:
    """Represents a single script tab with its own editor and output"""
    def __init__(self, parent, tab_id, runner):
        self.parent = parent
        self.tab_id = tab_id
        self.runner = runner
        self.process = None
        self.output_queue = queue.Queue()
        self.file_path = None
        self.last_saved_content = ""
        self.syntax_errors = []
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI for this script tab"""
        # Main container
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # Paned window for editor and output
        paned = ttk.PanedWindow(self.frame, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Top frame for editor
        editor_frame = ttk.LabelFrame(paned, text="Script Editor", padding=5)
        paned.add(editor_frame, weight=3)
        
        # Toolbar
        toolbar = ttk.Frame(editor_frame)
        toolbar.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(toolbar, text="📁 Load", command=self.load_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="💾 Save", command=self.save_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="▶️ Run", command=self.run_script).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="⏹️ Stop", command=self.stop_script).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="✓ Check Syntax", command=self.check_syntax).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient='vertical').pack(side=tk.LEFT, fill='y', padx=5)
        
        # Script type
        ttk.Label(toolbar, text="Type:").pack(side=tk.LEFT, padx=2)
        self.script_type = tk.StringVar(value="python")
        type_combo = ttk.Combobox(toolbar, textvariable=self.script_type,
                                 values=["python", "bash", "node", "ruby"],
                                 width=8, state="readonly")
        type_combo.pack(side=tk.LEFT, padx=2)
        
        # Status label
        self.status_label = ttk.Label(toolbar, text="Ready", foreground="green")
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # Text editor with line numbers
        editor_container = ttk.Frame(editor_frame)
        editor_container.pack(fill=tk.BOTH, expand=True)
        
        # Line numbers
        self.line_numbers = tk.Text(editor_container, width=4, padx=3, takefocus=0,
                                   wrap=tk.NONE, state='disabled',
                                   font=("Consolas", 10))
        self.line_numbers.pack(side=tk.LEFT, fill=tk.Y)
        
        # Script editor
        self.editor = scrolledtext.ScrolledText(editor_container, wrap=tk.NONE,
                                               font=("Consolas", 10), undo=True)
        self.editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Syntax error display
        self.error_text = tk.Text(editor_frame, height=3, wrap=tk.WORD,
                                 font=("Consolas", 9), foreground="red",
                                 state='disabled')
        
        # Bind events
        self.editor.bind('<KeyRelease>', self.on_editor_change)
        self.editor.bind('<Return>', self.on_editor_change)
        self.editor.bind('<BackSpace>', self.on_editor_change)
        self.editor.bind('<<Modified>>', self.on_editor_modified)
        
        # Bottom frame for output
        output_frame = ttk.LabelFrame(paned, text="Output", padding=5)
        paned.add(output_frame, weight=2)
        
        # Output controls
        output_toolbar = ttk.Frame(output_frame)
        output_toolbar.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(output_toolbar, text="Clear", command=self.clear_output).pack(side=tk.LEFT, padx=2)
        ttk.Button(output_toolbar, text="Copy", command=self.copy_output).pack(side=tk.LEFT, padx=2)
        
        self.auto_scroll = tk.BooleanVar(value=True)
        ttk.Checkbutton(output_toolbar, text="Auto-scroll", 
                       variable=self.auto_scroll).pack(side=tk.LEFT, padx=10)
        
        # Output text
        self.output = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD,
                                               font=("Consolas", 9),
                                               state='disabled')
        self.output.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags for colored output
        self.output.tag_config("stdout", foreground="black")
        self.output.tag_config("stderr", foreground="red")
        self.output.tag_config("system", foreground="blue", font=("Consolas", 9, "italic"))
        
        # Update line numbers
        self.update_line_numbers()
        
    def on_editor_change(self, event=None):
        """Handle editor content changes"""
        self.update_line_numbers()
        # Schedule syntax check (debounced)
        if hasattr(self, '_syntax_check_timer'):
            self.frame.after_cancel(self._syntax_check_timer)
        self._syntax_check_timer = self.frame.after(1000, self.auto_check_syntax)
        
    def on_editor_modified(self, event=None):
        """Handle editor modification flag"""
        if self.editor.edit_modified():
            self.update_tab_title()
            self.editor.edit_modified(False)
            
    def update_tab_title(self):
        """Update tab title to show modification status"""
        current_content = self.editor.get(1.0, tk.END)
        is_modified = current_content != self.last_saved_content
        
        base_title = f"Script {self.tab_id}"
        if self.file_path:
            base_title = os.path.basename(self.file_path)
            
        title = f"{'*' if is_modified else ''}{base_title}"
        self.runner.notebook.tab(self.frame, text=title)
        
    def update_line_numbers(self):
        """Update line numbers in the editor"""
        self.line_numbers.config(state='normal')
        self.line_numbers.delete(1.0, tk.END)
        
        line_count = self.editor.get(1.0, tk.END).count('\n')
        line_numbers_string = "\n".join(str(i) for i in range(1, line_count + 1))
        self.line_numbers.insert(1.0, line_numbers_string)
        self.line_numbers.config(state='disabled')
        
    def load_file(self):
        """Load a script file"""
        filename = filedialog.askopenfilename(
            title="Open Script",
            filetypes=[
                ("All Scripts", "*.py;*.sh;*.js;*.rb"),
                ("Python files", "*.py"),
                ("Shell scripts", "*.sh;*.bash"),
                ("JavaScript", "*.js"),
                ("Ruby", "*.rb"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.editor.delete(1.0, tk.END)
                self.editor.insert(1.0, content)
                self.file_path = filename
                self.last_saved_content = content
                
                # Auto-detect script type
                ext = os.path.splitext(filename)[1].lower()
                if ext == '.py':
                    self.script_type.set("python")
                elif ext in ['.sh', '.bash']:
                    self.script_type.set("bash")
                elif ext == '.js':
                    self.script_type.set("node")
                elif ext == '.rb':
                    self.script_type.set("ruby")
                    
                self.update_tab_title()
                self.check_syntax()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
                
    def save_file(self):
        """Save the script to file"""
        if not self.file_path:
            self.file_path = filedialog.asksaveasfilename(
                title="Save Script",
                defaultextension=".py",
                filetypes=[
                    ("Python files", "*.py"),
                    ("Shell scripts", "*.sh"),
                    ("JavaScript", "*.js"),
                    ("Ruby", "*.rb"),
                    ("All files", "*.*")
                ]
            )
            
        if self.file_path:
            try:
                content = self.editor.get(1.0, tk.END)
                with open(self.file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.last_saved_content = content
                self.update_tab_title()
                self.add_output(f"Saved to: {self.file_path}\n", "system")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")
                
    def auto_check_syntax(self):
        """Automatically check syntax (debounced)"""
        if self.runner.config.get("auto_syntax_check", True):
            self.check_syntax(silent=True)
            
    def check_syntax(self, silent=False):
        """Check syntax of the current script"""
        content = self.editor.get(1.0, tk.END).strip()
        if not content:
            return
            
        script_type = self.script_type.get()
        self.syntax_errors = []
        
        try:
            if script_type == "python":
                self.check_python_syntax(content)
            elif script_type == "bash":
                self.check_bash_syntax(content)
            elif script_type == "node":
                self.check_javascript_syntax(content)
            elif script_type == "ruby":
                self.check_ruby_syntax(content)
                
            if self.syntax_errors:
                self.show_syntax_errors()
                self.status_label.config(text="Syntax errors found", foreground="red")
            else:
                self.hide_syntax_errors()
                self.status_label.config(text="Syntax OK", foreground="green")
                if not silent:
                    self.add_output("✓ Syntax check passed\n", "system")
                    
        except Exception as e:
            if not silent:
                self.add_output(f"Syntax check error: {str(e)}\n", "stderr")
                
    def check_python_syntax(self, content):
        """Check Python syntax"""
        try:
            # First, try to compile the code
            compile(content, '<string>', 'exec')
            
            # If that passes, run a more detailed check with pylint
            if self.runner.config.get("use_pylint", False):
                self.check_with_pylint(content)
                
        except SyntaxError as e:
            self.syntax_errors.append({
                'line': e.lineno or 1,
                'column': e.offset or 0,
                'message': str(e.msg),
                'type': 'error'
            })
            
    def check_with_pylint(self, content):
        """Run pylint for more detailed Python checks"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                temp_file = f.name
                
            # Run pylint
            output = StringIO()
            reporter = pylint.reporters.text.TextReporter(output)
            
            pylint.lint.Run([temp_file, '--errors-only'], reporter=reporter, do_exit=False)
            
            # Parse output
            for line in output.getvalue().split('\n'):
                match = re.match(r'.*:(\d+):(\d+): ([A-Z]\d+): (.+)', line)
                if match:
                    self.syntax_errors.append({
                        'line': int(match.group(1)),
                        'column': int(match.group(2)),
                        'message': f"{match.group(3)}: {match.group(4)}",
                        'type': 'warning' if match.group(3).startswith('W') else 'error'
                    })
                    
            os.unlink(temp_file)
            
        except Exception:
            pass  # Fail silently for pylint errors
            
    def check_bash_syntax(self, content):
        """Check Bash syntax"""
        try:
            # Use bash -n for syntax checking
            result = subprocess.run(['bash', '-n'], input=content, 
                                  capture_output=True, text=True)
            if result.stderr:
                for line in result.stderr.split('\n'):
                    match = re.search(r'line (\d+):', line)
                    if match:
                        self.syntax_errors.append({
                            'line': int(match.group(1)),
                            'column': 0,
                            'message': line,
                            'type': 'error'
                        })
        except FileNotFoundError:
            pass  # Bash not available
            
    def check_javascript_syntax(self, content):
        """Check JavaScript syntax"""
        try:
            # Use node --check for syntax checking
            result = subprocess.run(['node', '--check'], input=content,
                                  capture_output=True, text=True)
            if result.stderr:
                for line in result.stderr.split('\n'):
                    match = re.search(r':(\d+)', line)
                    if match:
                        self.syntax_errors.append({
                            'line': int(match.group(1)),
                            'column': 0,
                            'message': line,
                            'type': 'error'
                        })
        except FileNotFoundError:
            pass  # Node not available
            
    def check_ruby_syntax(self, content):
        """Check Ruby syntax"""
        try:
            # Use ruby -c for syntax checking
            result = subprocess.run(['ruby', '-c'], input=content,
                                  capture_output=True, text=True)
            if result.stderr and 'Syntax OK' not in result.stderr:
                for line in result.stderr.split('\n'):
                    match = re.search(r':(\d+):', line)
                    if match:
                        self.syntax_errors.append({
                            'line': int(match.group(1)),
                            'column': 0,
                            'message': line,
                            'type': 'error'
                        })
        except FileNotFoundError:
            pass  # Ruby not available
            
    def show_syntax_errors(self):
        """Display syntax errors below the editor"""
        if not self.error_text.winfo_viewable():
            self.error_text.pack(fill=tk.X, pady=(5, 0))
            
        self.error_text.config(state='normal')
        self.error_text.delete(1.0, tk.END)
        
        for error in self.syntax_errors[:3]:  # Show max 3 errors
            error_msg = f"Line {error['line']}: {error['message']}\n"
            self.error_text.insert(tk.END, error_msg)
            
        if len(self.syntax_errors) > 3:
            self.error_text.insert(tk.END, f"... and {len(self.syntax_errors) - 3} more errors")
            
        self.error_text.config(state='disabled')
        
        # Highlight error lines in editor
        self.editor.tag_remove("error", 1.0, tk.END)
        for error in self.syntax_errors:
            line_start = f"{error['line']}.0"
            line_end = f"{error['line']}.end"
            self.editor.tag_add("error", line_start, line_end)
            
        self.editor.tag_config("error", background="#ffcccc")
        
    def hide_syntax_errors(self):
        """Hide syntax error display"""
        self.error_text.pack_forget()
        self.editor.tag_remove("error", 1.0, tk.END)
        
    def run_script(self):
        """Run the current script"""
        if self.process and self.process.poll() is None:
            messagebox.showwarning("Warning", "Script is already running!")
            return
            
        content = self.editor.get(1.0, tk.END).strip()
        if not content:
            messagebox.showwarning("Warning", "No script to run!")
            return
            
        # Check syntax first
        self.check_syntax(silent=True)
        if self.syntax_errors and self.runner.config.get("block_on_syntax_error", True):
            if not messagebox.askyesno("Syntax Errors", 
                                     "Syntax errors found. Run anyway?"):
                return
                
        # Save to temporary file
        script_type = self.script_type.get()
        suffix = {
            'python': '.py',
            'bash': '.sh',
            'node': '.js',
            'ruby': '.rb'
        }.get(script_type, '.txt')
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(content)
            self.temp_script = f.name
            
        # Prepare command
        if script_type == "python":
            cmd = [sys.executable, self.temp_script]
        elif script_type == "bash":
            if platform.system() != "Windows":
                os.chmod(self.temp_script, 0o755)
            cmd = ["bash", self.temp_script]
        elif script_type == "node":
            cmd = ["node", self.temp_script]
        elif script_type == "ruby":
            cmd = ["ruby", self.temp_script]
        else:
            cmd = [self.temp_script]
            
        # Clear output
        self.clear_output()
        self.add_output(f"Running {script_type} script...\n", "system")
        self.add_output(f"Command: {' '.join(cmd)}\n", "system")
        self.add_output("-" * 50 + "\n", "system")
        
        # Start process
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env={**os.environ, **self.runner.shared_env}
            )
            
            self.status_label.config(text="Running...", foreground="orange")
            
            # Start output monitoring threads
            threading.Thread(target=self.monitor_output, args=(self.process.stdout, "stdout"), daemon=True).start()
            threading.Thread(target=self.monitor_output, args=(self.process.stderr, "stderr"), daemon=True).start()
            threading.Thread(target=self.monitor_process, daemon=True).start()
            
        except Exception as e:
            self.add_output(f"Failed to start script: {str(e)}\n", "stderr")
            self.status_label.config(text="Error", foreground="red")
            
    def monitor_output(self, pipe, tag):
        """Monitor process output"""
        try:
            for line in pipe:
                self.output_queue.put((line, tag))
                self.frame.after(0, self.process_output_queue)
        except:
            pass
            
    def monitor_process(self):
        """Monitor process completion"""
        self.process.wait()
        return_code = self.process.returncode
        
        self.frame.after(0, lambda: self.on_process_complete(return_code))
        
    def process_output_queue(self):
        """Process queued output"""
        try:
            while True:
                line, tag = self.output_queue.get_nowait()
                self.add_output(line, tag)
        except queue.Empty:
            pass
            
    def on_process_complete(self, return_code):
        """Handle process completion"""
        self.add_output("-" * 50 + "\n", "system")
        self.add_output(f"Process exited with code: {return_code}\n", "system")
        
        if return_code == 0:
            self.status_label.config(text="Completed", foreground="green")
        else:
            self.status_label.config(text=f"Error (code {return_code})", foreground="red")
            
        # Clean up temp file
        try:
            if hasattr(self, 'temp_script') and os.path.exists(self.temp_script):
                os.unlink(self.temp_script)
        except:
            pass
            
        self.process = None
        
    def stop_script(self):
        """Stop the running script"""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.add_output("\nScript terminated by user\n", "system")
            self.status_label.config(text="Terminated", foreground="red")
            
    def add_output(self, text, tag="stdout"):
        """Add text to output"""
        self.output.config(state='normal')
        self.output.insert(tk.END, text, tag)
        if self.auto_scroll.get():
            self.output.see(tk.END)
        self.output.config(state='disabled')
        
    def clear_output(self):
        """Clear output text"""
        self.output.config(state='normal')
        self.output.delete(1.0, tk.END)
        self.output.config(state='disabled')
        
    def copy_output(self):
        """Copy output to clipboard"""
        content = self.output.get(1.0, tk.END)
        self.frame.clipboard_clear()
        self.frame.clipboard_append(content)
        
    def send_input(self, text):
        """Send input to the running process"""
        if self.process and self.process.poll() is None:
            try:
                self.process.stdin.write(text + '\n')
                self.process.stdin.flush()
                self.add_output(f">>> {text}\n", "system")
            except:
                self.add_output("Failed to send input\n", "stderr")


class MultiScriptRunner:
    """Main application for running multiple scripts"""
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Multi-Script Runner")
        self.root.geometry("1200x800")
        
        self.tabs = {}
        self.tab_counter = 0
        self.shared_env = {}  # Shared environment variables
        
        # Load configuration
        self.config_file = Path.home() / ".multi_script_runner_config.json"
        self.load_config()
        
        self.setup_ui()
        self.create_new_tab()
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-t>', lambda e: self.create_new_tab())
        self.root.bind('<Control-w>', lambda e: self.close_current_tab())
        self.root.bind('<Control-Tab>', lambda e: self.next_tab())
        self.root.bind('<Control-Shift-Tab>', lambda e: self.prev_tab())
        
    def load_config(self):
        """Load configuration"""
        default_config = {
            "auto_syntax_check": True,
            "block_on_syntax_error": False,
            "use_pylint": False,
            "terminal_theme": "dark",
            "font_size": 10,
            "shared_env_vars": {}
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.config = {**default_config, **json.load(f)}
            except:
                self.config = default_config
        else:
            self.config = default_config
            
    def save_config(self):
        """Save configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
            
    def setup_ui(self):
        """Setup main UI"""
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Tab (Ctrl+T)", command=self.create_new_tab)
        file_menu.add_command(label="Close Tab (Ctrl+W)", command=self.close_current_tab)
        file_menu.add_separator()
        file_menu.add_command(label="Save All", command=self.save_all_tabs)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit_app)
        
        # Scripts menu
        scripts_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Scripts", menu=scripts_menu)
        scripts_menu.add_command(label="Run All", command=self.run_all_scripts)
        scripts_menu.add_command(label="Stop All", command=self.stop_all_scripts)
        scripts_menu.add_separator()
        scripts_menu.add_command(label="Check All Syntax", command=self.check_all_syntax)
        scripts_menu.add_separator()
        scripts_menu.add_command(label="Pipe Output to Next Tab", command=self.setup_pipe)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Environment Variables", command=self.manage_env_vars)
        tools_menu.add_command(label="Inter-Script Communication", command=self.setup_communication)
        tools_menu.add_separator()
        tools_menu.add_command(label="Settings", command=self.open_settings)
        
        # Main container
        main_frame = ttk.Frame(self.root, padding=5)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_frame = ttk.Frame(main_frame)
        self.status_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.status_label = ttk.Label(self.status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT)
        
        ttk.Label(self.status_frame, text=" | ").pack(side=tk.LEFT)
        
        self.tab_count_label = ttk.Label(self.status_frame, text="Tabs: 0")
        self.tab_count_label.pack(side=tk.LEFT)
        
    def create_new_tab(self):
        """Create a new script tab"""
        self.tab_counter += 1
        tab_id = self.tab_counter
        
        tab = ScriptTab(self.notebook, tab_id, self)
        self.tabs[tab_id] = tab
        
        self.notebook.add(tab.frame, text=f"Script {tab_id}")
        self.notebook.select(tab.frame)
        
        self.update_tab_count()
        
    def close_current_tab(self):
        """Close the current tab"""
        if len(self.tabs) <= 1:
            messagebox.showwarning("Warning", "Cannot close the last tab!")
            return
            
        current_tab = self.notebook.select()
        for tab_id, tab in self.tabs.items():
            if str(tab.frame) == current_tab:
                # Check if script is running
                if tab.process and tab.process.poll() is None:
                    if not messagebox.askyesno("Script Running", 
                                             "Script is still running. Close anyway?"):
                        return
                    tab.stop_script()
                    
                # Check if content is unsaved
                current_content = tab.editor.get(1.0, tk.END)
                if current_content != tab.last_saved_content:
                    if not messagebox.askyesno("Unsaved Changes",
                                             "Tab has unsaved changes. Close anyway?"):
                        return
                        
                self.notebook.forget(current_tab)
                del self.tabs[tab_id]
                self.update_tab_count()
                break
                
    def next_tab(self):
        """Switch to next tab"""
        tabs = self.notebook.tabs()
        if tabs:
            current = self.notebook.select()
            current_index = tabs.index(current)
            next_index = (current_index + 1) % len(tabs)
            self.notebook.select(tabs[next_index])
            
    def prev_tab(self):
        """Switch to previous tab"""
        tabs = self.notebook.tabs()
        if tabs:
            current = self.notebook.select()
            current_index = tabs.index(current)
            prev_index = (current_index - 1) % len(tabs)
            self.notebook.select(tabs[prev_index])
            
    def update_tab_count(self):
        """Update tab count in status bar"""
        self.tab_count_label.config(text=f"Tabs: {len(self.tabs)}")
        
    def save_all_tabs(self):
        """Save all tabs"""
        for tab in self.tabs.values():
            if tab.editor.get(1.0, tk.END).strip():
                tab.save_file()
                
    def run_all_scripts(self):
        """Run all scripts"""
        for tab in self.tabs.values():
            if tab.editor.get(1.0, tk.END).strip():
                tab.run_script()
                time.sleep(0.1)  # Small delay between starts
                
    def stop_all_scripts(self):
        """Stop all running scripts"""
        for tab in self.tabs.values():
            tab.stop_script()
            
    def check_all_syntax(self):
        """Check syntax for all scripts"""
        errors_found = False
        for tab in self.tabs.values():
            if tab.editor.get(1.0, tk.END).strip():
                tab.check_syntax()
                if tab.syntax_errors:
                    errors_found = True
                    
        if not errors_found:
            messagebox.showinfo("Syntax Check", "All scripts passed syntax check!")
            
    def manage_env_vars(self):
        """Manage shared environment variables"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Environment Variables")
        dialog.geometry("600x400")
        
        ttk.Label(dialog, text="Shared Environment Variables:", 
                 font=("Arial", 10, "bold")).pack(pady=10)
        
        # Text editor for env vars
        env_text = scrolledtext.ScrolledText(dialog, height=15, width=60)
        env_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        # Load current env vars
        env_content = "\n".join([f"{k}={v}" for k, v in self.shared_env.items()])
        env_text.insert(1.0, env_content)
        
        def save_env_vars():
            # Parse and save env vars
            self.shared_env.clear()
            for line in env_text.get(1.0, tk.END).strip().split('\n'):
                if '=' in line:
                    k, v = line.split('=', 1)
                    self.shared_env[k.strip()] = v.strip()
                    
            self.config["shared_env_vars"] = self.shared_env
            self.save_config()
            messagebox.showinfo("Success", "Environment variables updated!")
            dialog.destroy()
            
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Save", command=save_env_vars).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
    def setup_pipe(self):
        """Setup pipe between tabs"""
        tabs_list = list(self.tabs.values())
        if len(tabs_list) < 2:
            messagebox.showwarning("Warning", "Need at least 2 tabs to setup pipe!")
            return
            
        dialog = tk.Toplevel(self.root)
        dialog.title("Setup Pipe")
        dialog.geometry("400x200")
        
        ttk.Label(dialog, text="Pipe output from:").grid(row=0, column=0, padx=10, pady=10)
        
        source_var = tk.StringVar()
        source_combo = ttk.Combobox(dialog, textvariable=source_var, width=30)
        source_combo['values'] = [f"Script {tab.tab_id}" for tab in tabs_list]
        source_combo.grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Label(dialog, text="To input of:").grid(row=1, column=0, padx=10, pady=10)
        
        dest_var = tk.StringVar()
        dest_combo = ttk.Combobox(dialog, textvariable=dest_var, width=30)
        dest_combo['values'] = [f"Script {tab.tab_id}" for tab in tabs_list]
        dest_combo.grid(row=1, column=1, padx=10, pady=10)
        
        def create_pipe():
            # Implementation would involve redirecting output
            # This is a placeholder for the concept
            messagebox.showinfo("Info", "Pipe setup feature coming soon!")
            dialog.destroy()
            
        ttk.Button(dialog, text="Create Pipe", command=create_pipe).grid(row=2, column=0, columnspan=2, pady=20)
        
    def setup_communication(self):
        """Setup inter-script communication"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Inter-Script Communication")
        dialog.geometry("600x400")
        
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Shared files tab
        files_frame = ttk.Frame(notebook)
        notebook.add(files_frame, text="Shared Files")
        
        ttk.Label(files_frame, text="Scripts can communicate via shared files:").pack(pady=10)
        
        shared_dir = Path.home() / ".multi_script_runner" / "shared"
        shared_dir.mkdir(parents=True, exist_ok=True)
        
        ttk.Label(files_frame, text=f"Shared directory: {shared_dir}").pack(pady=5)
        
        # Message queue tab
        queue_frame = ttk.Frame(notebook)
        notebook.add(queue_frame, text="Message Queue")
        
        ttk.Label(queue_frame, text="Scripts can send messages via a queue:").pack(pady=10)
        ttk.Label(queue_frame, text="(Implementation coming soon)").pack(pady=5)
        
        # Sockets tab
        socket_frame = ttk.Frame(notebook)
        notebook.add(socket_frame, text="Sockets")
        
        ttk.Label(socket_frame, text="Scripts can communicate via sockets:").pack(pady=10)
        ttk.Label(socket_frame, text="(Implementation coming soon)").pack(pady=5)
        
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
        
    def open_settings(self):
        """Open settings dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Settings")
        dialog.geometry("500x400")
        
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Editor settings
        editor_frame = ttk.Frame(notebook)
        notebook.add(editor_frame, text="Editor")
        
        settings = [
            ("Auto syntax check", "auto_syntax_check", "bool"),
            ("Block execution on syntax error", "block_on_syntax_error", "bool"),
            ("Use pylint for Python", "use_pylint", "bool"),
            ("Font size", "font_size", "int"),
        ]
        
        vars = {}
        row = 0
        
        for label, key, type_ in settings:
            ttk.Label(editor_frame, text=label + ":").grid(row=row, column=0, sticky=tk.W, padx=10, pady=5)
            
            if type_ == "bool":
                var = tk.BooleanVar(value=self.config.get(key, False))
                ttk.Checkbutton(editor_frame, variable=var).grid(row=row, column=1, sticky=tk.W, padx=10, pady=5)
            elif type_ == "int":
                var = tk.IntVar(value=self.config.get(key, 10))
                ttk.Spinbox(editor_frame, from_=8, to=20, textvariable=var, width=10).grid(row=row, column=1, sticky=tk.W, padx=10, pady=5)
                
            vars[key] = var
            row += 1
            
        def save_settings():
            for key, var in vars.items():
                self.config[key] = var.get()
            self.save_config()
            messagebox.showinfo("Success", "Settings saved!")
            dialog.destroy()
            
        ttk.Button(dialog, text="Save", command=save_settings).pack(pady=10)
        
    def quit_app(self):
        """Quit application"""
        # Check for running scripts
        running_scripts = [tab for tab in self.tabs.values() 
                          if tab.process and tab.process.poll() is None]
        
        if running_scripts:
            if not messagebox.askyesno("Scripts Running",
                                     f"{len(running_scripts)} scripts are still running. Quit anyway?"):
                return
                
        # Check for unsaved changes
        unsaved_tabs = [tab for tab in self.tabs.values()
                       if tab.editor.get(1.0, tk.END) != tab.last_saved_content]
        
        if unsaved_tabs:
            if not messagebox.askyesno("Unsaved Changes",
                                     f"{len(unsaved_tabs)} tabs have unsaved changes. Quit anyway?"):
                return
                
        self.root.quit()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = MultiScriptRunner(root)
    
    # Set icon if available
    try:
        if platform.system() == "Windows":
            root.iconbitmap(default='icon.ico')
    except:
        pass
        
    root.mainloop()


if __name__ == "__main__":
    main()

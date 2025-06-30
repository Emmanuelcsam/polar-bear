#!/usr/bin/env python3
"""
Neural Nexus IDE - Advanced Neural Network Development Environment
A revolutionary IDE for rapid AI script development with automatic error correction,
visualization, and seamless AI integration.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import threading
import queue
import os
import sys
import json
import time
import tempfile
import shutil
import webbrowser
import platform
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import ast
import re
import psutil
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')

# Optional imports for enhanced features
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False

# Constants
APP_NAME = "Neural Nexus IDE"
APP_VERSION = "3.0.0"
CONFIG_FILE = Path.home() / ".neural_nexus" / "config.json"
SCRIPTS_DIR = Path.home() / ".neural_nexus" / "scripts"
LOGS_DIR = Path.home() / ".neural_nexus" / "logs"

# Ensure directories exist
CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class Script:
    """Represents a script in the IDE"""
    id: str
    name: str
    content: str
    file_path: Optional[Path] = None
    dependencies: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "idle"  # idle, running, error, success
    process: Optional[subprocess.Popen] = None
    output: str = ""
    error_output: str = ""

class NeuralNexusIDE:
    """Main IDE Application"""

    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.root.geometry("1600x900")

        # Core state
        self.scripts: Dict[str, Script] = {}
        self.active_script_id: Optional[str] = None
        self.config = self.load_config()
        self.output_queue = queue.Queue()
        self.auto_heal_active = False
        self.terminal_process = None

        # AI clients
        self.setup_ai_clients()

        # UI Setup
        self.setup_ui()
        self.setup_styles()
        self.bind_shortcuts()

        # Start background threads
        self.start_output_processor()
        self.start_dependency_analyzer()

        # Show welcome
        self.show_welcome()

    def setup_ai_clients(self):
        """Initialize AI clients"""
        self.openai_client = None
        if HAS_OPENAI and self.config.get('openai_api_key'):
            try:
                openai.api_key = self.config['openai_api_key']
                self.openai_client = openai
            except Exception as e:
                print(f"Failed to initialize OpenAI: {e}")

        self.copilot_available = self.check_copilot_availability()

    def check_copilot_availability(self):
        """Check if GitHub Copilot CLI is available"""
        try:
            result = subprocess.run(['gh', 'copilot', '--version'],
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False

    def setup_ui(self):
        """Create the main UI layout"""
        # Main container
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Left panel - Script Manager
        left_panel = ttk.Frame(main_container, width=300)
        main_container.add(left_panel, weight=1)
        self.create_script_manager(left_panel)

        # Center panel - Editor
        center_panel = ttk.Frame(main_container)
        main_container.add(center_panel, weight=3)
        self.create_editor_panel(center_panel)

        # Right panel - Visualization & Tools
        right_panel = ttk.Frame(main_container, width=400)
        main_container.add(right_panel, weight=2)
        self.create_tools_panel(right_panel)

        # Status bar
        self.create_status_bar()

    def create_script_manager(self, parent):
        """Create script management panel"""
        # Header
        header = ttk.Frame(parent)
        header.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(header, text="Scripts", font=('Arial', 12, 'bold')).pack(side=tk.LEFT)

        # Buttons
        btn_frame = ttk.Frame(header)
        btn_frame.pack(side=tk.RIGHT)

        ttk.Button(btn_frame, text="➕", width=3, command=self.new_script).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="📁", width=3, command=self.open_script).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="💾", width=3, command=self.save_script).pack(side=tk.LEFT)

        # Script list
        list_frame = ttk.Frame(parent)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5)

        self.script_tree = ttk.Treeview(list_frame, columns=('status',), show='tree headings')
        self.script_tree.heading('#0', text='Name')
        self.script_tree.heading('status', text='Status')
        self.script_tree.column('status', width=80)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.script_tree.yview)
        self.script_tree.configure(yscrollcommand=scrollbar.set)

        self.script_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.script_tree.bind('<<TreeviewSelect>>', self.on_script_select)
        self.script_tree.bind('<Double-Button-1>', lambda e: self.run_script())

    def create_editor_panel(self, parent):
        """Create code editor panel"""
        # Toolbar
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, padx=5, pady=2)

        # Quick actions
        ttk.Button(toolbar, text="▶️ Run", command=self.run_script).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="⏹️ Stop", command=self.stop_script).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🔧 Auto Fix", command=self.auto_fix_errors).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🔄 Auto Heal", command=self.toggle_auto_heal).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)

        ttk.Button(toolbar, text="📦 Install Deps", command=self.install_dependencies).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🔍 Analyze", command=self.analyze_script).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="💡 AI Suggest", command=self.get_ai_suggestions).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🚁 Copilot", command=self.use_copilot).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📝 VS Code", command=self.open_in_vscode).pack(side=tk.LEFT, padx=2)

        # Editor notebook
        self.editor_notebook = ttk.Notebook(parent)
        self.editor_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Code editor tab
        editor_frame = ttk.Frame(self.editor_notebook)
        self.editor_notebook.add(editor_frame, text="Code")

        # Line numbers and editor
        editor_container = ttk.Frame(editor_frame)
        editor_container.pack(fill=tk.BOTH, expand=True)

        self.line_numbers = tk.Text(editor_container, width=4, padx=3, takefocus=0,
                                   wrap=tk.NONE, state='disabled', font=('Consolas', 11))
        self.line_numbers.pack(side=tk.LEFT, fill=tk.Y)

        self.editor = scrolledtext.ScrolledText(editor_container, wrap=tk.NONE,
                                               undo=True, font=('Consolas', 11))
        self.editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.editor.bind('<KeyRelease>', self.on_editor_change)
        self.editor.bind('<Control-Return>', lambda e: self.run_script())

        # Output tab
        output_frame = ttk.Frame(self.editor_notebook)
        self.editor_notebook.add(output_frame, text="Output")

        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD,
                                                    font=('Consolas', 10))
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # Configure tags for output
        self.output_text.tag_config("stdout", foreground="white")
        self.output_text.tag_config("stderr", foreground="#ff6b6b")
        self.output_text.tag_config("info", foreground="#51cf66")
        self.output_text.tag_config("warning", foreground="#ffd93d")

    def create_tools_panel(self, parent):
        """Create visualization and tools panel"""
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Neural Network Visualization
        viz_frame = ttk.Frame(notebook)
        notebook.add(viz_frame, text="Network View")
        self.create_network_viz(viz_frame)

        # Error Analysis
        error_frame = ttk.Frame(notebook)
        notebook.add(error_frame, text="Errors")
        self.create_error_panel(error_frame)

        # Dependencies
        deps_frame = ttk.Frame(notebook)
        notebook.add(deps_frame, text="Dependencies")
        self.create_deps_panel(deps_frame)

        # Terminal
        terminal_frame = ttk.Frame(notebook)
        notebook.add(terminal_frame, text="Terminal")
        self.create_terminal_panel(terminal_frame)

        # Settings
        settings_frame = ttk.Frame(notebook)
        notebook.add(settings_frame, text="Settings")
        self.create_settings_panel(settings_frame)

    def create_network_viz(self, parent):
        """Create neural network visualization"""
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, pady=2)

        ttk.Button(toolbar, text="Refresh", command=self.update_network_viz).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Layout", command=self.change_network_layout).pack(side=tk.LEFT, padx=2)

        self.viz_canvas = tk.Canvas(parent, bg='black')
        self.viz_canvas.pack(fill=tk.BOTH, expand=True)

        # Initialize matplotlib figure
        self.fig = plt.Figure(figsize=(5, 4), dpi=100, facecolor='black')
        self.ax = self.fig.add_subplot(111, facecolor='black')
        self.canvas = FigureCanvasTkAgg(self.fig, self.viz_canvas)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_error_panel(self, parent):
        """Create error analysis panel"""
        # Error tree
        self.error_tree = ttk.Treeview(parent, columns=('type', 'line', 'message'), show='tree headings')
        self.error_tree.heading('#0', text='Script')
        self.error_tree.heading('type', text='Type')
        self.error_tree.heading('line', text='Line')
        self.error_tree.heading('message', text='Message')

        self.error_tree.column('#0', width=100)
        self.error_tree.column('type', width=80)
        self.error_tree.column('line', width=50)

        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.error_tree.yview)
        self.error_tree.configure(yscrollcommand=scrollbar.set)

        self.error_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.error_tree.bind('<Double-Button-1>', self.jump_to_error)

    def create_deps_panel(self, parent):
        """Create dependencies panel"""
        # Commands frame
        cmd_frame = ttk.Frame(parent)
        cmd_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(cmd_frame, text="Quick Install:").pack(side=tk.LEFT)
        self.deps_entry = ttk.Entry(cmd_frame)
        self.deps_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(cmd_frame, text="Install", command=self.quick_install).pack(side=tk.LEFT)

        # Dependencies list
        self.deps_text = scrolledtext.ScrolledText(parent, height=10, font=('Consolas', 10))
        self.deps_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_terminal_panel(self, parent):
        """Create integrated terminal panel"""
        # Terminal type selector
        term_frame = ttk.Frame(parent)
        term_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(term_frame, text="Terminal:").pack(side=tk.LEFT)
        self.terminal_var = tk.StringVar(value="auto")

        terminals = [("Auto", "auto"), ("Git Bash", "gitbash"), ("WSL", "wsl"), ("CMD", "cmd")]
        for text, value in terminals:
            ttk.Radiobutton(term_frame, text=text, variable=self.terminal_var,
                           value=value).pack(side=tk.LEFT, padx=5)

        # Command entry
        cmd_frame = ttk.Frame(parent)
        cmd_frame.pack(fill=tk.X, padx=5, pady=5)

        self.terminal_entry = ttk.Entry(cmd_frame, font=('Consolas', 10))
        self.terminal_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.terminal_entry.bind('<Return>', self.execute_terminal_command)

        ttk.Button(cmd_frame, text="Run", command=self.execute_terminal_command).pack(side=tk.LEFT, padx=5)

        # Terminal output
        self.terminal_output = scrolledtext.ScrolledText(parent, font=('Consolas', 10),
                                                        bg='black', fg='white')
        self.terminal_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_settings_panel(self, parent):
        """Create settings panel"""
        settings_frame = ttk.LabelFrame(parent, text="Configuration", padding=10)
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # OpenAI API Key
        api_frame = ttk.Frame(settings_frame)
        api_frame.pack(fill=tk.X, pady=5)

        ttk.Label(api_frame, text="OpenAI API Key:").pack(side=tk.LEFT)
        self.api_key_var = tk.StringVar(value=self.config.get('openai_api_key', ''))
        api_entry = ttk.Entry(api_frame, textvariable=self.api_key_var, show='*')
        api_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(api_frame, text="Save", command=self.save_api_key).pack(side=tk.LEFT)

        # Auto-save
        self.auto_save_var = tk.BooleanVar(value=self.config.get('auto_save', True))
        ttk.Checkbutton(settings_frame, text="Auto-save scripts",
                       variable=self.auto_save_var, command=self.update_config).pack(anchor=tk.W, pady=5)

        # Auto-analyze
        self.auto_analyze_var = tk.BooleanVar(value=self.config.get('auto_analyze', True))
        ttk.Checkbutton(settings_frame, text="Auto-analyze on change",
                       variable=self.auto_analyze_var, command=self.update_config).pack(anchor=tk.W, pady=5)

        # Theme
        theme_frame = ttk.Frame(settings_frame)
        theme_frame.pack(fill=tk.X, pady=5)

        ttk.Label(theme_frame, text="Theme:").pack(side=tk.LEFT)
        self.theme_var = tk.StringVar(value=self.config.get('theme', 'dark'))
        ttk.Radiobutton(theme_frame, text="Dark", variable=self.theme_var,
                       value="dark", command=self.apply_theme).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(theme_frame, text="Light", variable=self.theme_var,
                       value="light", command=self.apply_theme).pack(side=tk.LEFT, padx=5)

    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = ttk.Label(self.status_bar, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=5)

        ttk.Separator(self.status_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self.script_status_label = ttk.Label(self.status_bar, text="No script loaded")
        self.script_status_label.pack(side=tk.LEFT, padx=5)

        # Progress bar for auto-heal
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.status_bar, variable=self.progress_var,
                                           length=200, mode='determinate')

    # Core Functionality

    def new_script(self):
        """Create a new script"""
        name = tk.simpledialog.askstring("New Script", "Enter script name:")
        if name:
            script_id = f"script_{int(time.time())}"
            script = Script(
                id=script_id,
                name=name,
                content="#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n\n"
            )
            self.scripts[script_id] = script
            self.add_script_to_tree(script)
            self.select_script(script_id)

    def open_script(self):
        """Open a script file"""
        filename = filedialog.askopenfilename(
            title="Open Script",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        if filename:
            path = Path(filename)
            try:
                content = path.read_text(encoding='utf-8')
                script_id = f"script_{int(time.time())}"
                script = Script(
                    id=script_id,
                    name=path.name,
                    content=content,
                    file_path=path
                )
                self.scripts[script_id] = script
                self.add_script_to_tree(script)
                self.select_script(script_id)
                self.analyze_script()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open file: {e}")

    def save_script(self):
        """Save current script"""
        if not self.active_script_id:
            return

        script = self.scripts[self.active_script_id]
        script.content = self.editor.get(1.0, tk.END)

        if script.file_path:
            try:
                script.file_path.write_text(script.content, encoding='utf-8')
                self.update_status(f"Saved: {script.file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")
        else:
            self.save_script_as()

    def save_script_as(self):
        """Save script with new name"""
        if not self.active_script_id:
            return

        script = self.scripts[self.active_script_id]
        filename = filedialog.asksaveasfilename(
            title="Save Script As",
            defaultextension=".py",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        if filename:
            path = Path(filename)
            script.file_path = path
            script.name = path.name
            self.save_script()
            self.update_script_tree()

    def run_script(self):
        """Run current script"""
        if not self.active_script_id:
            return

        script = self.scripts[self.active_script_id]

        # Save script to temp file if needed
        if not script.file_path:
            temp_file = SCRIPTS_DIR / f"{script.name}.py"
            temp_file.write_text(script.content, encoding='utf-8')
            script.file_path = temp_file

        # Clear output
        self.output_text.delete(1.0, tk.END)
        script.output = ""
        script.error_output = ""

        # Update status
        script.status = "running"
        self.update_script_tree()
        self.editor_notebook.select(1)  # Switch to output tab

        # Determine Python command
        if sys.platform == "win32" and shutil.which("uv"):
            python_cmd = ["uv", "run", "python"]
        else:
            python_cmd = [sys.executable]

        # Start process
        try:
            script.process = subprocess.Popen(
                python_cmd + [str(script.file_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Start output monitoring
            threading.Thread(target=self.monitor_output, args=(script,), daemon=True).start()

        except Exception as e:
            self.add_output(f"Failed to run script: {e}\n", "stderr")
            script.status = "error"
            self.update_script_tree()

    def stop_script(self):
        """Stop running script"""
        if not self.active_script_id:
            return

        script = self.scripts[self.active_script_id]
        if script.process and script.process.poll() is None:
            script.process.terminate()
            self.add_output("\n--- Script terminated ---\n", "info")
            script.status = "idle"
            self.update_script_tree()

    def monitor_output(self, script):
        """Monitor script output"""
        def read_output(pipe, tag):
            for line in iter(pipe.readline, ''):
                if line:
                    self.output_queue.put((script.id, line, tag))
                    if tag == "stderr":
                        script.error_output += line
                    else:
                        script.output += line

        # Start threads for stdout and stderr
        stdout_thread = threading.Thread(target=read_output, args=(script.process.stdout, "stdout"))
        stderr_thread = threading.Thread(target=read_output, args=(script.process.stderr, "stderr"))

        stdout_thread.start()
        stderr_thread.start()

        # Wait for process to complete
        script.process.wait()
        stdout_thread.join()
        stderr_thread.join()

        # Update status
        if script.process.returncode == 0:
            script.status = "success"
            self.output_queue.put((script.id, "\n--- Script completed successfully ---\n", "info"))
        else:
            script.status = "error"
            self.output_queue.put((script.id, f"\n--- Script failed with code {script.process.returncode} ---\n", "stderr"))

            # Auto-heal if enabled
            if self.auto_heal_active:
                self.root.after(1000, lambda: self.auto_fix_runtime_error(script))

        self.update_script_tree()

    def analyze_script(self):
        """Analyze current script for syntax & then kick off async dependency check."""
        if not self.active_script_id:
            return

        script = self.scripts[self.active_script_id]
        script.errors.clear()
        script.imports.clear()

        # Quick syntax & import extraction on main thread
        try:
            tree = ast.parse(script.content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        script.imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    script.imports.append(node.module)
        except SyntaxError as e:
            script.errors.append({
                'type': 'SyntaxError',
                'line': e.lineno,
                'message': str(e.msg),
                'offset': e.offset
            })

        # Offload the heavy pip-list call so the UI stays responsive
        # … inside analyze_script …
        threading.Thread(
            target=self.check_dependencies,    # ← call the real method
            args=(script,),
            daemon=True
        ).start()


    def check_dependencies(self, script):
        """Public API unchanged—now offloads work asynchronously."""
        threading.Thread(
            target=self._check_dependencies_worker,
            args=(script,),
            daemon=True
        ).start()

    def _check_dependencies_worker(self, script):
        """Background worker: blocking pip‑list + UI update back on main thread."""
        installed = self.get_installed_packages()

        missing = []
        for imp in script.imports:
            base = imp.split('.')[0]
            if base in sys.stdlib_module_names:
                continue
            if base not in installed:
                missing.append({
                    'type': 'MissingDependency',
                    'line': 0,
                    'message': f"Module '{base}' is not installed"
                })

        # Remove any old MissingDependency errors, then add the new ones
        script.errors = [e for e in script.errors if e['type'] != 'MissingDependency'] + missing

        # Schedule the UI updates back on the main thread
        self.root.after(0, self.update_error_tree)
        self.root.after(0, self.update_deps_panel)

    def get_installed_packages(self):
        """Get list of installed packages"""
        try:
            if shutil.which("uv"):
                result = subprocess.run(["uv", "pip", "list"], capture_output=True, text=True)
            else:
                result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)

            packages = set()
            for line in result.stdout.splitlines()[2:]:  # Skip header
                if line.strip():
                    package = line.split()[0].lower()
                    packages.add(package)
            return packages
        except:
            return set()

    def install_dependencies(self):
        """Install missing dependencies"""
        if not self.active_script_id:
            return

        script = self.scripts[self.active_script_id]
        missing = []

        for error in script.errors:
            if error['type'] == 'MissingDependency':
                module = error['message'].split("'")[1]
                missing.append(module)

        if missing:
            self.install_packages(missing)
        else:
            messagebox.showinfo("Dependencies", "No missing dependencies found!")

    def install_packages(self, packages):
        """Install packages using uv or pip"""
        if shutil.which("uv"):
            cmd = ["uv", "pip", "install"] + packages
        else:
            cmd = [sys.executable, "-m", "pip", "install"] + packages

        self.add_output(f"Installing: {' '.join(packages)}\n", "info")

        def install():
            try:
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                stdout, stderr = process.communicate()

                self.output_queue.put((self.active_script_id, stdout, "stdout"))
                if stderr:
                    self.output_queue.put((self.active_script_id, stderr, "stderr"))

                if process.returncode == 0:
                    self.output_queue.put((self.active_script_id, "\nPackages installed successfully!\n", "info"))
                    self.root.after(0, self.analyze_script)
                else:
                    self.output_queue.put((self.active_script_id, "\nInstallation failed!\n", "stderr"))
            except Exception as e:
                self.output_queue.put((self.active_script_id, f"Error: {e}\n", "stderr"))

        threading.Thread(target=install, daemon=True).start()

    def auto_fix_errors(self):
        """Automatically fix all detected errors"""
        if not self.active_script_id:
            return

        script = self.scripts[self.active_script_id]

        # First, install missing dependencies
        missing_deps = [e for e in script.errors if e['type'] == 'MissingDependency']
        if missing_deps:
            self.install_dependencies()

        # Then, fix syntax errors using AI
        syntax_errors = [e for e in script.errors if e['type'] == 'SyntaxError']
        if syntax_errors and self.openai_client:
            self.fix_syntax_errors_with_ai(script, syntax_errors)

    def fix_syntax_errors_with_ai(self, script, errors):
        """Use AI to fix syntax errors"""
        error_desc = "\n".join([f"Line {e['line']}: {e['message']}" for e in errors])

        prompt = f"""Fix the following Python code that has syntax errors:

{script.content}

Errors:
{error_desc}

Return only the corrected code without any explanation."""

        try:
            response = self.openai_client.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )

            fixed_code = response.choices[0].message.content
            script.content = fixed_code
            self.editor.delete(1.0, tk.END)
            self.editor.insert(1.0, fixed_code)
            self.analyze_script()
            self.add_output("Syntax errors fixed with AI!\n", "info")

        except Exception as e:
            self.add_output(f"AI fix failed: {e}\n", "stderr")

    def auto_fix_runtime_error(self, script):
        """Automatically fix runtime errors"""
        if not script.error_output or not self.openai_client:
            return

        self.add_output("\n--- Auto-healing runtime error ---\n", "info")

        prompt = f"""Fix this Python script that produced a runtime error:

Code:
{script.content}

Error output:
{script.error_output}

Return only the corrected code that fixes the error."""

        try:
            response = self.openai_client.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )

            fixed_code = response.choices[0].message.content
            script.content = fixed_code

            if script.id == self.active_script_id:
                self.editor.delete(1.0, tk.END)
                self.editor.insert(1.0, fixed_code)

            self.add_output("Applied auto-heal fix. Running again...\n", "info")

            # Save and run again
            if script.file_path:
                script.file_path.write_text(fixed_code, encoding='utf-8')
            self.run_script()

        except Exception as e:
            self.add_output(f"Auto-heal failed: {e}\n", "stderr")

    def toggle_auto_heal(self):
        """Toggle auto-heal mode"""
        self.auto_heal_active = not self.auto_heal_active
        status = "enabled" if self.auto_heal_active else "disabled"
        self.add_output(f"Auto-heal {status}\n", "info")
        self.update_status(f"Auto-heal {status}")

    def get_ai_suggestions(self):
        """Get AI suggestions for improving code"""
        if not self.active_script_id or not self.openai_client:
            messagebox.showwarning("AI Suggestions", "OpenAI API key required!")
            return

        script = self.scripts[self.active_script_id]

        prompt = f"""Analyze this Python code and suggest improvements:

{script.content}

Provide:
1. Code quality improvements
2. Performance optimizations
3. Best practices
4. Potential bugs

Format as a bulleted list."""

        try:
            response = self.openai_client.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            suggestions = response.choices[0].message.content

            # Show in dialog
            dialog = tk.Toplevel(self.root)
            dialog.title("AI Suggestions")
            dialog.geometry("600x400")

            text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, font=('Arial', 10))
            text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            text.insert(1.0, suggestions)
            text.config(state='disabled')

        except Exception as e:
            messagebox.showerror("AI Error", f"Failed to get suggestions: {e}")

    def use_copilot(self):
        """Use GitHub Copilot for suggestions"""
        if not self.copilot_available:
            messagebox.showwarning("Copilot", "GitHub Copilot CLI not found!\nInstall with: gh extension install github/gh-copilot")
            return

        if not self.active_script_id:
            return

        script = self.scripts[self.active_script_id]

        # Save current code to temp file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        temp_file.write(script.content)
        temp_file.close()

        try:
            # Get current cursor position
            cursor_pos = self.editor.index(tk.INSERT)
            line = int(cursor_pos.split('.')[0])

            # Use Copilot CLI
            result = subprocess.run(
                ['gh', 'copilot', 'suggest', '--target', 'code', '--file', temp_file.name, '--line', str(line)],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                suggestion = result.stdout

                # Show suggestion dialog
                dialog = tk.Toplevel(self.root)
                dialog.title("GitHub Copilot Suggestion")
                dialog.geometry("600x400")

                text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, font=('Consolas', 11))
                text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                text.insert(1.0, suggestion)

                btn_frame = ttk.Frame(dialog)
                btn_frame.pack(fill=tk.X, pady=10)

                def apply_suggestion():
                    # Insert suggestion at cursor
                    self.editor.insert(cursor_pos, suggestion)
                    dialog.destroy()

                ttk.Button(btn_frame, text="Apply", command=apply_suggestion).pack(side=tk.RIGHT, padx=10)
                ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)

            else:
                self.add_output(f"Copilot error: {result.stderr}\n", "stderr")

        finally:
            os.unlink(temp_file.name)

    def open_in_vscode(self):
        """Open current script in VS Code"""
        if not self.active_script_id:
            return

        script = self.scripts[self.active_script_id]

        # Save script first
        if not script.file_path:
            self.save_script_as()

        if script.file_path and script.file_path.exists():
            try:
                if sys.platform == "win32":
                    os.startfile(script.file_path)
                elif sys.platform == "darwin":
                    subprocess.run(["open", script.file_path])
                else:
                    # Try VS Code first, then default editor
                    if shutil.which("code"):
                        subprocess.run(["code", script.file_path])
                    else:
                        subprocess.run(["xdg-open", script.file_path])
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open in editor: {e}")

    def execute_terminal_command(self, event=None):
        """Execute command in terminal"""
        command = self.terminal_entry.get()
        if not command:
            return

        self.terminal_entry.delete(0, tk.END)
        self.terminal_output.insert(tk.END, f"$ {command}\n")

        # Determine terminal
        terminal = self.terminal_var.get()

        if terminal == "auto":
            if sys.platform == "win32":
                terminal = "gitbash" if shutil.which("git") else "cmd"
            else:
                terminal = "bash"

        # Execute command
        try:
            if terminal == "gitbash" and sys.platform == "win32":
                # Use Git Bash
                git_path = shutil.which("git")
                if git_path:
                    bash_path = Path(git_path).parent.parent / "bin" / "bash.exe"
                    process = subprocess.Popen(
                        [str(bash_path), "-c", command],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                else:
                    raise Exception("Git Bash not found")

            elif terminal == "wsl" and sys.platform == "win32":
                # Use WSL
                process = subprocess.Popen(
                    ["wsl", "bash", "-c", command],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

            else:
                # Default shell
                shell = True if sys.platform == "win32" else False
                process = subprocess.Popen(
                    command,
                    shell=shell,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

            # Monitor output
            def monitor():
                stdout, stderr = process.communicate()
                self.terminal_output.insert(tk.END, stdout)
                if stderr:
                    self.terminal_output.insert(tk.END, stderr)
                self.terminal_output.see(tk.END)

            threading.Thread(target=monitor, daemon=True).start()

        except Exception as e:
            self.terminal_output.insert(tk.END, f"Error: {e}\n")
            self.terminal_output.see(tk.END)

    def update_network_viz(self):
        """Update neural network visualization"""
        if not self.scripts:
            return

        # Create graph
        G = nx.DiGraph()

        # Add nodes
        for script_id, script in self.scripts.items():
            color = {
                'idle': '#gray',
                'running': '#yellow',
                'success': '#green',
                'error': '#red'
            }.get(script.status, '#gray')

            G.add_node(script.name, color=color, script_id=script_id)

        # Add edges based on imports/dependencies
        for script_id, script in self.scripts.items():
            for other_id, other in self.scripts.items():
                if script_id != other_id:
                    # Check if script imports from other
                    for imp in script.imports:
                        if imp == other.name.replace('.py', ''):
                            G.add_edge(other.name, script.name)

        # Clear and redraw
        self.ax.clear()

        if G.nodes():
            pos = nx.spring_layout(G, k=2, iterations=50)

            # Draw nodes
            colors = [G.nodes[node]['color'] for node in G.nodes()]
            nx.draw_networkx_nodes(G, pos, ax=self.ax, node_color=colors,
                                 node_size=3000, alpha=0.8)

            # Draw edges
            nx.draw_networkx_edges(G, pos, ax=self.ax, edge_color='white',
                                 arrows=True, alpha=0.5, width=2)

            # Draw labels
            nx.draw_networkx_labels(G, pos, ax=self.ax, font_color='white')

        self.ax.set_facecolor('black')
        self.ax.axis('off')
        self.canvas.draw()

    def change_network_layout(self):
        """Change network visualization layout"""
        # Cycle through different layouts
        layouts = ['spring', 'circular', 'shell', 'kamada_kawai']
        # Implementation for layout cycling
        self.update_network_viz()

    # UI Updates

    def add_script_to_tree(self, script):
        """Add script to tree view"""
        self.script_tree.insert('', 'end', iid=script.id, text=script.name,
                               values=(script.status,))

    def update_script_tree(self):
        """Update script tree view"""
        for script_id, script in self.scripts.items():
            if self.script_tree.exists(script_id):
                self.script_tree.item(script_id, text=script.name, values=(script.status,))

    def select_script(self, script_id):
        """Select a script"""
        self.active_script_id = script_id
        script = self.scripts[script_id]

        # Update editor
        self.editor.delete(1.0, tk.END)
        self.editor.insert(1.0, script.content)

        # Update selection in tree
        self.script_tree.selection_set(script_id)
        self.script_tree.focus(script_id)

        # Update UI
        self.update_line_numbers()
        self.script_status_label.config(text=f"Script: {script.name}")

        # Auto-analyze if enabled
        if self.auto_analyze_var.get():
            self.analyze_script()

    def on_script_select(self, event):
        """Handle script selection"""
        selection = self.script_tree.selection()
        if selection:
            self.select_script(selection[0])

    def on_editor_change(self, event=None):
        """Handle editor content change"""
        if self.active_script_id:
            script = self.scripts[self.active_script_id]
            script.content = self.editor.get(1.0, tk.END)

            # Update line numbers
            self.update_line_numbers()

            # Auto-save if enabled
            if self.auto_save_var.get() and script.file_path:
                self.save_script()

            # Auto-analyze if enabled
            if self.auto_analyze_var.get():
                # Debounce analysis
                if hasattr(self, '_analyze_timer'):
                    self.root.after_cancel(self._analyze_timer)
                self._analyze_timer = self.root.after(1000, self.analyze_script)

    def update_line_numbers(self):
        """Update line numbers"""
        self.line_numbers.config(state='normal')
        self.line_numbers.delete(1.0, tk.END)

        lines = self.editor.get(1.0, tk.END).count('\n')
        line_nums = '\n'.join(str(i) for i in range(1, lines + 1))
        self.line_numbers.insert(1.0, line_nums)
        self.line_numbers.config(state='disabled')

    def update_error_tree(self):
        """Update error tree view"""
        self.error_tree.delete(*self.error_tree.get_children())

        for script_id, script in self.scripts.items():
            if script.errors:
                parent = self.error_tree.insert('', 'end', text=script.name)
                for error in script.errors:
                    self.error_tree.insert(parent, 'end',
                                         values=(error['type'], error['line'], error['message']))

    def update_deps_panel(self):
        """Update dependencies panel"""
        if not self.active_script_id:
            return

        script = self.scripts[self.active_script_id]

        self.deps_text.delete(1.0, tk.END)
        self.deps_text.insert(tk.END, "Imports:\n")
        for imp in sorted(set(script.imports)):
            self.deps_text.insert(tk.END, f"  - {imp}\n")

        missing = [e['message'].split("'")[1] for e in script.errors
                  if e['type'] == 'MissingDependency']
        if missing:
            self.deps_text.insert(tk.END, "\nMissing:\n")
            for mod in missing:
                self.deps_text.insert(tk.END, f"  - {mod}\n")

    def jump_to_error(self, event):
        """Jump to error location in editor"""
        selection = self.error_tree.selection()
        if selection:
            item = self.error_tree.item(selection[0])
            values = item['values']
            if values and len(values) >= 2:
                line = values[1]
                if line:
                    self.editor.mark_set(tk.INSERT, f"{line}.0")
                    self.editor.see(f"{line}.0")
                    self.editor_notebook.select(0)  # Switch to code tab

    def add_output(self, text, tag="stdout"):
        """Add text to output"""
        self.output_queue.put((self.active_script_id, text, tag))

    def quick_install(self):
        """Quick install from dependencies panel"""
        package = self.deps_entry.get()
        if package:
            self.install_packages([package])
            self.deps_entry.delete(0, tk.END)

    def update_status(self, message):
        """Update status bar"""
        self.status_label.config(text=message)

    # Background processes

    def start_output_processor(self):
        """Start output queue processor"""
        def process():
            while True:
                try:
                    script_id, text, tag = self.output_queue.get(timeout=0.1)
                    if script_id == self.active_script_id:
                        self.output_text.insert(tk.END, text, tag)
                        self.output_text.see(tk.END)
                except queue.Empty:
                    pass
                except Exception as e:
                    print(f"Output processor error: {e}")

        threading.Thread(target=process, daemon=True).start()

    def start_dependency_analyzer(self):
        """Start background dependency analyzer"""
        def analyze():
            while True:
                try:
                    time.sleep(5)
                    # Periodically update network visualization
                    self.root.after(0, self.update_network_viz)
                except Exception as e:
                    print(f"Dependency analyzer error: {e}")

        threading.Thread(target=analyze, daemon=True).start()

    # Configuration

    def load_config(self):
        """Load configuration"""
        if CONFIG_FILE.exists():
            try:
                return json.loads(CONFIG_FILE.read_text())
            except:
                pass
        return {}

    def save_config(self):
        """Save configuration"""
        CONFIG_FILE.write_text(json.dumps(self.config, indent=2))

    def update_config(self):
        """Update configuration from UI"""
        self.config['auto_save'] = self.auto_save_var.get()
        self.config['auto_analyze'] = self.auto_analyze_var.get()
        self.config['theme'] = self.theme_var.get()
        self.save_config()

    def save_api_key(self):
        """Save OpenAI API key"""
        key = self.api_key_var.get()
        if key:
            self.config['openai_api_key'] = key
            self.save_config()
            self.setup_ai_clients()
            messagebox.showinfo("API Key", "OpenAI API key saved!")

    def setup_styles(self):
        """Setup UI styles"""
        style = ttk.Style()

        if self.config.get('theme', 'dark') == 'dark':
            # Dark theme
            self.root.configure(bg='#1e1e1e')
            style.configure('TLabel', background='#1e1e1e', foreground='white')
            style.configure('TFrame', background='#1e1e1e')
            self.editor.configure(bg='#1e1e1e', fg='white', insertbackground='white',
                                selectbackground='#264f78', selectforeground='white')
            self.output_text.configure(bg='#1e1e1e', fg='white', insertbackground='white')
            self.line_numbers.configure(bg='#2d2d2d', fg='#858585')
        else:
            # Light theme
            self.apply_theme()

    def apply_theme(self):
        """Apply selected theme"""
        self.config['theme'] = self.theme_var.get()
        self.save_config()
        # Restart to apply theme
        messagebox.showinfo("Theme", "Please restart the IDE to apply the theme.")

    def bind_shortcuts(self):
        """Bind keyboard shortcuts"""
        self.root.bind('<Control-n>', lambda e: self.new_script())
        self.root.bind('<Control-o>', lambda e: self.open_script())
        self.root.bind('<Control-s>', lambda e: self.save_script())
        self.root.bind('<Control-Shift-S>', lambda e: self.save_script_as())
        self.root.bind('<F5>', lambda e: self.run_script())
        self.root.bind('<Shift-F5>', lambda e: self.stop_script())
        self.root.bind('<Control-i>', lambda e: self.install_dependencies())
        self.root.bind('<Control-Shift-I>', lambda e: self.get_ai_suggestions())

    def show_welcome(self):
        """Show welcome message"""
        welcome = f"""Welcome to {APP_NAME} v{APP_VERSION}!

🚀 Quick Start:
• Ctrl+N: New Script
• Ctrl+O: Open Script
• F5: Run Script
• Ctrl+I: Install Dependencies

🤖 AI Features:
• Auto-fix errors with AI
• Get code suggestions
• Auto-heal runtime errors

💡 Tips:
• Double-click a script to run it
• Enable auto-heal for continuous error fixing
• Use GitHub Copilot for advanced suggestions
"""
        self.output_text.insert(1.0, welcome, "info")

# Main entry point
def main():
    """Main application entry point"""
    root = tk.Tk()
    app = NeuralNexusIDE(root)

    # Handle window close
    def on_closing():
        # Stop all running scripts
        for script in app.scripts.values():
            if script.process and script.process.poll() is None:
                script.process.terminate()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    # Install required packages if missing
    required = ['psutil', 'networkx', 'matplotlib']
    missing = []

    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Installing required packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)

    main()

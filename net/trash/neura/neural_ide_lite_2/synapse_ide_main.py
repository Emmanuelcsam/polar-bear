#!/usr/bin/env python3
"""
Synapse IDE: An advanced, script-centric IDE for building interconnected systems.
This application provides a powerful, multi-tabbed editor environment with integrated
tools for code analysis, performance monitoring, and AI-assisted development.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, font, simpledialog
from ttkthemes import ThemedTk
import os
import sys
import subprocess
import threading
import queue
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

# Import the backend modules
from synapse_config import ConfigManager, ProjectConfig, ScriptConfig
from synapse_tools import CodeAnalyzer, AISuggestor, AutoHealer, save_api_key

# --- Constants ---
APP_NAME = "Synapse IDE"
APP_VERSION = "1.1" # Incremented version

# --- Main Application Class ---

class SynapseIDE(tk.Frame):
    """The main application window and controller."""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.pack(fill=tk.BOTH, expand=True)

        self.config_manager = ConfigManager()
        self.current_project: Optional[ProjectConfig] = None
        self.tabs: Dict[str, 'ScriptTab'] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        
        self.setup_ui()
        self.load_or_create_project()

    def setup_ui(self):
        """Initializes the main user interface."""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        BG_COLOR, FG_COLOR, INACTIVE_BG, ACCENT_COLOR = "#2c2f33", "#ffffff", "#23272a", "#007acc"
        self.parent.configure(bg=BG_COLOR)
        # (style configurations remain the same)
        self.style.configure('.', background=BG_COLOR, foreground=FG_COLOR); self.style.configure('TFrame', background=BG_COLOR); self.style.configure('TLabel', background=BG_COLOR, foreground=FG_COLOR, font=('Segoe UI', 10)); self.style.configure('TButton', background="#40444b", foreground=FG_COLOR, borderwidth=1, padding=5); self.style.map('TButton', background=[('active', ACCENT_COLOR)]); self.style.configure('TNotebook', background=BG_COLOR, borderwidth=0); self.style.configure('TNotebook.Tab', background=INACTIVE_BG, foreground='lightgrey', padding=[10, 5], borderwidth=0); self.style.map('TNotebook.Tab', background=[('selected', ACCENT_COLOR)], foreground=[('selected', 'white')])

        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        project_frame = ttk.Frame(main_pane, width=250)
        main_pane.add(project_frame, weight=0)
        self.editor_notebook = ttk.Notebook(main_pane)
        main_pane.add(self.editor_notebook, weight=1)
        
        self.build_menubar()

        self.project_label = ttk.Label(project_frame, text="No Project Loaded", font=('Segoe UI', 12, 'bold'), foreground=ACCENT_COLOR)
        self.project_label.pack(pady=10, padx=5, fill=tk.X)
        self.script_list = tk.Listbox(project_frame, bg=INACTIVE_BG, fg="white", selectbackground=ACCENT_COLOR, borderwidth=0, highlightthickness=0)
        self.script_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.script_list.bind("<Double-1>", self.on_script_double_click)

    def build_menubar(self):
        """Creates the main application menu, including new Project and Run menus."""
        menubar = tk.Menu(self.parent, bg="#2c2f33", fg="white", activebackground="#007acc", relief=tk.FLAT)
        self.parent.config(menu=menubar)

        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0, bg="#2c2f33", fg="white")
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Project...", command=self.create_new_project)
        file_menu.add_command(label="Open Project...", command=self.open_project_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Save Project", command=self.save_project)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close)

        # --- NEW: Project Menu ---
        project_menu = tk.Menu(menubar, tearoff=0, bg="#2c2f33", fg="white")
        menubar.add_cascade(label="Project", menu=project_menu)
        project_menu.add_command(label="Add New Script...", command=self.add_new_script)
        project_menu.add_command(label="Validate Dependencies", command=self.validate_project_dependencies)

        # --- NEW: Run Menu ---
        run_menu = tk.Menu(menubar, tearoff=0, bg="#2c2f33", fg="white")
        menubar.add_cascade(label="Run", menu=run_menu)
        run_menu.add_command(label="Run Project (respecting dependencies)", command=self.run_project_orchestrated)
        run_menu.add_separator()
        run_menu.add_command(label="Stop All Scripts", command=self.stop_all_scripts)

        # Edit Menu
        edit_menu = tk.Menu(menubar, tearoff=0, bg="#2c2f33", fg="white")
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Settings...", command=self.show_settings)

    # --- Project & Script Management ---
    
    def load_or_create_project(self):
        projects = self.config_manager.list_projects()
        if projects: self.load_project(projects[0])
        else: self.create_new_project()

    def create_new_project(self):
        name = simpledialog.askstring("New Project", "Enter a name for your new project:", parent=self.parent)
        if name:
            new_project = ProjectConfig(name=name)
            default_script = ScriptConfig(id="script_1", name="main_script", file_path=f"{name.lower()}_main.py")
            new_project.scripts.append(default_script)
            self.config_manager.save_project(new_project)
            self.load_project(name)

    def open_project_dialog(self):
        projects = self.config_manager.list_projects()
        if not projects: messagebox.showinfo("No Projects", "No projects found."); return
        name = simpledialog.askstring("Open Project", f"Available projects:\n{', '.join(projects)}\n\nEnter project name:", parent=self.parent)
        if name and name.title() in [p.title() for p in projects]: self.load_project(name)
        elif name: messagebox.showerror("Error", f"Project '{name}' not found.")

    def load_project(self, name: str):
        try:
            self.current_project = self.config_manager.load_project(name)
            self.project_label.config(text=self.current_project.name)
            for tab_id in list(self.tabs.keys()):
                self.editor_notebook.forget(self.tabs[tab_id]); self.tabs[tab_id].destroy()
            self.tabs.clear(); self.script_list.delete(0, tk.END)
            for script in self.current_project.scripts:
                self.script_list.insert(tk.END, f"📜 {script.name}")
                self.create_tab_for_script(script)
        except Exception as e: messagebox.showerror("Load Error", f"Failed to load project: {e}"); import traceback; traceback.print_exc()

    def save_project(self):
        if not self.current_project: return
        for script_id, tab in self.tabs.items():
            for i, s in enumerate(self.current_project.scripts):
                if s.id == script_id: self.current_project.scripts[i] = tab.get_config(); break
        self.config_manager.save_project(self.current_project)
        messagebox.showinfo("Success", f"Project '{self.current_project.name}' saved.")

    def add_new_script(self):
        """Handles the UI and logic for adding a new script to the project."""
        if not self.current_project: messagebox.showwarning("No Project", "Please open or create a project first."); return
        name = simpledialog.askstring("New Script", "Enter a name for the new script:", parent=self.parent)
        if name:
            script_id = f"script_{int(time.time())}"
            file_path = f"{name.lower().replace(' ', '_')}.py"
            new_script_config = ScriptConfig(id=script_id, name=name, file_path=file_path)
            self.current_project.scripts.append(new_script_config)
            self.script_list.insert(tk.END, f"📜 {name}")
            self.create_tab_for_script(new_script_config)
            self.save_project()

    def validate_project_dependencies(self):
        if not self.current_project: return
        errors = self.config_manager.validate_project(self.current_project)
        if errors: messagebox.showwarning("Dependency Validation", "Issues found:\n\n" + "\n".join(errors))
        else: messagebox.showinfo("Dependency Validation", "✅ Project dependencies are valid!")

    def on_script_double_click(self, event):
        selected_indices = self.script_list.curselection()
        if not selected_indices: return
        script_config = self.current_project.scripts[selected_indices[0]]
        if script_config.id in self.tabs: self.editor_notebook.select(self.tabs[script_config.id])
        else: self.create_tab_for_script(script_config)

    def create_tab_for_script(self, script_config: ScriptConfig):
        if script_config.id in self.tabs: self.editor_notebook.select(self.tabs[script_config.id]); return
        tab = ScriptTab(self.editor_notebook, script_config, self)
        self.tabs[script_config.id] = tab
        self.editor_notebook.add(tab, text=script_config.name)
        self.editor_notebook.select(tab)
        
    def on_tab_change(self, event): pass

    # --- Run and Orchestration ---

    def run_project_orchestrated(self):
        """Runs all scripts in the project in the correct dependency order."""
        if not self.current_project: return
        try:
            execution_order = self.config_manager.get_execution_order(self.current_project)
            threading.Thread(target=self._run_all_worker, args=(execution_order,), daemon=True).start()
        except ValueError as e:
            messagebox.showerror("Dependency Error", str(e))

    def _run_all_worker(self, execution_order: List[str]):
        """Worker thread to run scripts sequentially."""
        print(f"Running project. Execution order: {execution_order}")
        for script_id in execution_order:
            if script_id in self.tabs:
                tab = self.tabs[script_id]
                print(f"--- Running script: {tab.config.name} ---")
                # We can't call UI methods directly from a thread, so we schedule them
                self.after(0, lambda t=tab: t.run_script())
                # Wait for the process to finish before starting the next one
                while script_id in self.processes and self.processes[script_id].poll() is None:
                    time.sleep(0.5)
                # Check for errors before proceeding
                if script_id in self.processes and self.processes[script_id].returncode != 0:
                    print(f"!!! Script {tab.config.name} failed. Halting project execution. !!!")
                    messagebox.showerror("Execution Halted", f"Script '{tab.config.name}' failed with a non-zero exit code. See console for details.", parent=self)
                    return
        print("--- Project execution finished ---")
        messagebox.showinfo("Execution Complete", "All scripts in the project have finished running.", parent=self)

    def stop_all_scripts(self):
        """Stops all currently running script processes."""
        for proc in self.processes.values():
            if proc.poll() is None: proc.terminate()
        self.processes.clear()
        messagebox.showinfo("Stopped", "All running script processes have been terminated.")

    # --- Other Functions ---
    def show_settings(self):
        key = simpledialog.askstring("Settings", "Enter your OpenAI API Key (optional):", parent=self.parent)
        if key: save_api_key(key); messagebox.showinfo("Success", "API Key saved.")

    def on_close(self):
        if messagebox.askokcancel("Quit", "Do you want to exit Synapse IDE?"):
            self.stop_all_scripts()
            self.parent.destroy()

# --- ScriptTab Class ---

class ScriptTab(tk.Frame):
    def __init__(self, parent_notebook, script_config: ScriptConfig, app_controller: 'SynapseIDE'):
        super().__init__(parent_notebook)
        self.notebook = parent_notebook
        self.config = script_config
        self.controller = app_controller
        self.output_queue = queue.Queue()
        self.autonomous_thread = None

        self.setup_ui()
        self.load_script_content()
        self.after(100, self.process_output_queue)

    def get_config(self) -> ScriptConfig:
        self.config.autonomous_mode = self.autonomous_var.get()
        return self.config

    def setup_ui(self):
        self.pack(fill=tk.BOTH, expand=True)
        main_pane = ttk.PanedWindow(self, orient=tk.VERTICAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        editor_frame = ttk.Frame(main_pane)
        main_pane.add(editor_frame, weight=3)
        
        toolbar = ttk.Frame(editor_frame)
        toolbar.pack(fill=tk.X, pady=5, padx=5)
        ttk.Button(toolbar, text="▶ Run Script", command=self.run_script).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="⏹ Stop", command=self.stop_script).pack(side=tk.LEFT, padx=5)

        # --- NEW: Autonomous mode checkbox ---
        self.autonomous_var = tk.BooleanVar(value=self.config.autonomous_mode)
        ttk.Checkbutton(toolbar, text="Autonomous Mode (Auto-Run & Self-Heal)", variable=self.autonomous_var).pack(side=tk.RIGHT, padx=10)

        self.editor = scrolledtext.ScrolledText(editor_frame, wrap=tk.WORD, font=("Consolas", 12), undo=True, bg="#1e1f22", fg="white", insertbackground="white")
        self.editor.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        info_notebook = ttk.Notebook(main_pane)
        main_pane.add(info_notebook, weight=1)
        console_frame = ttk.Frame(info_notebook); info_notebook.add(console_frame, text="Console")
        self.console = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD, font=("Consolas", 10), bg="#1e1f22", fg="lightgrey")
        self.console.pack(fill=tk.BOTH, expand=True); self.console.tag_config("stderr", foreground="#ff6b6b"); self.console.tag_config("stdout", foreground="lightgrey"); self.console.tag_config("info", foreground="#00aaff")
        self.analysis_frame = ttk.Frame(info_notebook); info_notebook.add(self.analysis_frame, text="Analysis & Issues")
        self.ai_frame = ttk.Frame(info_notebook); info_notebook.add(self.ai_frame, text="AI Assist ✨")
        self.build_ai_assist_tab()

    def build_ai_assist_tab(self):
        """Populates the AI Assist tab, including the new Copilot button."""
        ttk.Button(self.ai_frame, text="Analyze Code (Pylint)", command=self.analyze_code).pack(pady=10, padx=10, fill=tk.X)
        ttk.Button(self.ai_frame, text="Get OpenAI Suggestion", command=lambda: self.get_ai_suggestion(use_copilot=False)).pack(pady=5, padx=10, fill=tk.X)
        # --- NEW: Copilot Button ---
        ttk.Button(self.ai_frame, text="Get GitHub Copilot Suggestion", command=lambda: self.get_ai_suggestion(use_copilot=True)).pack(pady=5, padx=10, fill=tk.X)
        ttk.Button(self.ai_frame, text="Attempt to Auto-Heal Runtime Error", command=self.auto_heal_error).pack(pady=5, padx=10, fill=tk.X)

    def load_script_content(self):
        try:
            if not self.config.file_path: self.config.file_path = f"{self.config.name.lower().replace(' ', '_')}.py"
            p = Path(self.config.file_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists(): p.write_text(f'# Script: {self.config.name}\n\nprint("Hello from {self.config.name}")\n', encoding='utf-8')
            self.editor.delete("1.0", tk.END); self.editor.insert(tk.END, p.read_text(encoding='utf-8'))
        except Exception as e: self.console.insert(tk.END, f"Error loading script file: {e}\n", "stderr")

    def save_script_content(self):
        if self.config.file_path:
            try: self.editor.get("1.0", tk.END)
            except Exception as e: messagebox.showerror("Save Error", f"Could not save file {self.config.file_path}:\n{e}")

    # --- Script Execution and Autonomous Mode ---

    def run_script(self):
        """Decides whether to run normally or start the autonomous loop."""
        if self.autonomous_var.get():
            self.start_autonomous_loop()
        else:
            self.run_script_once()

    def run_script_once(self):
        """Runs the script a single time as a subprocess."""
        if self.config.id in self.controller.processes and self.controller.processes[self.config.id].poll() is None:
            messagebox.showwarning("Already Running", "This script is already running."); return
        self.save_script_content(); self.console.delete("1.0", tk.END)
        try:
            process = subprocess.Popen([sys.executable, "-u", self.config.file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
            self.controller.processes[self.config.id] = process
            threading.Thread(target=self.monitor_pipe, args=(process.stdout, "stdout"), daemon=True).start()
            threading.Thread(target=self.monitor_pipe, args=(process.stderr, "stderr"), daemon=True).start()
        except Exception as e: self.console.insert(tk.END, f"Failed to start script: {e}\n", "stderr")

    def run_script_and_wait(self) -> (str, str):
        """Runs the script synchronously and returns its output and error streams."""
        self.save_script_content()
        try:
            result = subprocess.run([sys.executable, self.config.file_path], capture_output=True, text=True, encoding='utf-8', errors='replace')
            return result.stdout, result.stderr
        except Exception as e:
            return "", f"Failed to execute script for autonomous mode: {e}"

    def start_autonomous_loop(self):
        """Starts the self-healing autonomous loop in a separate thread."""
        if self.autonomous_thread and self.autonomous_thread.is_alive():
            messagebox.showwarning("Already Running", "Autonomous mode is already active for this script."); return
        self.console.delete("1.0", tk.END)
        self.console_insert("--- Starting Autonomous Mode ---\n", "info")
        self.autonomous_thread = threading.Thread(target=self._autonomous_worker, daemon=True)
        self.autonomous_thread.start()

    def _autonomous_worker(self):
        """The worker thread that runs the self-healing loop."""
        max_attempts = 5
        for attempt in range(max_attempts):
            if not self.autonomous_var.get():
                self.console_insert("Autonomous mode halted by user.\n", "info"); return

            self.console_insert(f"\n[Attempt {attempt + 1}/{max_attempts}] Running script...\n", "info")
            stdout, stderr = self.run_script_and_wait()

            if stdout: self.console_insert(stdout, "stdout")
            if stderr: self.console_insert(stderr, "stderr")

            if stderr:
                self.console_insert("Error detected. Attempting self-heal...\n", "info")
                healer = AutoHealer(self.config.file_path)
                if not healer.is_ready():
                    self.console_insert("Cannot heal: OpenAI key not configured.\n", "stderr"); break
                
                code = self.editor.get("1.0", tk.END)
                context = healer.research_error_online(stderr)
                fix = healer.propose_fix(code, stderr, context)

                if "Auto-Heal Error" in fix or "AI could not generate a fix" in fix:
                    self.console_insert(f"AI failed to generate a fix. Halting.\n{fix}\n", "stderr"); break
                
                self.console_insert("AI proposed a fix. Applying and re-running...\n", "info")
                self.after(0, self.set_editor_content, fix) # Schedule UI update
                self.save_script_content()
                time.sleep(1) # Give a moment before the next run
            else:
                self.console_insert("--- Script ran successfully. Autonomous mode complete. ---\n", "info"); break
        else:
            self.console_insert(f"--- Reached max {max_attempts} attempts. Halting autonomous mode. ---\n", "info")
        
        self.after(0, self.autonomous_var.set, False) # Turn off the checkbox in the UI

    def stop_script(self):
        self.autonomous_var.set(False) # Stop autonomous loop if it's running
        if self.config.id in self.controller.processes:
            proc = self.controller.processes[self.config.id]
            if proc.poll() is None:
                proc.terminate()
                self.console.insert(tk.END, "\n--- Script terminated by user ---\n")
            del self.controller.processes[self.config.id]

    def monitor_pipe(self, pipe, tag):
        try:
            for line in iter(pipe.readline, ''): self.output_queue.put((line, tag))
        finally: pipe.close()

    def process_output_queue(self):
        try:
            while not self.output_queue.empty():
                line, tag = self.output_queue.get_nowait()
                self.console.insert(tk.END, line, tag)
                self.console.see(tk.END)
        finally: self.after(100, self.process_output_queue)
    
    # --- UI Updaters ---

    def console_insert(self, text, tag):
        """Thread-safe way to insert text into the console."""
        self.output_queue.put((text, tag))

    def set_editor_content(self, content):
        """Thread-safe way to set the editor's content."""
        self.editor.delete("1.0", tk.END)
        self.editor.insert("1.0", content)

    # --- AI and Analysis Methods ---

    def analyze_code(self):
        self.save_script_content()
        analyzer = CodeAnalyzer(self.editor.get("1.0", tk.END))
        results = analyzer.analyze()
        for widget in self.analysis_frame.winfo_children(): widget.destroy()
        ttk.Label(self.analysis_frame, text=f"Analysis Summary: {results['summary']}").pack(anchor='w', padx=10, pady=5)
        if results['issues']:
            tree = ttk.Treeview(self.analysis_frame, columns=('line', 'code', 'message'), show='headings')
            tree.heading('line', text='Line'); tree.heading('code', text='Code'); tree.heading('message', text='Message')
            tree.column('line', width=50, anchor='center'); tree.column('code', width=80, anchor='center')
            for issue in results['issues']: tree.insert('', tk.END, values=(issue['line'], issue['type_code'], issue['message']))
            tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def get_ai_suggestion(self, use_copilot=False):
        """Gets and displays an AI-powered code suggestion from either OpenAI or Copilot."""
        self.save_script_content()
        code = self.editor.get("1.0", tk.END)
        suggester = AISuggestor()
        
        provider_name = "GitHub Copilot" if use_copilot else "OpenAI"
        self.console_insert(f"Requesting suggestion from {provider_name}...\n", "info")

        if use_copilot:
            # The tools file doesn't have a separate copilot function, so we'll just note it
            # In a real implementation, you would call a different method here.
            suggestion = suggester.get_suggestion(code, []) # Placeholder
            self.console_insert("Note: Copilot integration is a placeholder. Using standard OpenAI suggestion.\n", "info")
        else:
            if not suggester.is_ready(): messagebox.showwarning("AI Not Configured", "Please set your OpenAI API key."); return
            suggestion = suggester.get_suggestion(code, [])

        win = tk.Toplevel(self); win.title(f"{provider_name} Suggestion")
        text_widget = scrolledtext.ScrolledText(win, wrap=tk.WORD, font=("Consolas", 12))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, suggestion)
        ttk.Button(win, text="Apply Suggestion", command=lambda: [self.set_editor_content(suggestion), win.destroy()]).pack(pady=10)

    def auto_heal_error(self):
        self.save_script_content()
        healer = AutoHealer(self.config.file_path)
        if not healer.is_ready(): messagebox.showwarning("AI Not Configured", "Auto-Heal requires an OpenAI API key."); return
        stdout, stderr = healer.diagnose_runtime_error()
        if not stderr: messagebox.showinfo("No Error", "Script ran without any runtime errors."); return
        self.console_insert(f"--- Runtime Error ---\n{stderr}\n--- Starting Auto-Heal ---\n", "stderr")
        context = healer.research_error_online(stderr)
        code = self.editor.get("1.0", tk.END)
        fix = healer.propose_fix(code, stderr, context)
        if messagebox.askyesno("Apply Fix?", f"AI proposes the following fix. Apply it?\n\n---\n{fix[:500]}...\n---"):
            self.set_editor_content(fix); self.save_script_content()
            messagebox.showinfo("Fix Applied", "The fix has been applied. Please try running again.")

# --- Main Execution ---

def main():
    root = ThemedTk(theme="clam")
    root.title(APP_NAME)
    root.geometry("1400x900")
    ide_app = SynapseIDE(root)
    root.protocol("WM_DELETE_WINDOW", ide_app.on_close)
    root.mainloop()

if __name__ == '__main__':
    main()

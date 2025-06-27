#!/usr/bin/env python3
"""
Neural Weaver: A Visual IDE for Building Data Flows
This is the main application file, rewritten from the ground up to provide
an intuitive, block-based, visual environment for creating and running
complex data processing and neural network flows.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog, font
from ttkthemes import ThemedTk
import os
import sys
import subprocess
import threading
import queue
from pathlib import Path
import psutil
from PIL import Image, ImageTk, ImageDraw, ImageFont

# --- Proactive Fix: Added missing imports from the 'typing' module ---
from typing import Optional, List, Dict, Any

# Import rewritten support modules
from weaver_config import ConfigManager, FlowConfig, BlockConfig
from ai_assist import BlockAnalyzer, AISuggestor, save_api_key, load_api_key
from auto_healer import AutoHealer
from weaver_tools import PerformanceProfiler, MessageDebugger, Message

# --- Constants and Configuration ---
APP_NAME = "Neural Weaver"
APP_VERSION = "2.0"
BLOCK_WIDTH = 180
BLOCK_HEIGHT = 80
PIN_RADIUS = 6
GRID_SIZE = 20

# --- Main Application Class ---

class NeuralWeaverApp:
    """
    The main class for the Neural Weaver application. It orchestrates the UI,
    user interactions, and the backend processing of flows.
    """
    def __init__(self, root):
        self.root = root
        self.setup_main_window()

        # Backend Services
        self.config_manager = ConfigManager()
        self.profiler = PerformanceProfiler()
        self.profiler.start_monitoring()
        self.debugger = MessageDebugger()

        # App State
        self.current_flow: Optional[FlowConfig] = None
        self.selected_block_id: Optional[str] = None
        self.process_manager: Dict[str, Any] = {} # Maps block_id to its subprocess and psutil process
        self.output_queue: queue.Queue = queue.Queue() # For thread-safe UI updates from subprocesses
        self.is_running_flow = False
        self._drag_data: Dict[str, Any] = {} # For handling drag-and-drop
        
        # UI Components
        self.build_ui()
        
        # Load last opened flow or show welcome
        self.load_initial_flow()
        
        # Start the UI update loop
        self.root.after(100, self.process_output_queue)

    def setup_main_window(self):
        """Configures the main application window."""
        self.root.title(f"{APP_NAME} - {APP_VERSION}")
        self.root.geometry("1600x900")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # --- Styling ---
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Glassy Dark Theme Colors
        BG_COLOR = "#2c2f33"
        FG_COLOR = "#ffffff"
        INACTIVE_BG_COLOR = "#23272a"
        ACCENT_COLOR = "#7289da"
        BORDER_COLOR = "#4f545c"
        CANVAS_BG = "#1e1f22"
        TEXT_AREA_BG = "#23272a"
        BUTTON_BG = "#40444b"

        # Configure styles for a dark, glassy feel
        self.style.configure('.', background=BG_COLOR, foreground=FG_COLOR, bordercolor=BORDER_COLOR, font=('Segoe UI', 10))
        self.style.configure('TFrame', background=BG_COLOR)
        self.style.configure('TLabel', background=BG_COLOR, foreground=FG_COLOR)
        self.style.configure('TButton', background=BUTTON_BG, foreground=FG_COLOR, borderwidth=1, focusthickness=3, focuscolor=ACCENT_COLOR, padding=5)
        self.style.map('TButton', background=[('active', ACCENT_COLOR)])
        self.style.configure('Treeview', background=TEXT_AREA_BG, foreground=FG_COLOR, fieldbackground=TEXT_AREA_BG, rowheight=25)
        self.style.map('Treeview', background=[('selected', ACCENT_COLOR)])
        self.style.configure('Treeview.Heading', background=BUTTON_BG, font=('Segoe UI', 10, 'bold'))
        self.style.configure('Vertical.TScrollbar', background=BG_COLOR, troughcolor=BUTTON_BG)
        self.style.configure('Inspector.TFrame', background=INACTIVE_BG_COLOR, relief='solid', borderwidth=1)
        self.style.configure('Inspector.TLabel', background=INACTIVE_BG_COLOR, font=('Segoe UI', 11, 'bold'))
        self.style.configure('Header.TLabel', font=('Segoe UI', 12, 'bold'), foreground=ACCENT_COLOR)
        self.style.configure('TPanedwindow', background=BG_COLOR)
        self.style.configure('TNotebook', background=BG_COLOR, tabmargins=[2, 5, 2, 0])
        self.style.configure('TNotebook.Tab', background=BUTTON_BG, foreground='white', padding=[10, 5])
        self.style.map('TNotebook.Tab', background=[('selected', ACCENT_COLOR)], foreground=[('selected', 'white')])
        
        self.root.configure(bg=BG_COLOR)

    def build_ui(self):
        """Constructs the main UI layout with Palette, Canvas, and Inspector."""
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Left Panel: Block Palette ---
        palette_frame = ttk.Frame(main_pane, width=250, style='Inspector.TFrame')
        main_pane.add(palette_frame, weight=0)
        ttk.Label(palette_frame, text="Block Palette", style='Header.TLabel', anchor='center').pack(pady=10, fill='x')

        self.block_palette = ttk.Treeview(palette_frame, show="tree")
        self.block_palette.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.populate_block_palette()
        self.block_palette.bind('<ButtonPress-1>', self.on_palette_drag_start)

        # --- Center Panel: Canvas ---
        canvas_container = ttk.Frame(main_pane)
        main_pane.add(canvas_container, weight=1)
        
        toolbar = ttk.Frame(canvas_container)
        toolbar.pack(fill=tk.X, pady=(0, 5))
        self.build_toolbar(toolbar)

        self.canvas = tk.Canvas(canvas_container, bg="#1e1f22", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Button-3>', self.on_canvas_right_click)

        # --- Right Panel: Inspector ---
        self.inspector_frame = ttk.Frame(main_pane, width=350, style='Inspector.TFrame')
        main_pane.add(self.inspector_frame, weight=0)
        
        self.inspector_notebook = ttk.Notebook(self.inspector_frame)
        self.inspector_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.build_inspector_tabs()
        self.show_inspector_message("Select a block to see its details or right-click the canvas for flow options.")

    def build_toolbar(self, parent):
        """Builds the main toolbar for flow control."""
        ttk.Button(parent, text="▶ Run Flow", command=self.run_flow).pack(side=tk.LEFT, padx=5)
        ttk.Button(parent, text="⏹ Stop Flow", command=self.stop_flow).pack(side=tk.LEFT, padx=5)
        ttk.Separator(parent, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill='y', pady=5)
        ttk.Button(parent, text="💾 Save Flow", command=self.save_current_flow).pack(side=tk.LEFT, padx=5)
        self.flow_title_label = ttk.Label(parent, text="No Flow Loaded", font=('Segoe UI', 12, 'bold'))
        self.flow_title_label.pack(side=tk.RIGHT, padx=20)

    def build_inspector_tabs(self):
        """Creates the tabs within the inspector panel."""
        self.settings_tab = ttk.Frame(self.inspector_notebook)
        self.logs_tab = ttk.Frame(self.inspector_notebook)
        self.perf_tab = ttk.Frame(self.inspector_notebook)
        self.ai_tab = ttk.Frame(self.inspector_notebook)
        
        self.inspector_notebook.add(self.settings_tab, text="Settings")
        self.inspector_notebook.add(self.logs_tab, text="Logs")
        self.inspector_notebook.add(self.perf_tab, text="Performance")
        self.inspector_notebook.add(self.ai_tab, text="AI Assist ✨")

        # Add log text area now so it's always available
        self.log_text = tk.Text(self.logs_tab, wrap=tk.WORD, bg="#23272a", fg="lightgrey", relief=tk.FLAT, borderwidth=0)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.log_text.config(state=tk.DISABLED)

    def populate_block_palette(self):
        """Fills the block palette with predefined block types."""
        data_cat = self.block_palette.insert("", "end", text="Data I/O")
        proc_cat = self.block_palette.insert("", "end", text="Processing")
        nn_cat = self.block_palette.insert("", "end", text="Neural Network")
        
        self.block_palette.insert(data_cat, "end", text="File Input", values=("data_input", "Read data from a file (CSV, TXT)."))
        self.block_palette.insert(data_cat, "end", text="File Output", values=("data_output", "Save data to a file."))
        self.block_palette.insert(proc_cat, "end", text="Custom Python", values=("custom_python", "Write your own Python logic."))
        self.block_palette.insert(nn_cat, "end", text="Dense Layer", values=("nn_layer", "A fully connected neural network layer."))
        
        self.block_palette.item(data_cat, open=True)
        self.block_palette.item(proc_cat, open=True)
        self.block_palette.item(nn_cat, open=True)

    def on_palette_drag_start(self, event):
        """Initiates a drag-and-drop operation from the palette."""
        item_id = self.block_palette.identify_row(event.y)
        if not item_id or not self.current_flow:
            return

        try:
            item_values = self.block_palette.item(item_id, "values")
        except tk.TclError:
            return # Clicked on a category header

        if not item_values:
            return
        
        block_type, description = item_values
        block_name = self.block_palette.item(item_id, "text")

        # Create a semi-transparent temporary window to represent the dragged block
        drag_window = tk.Toplevel(self.root)
        drag_window.overrideredirect(True)
        drag_window.attributes('-alpha', 0.7)
        drag_window.geometry(f"{BLOCK_WIDTH}x{BLOCK_HEIGHT}+{event.x_root}+{event.y_root}")
        
        label_frame = tk.Frame(drag_window, bg="#7289da")
        label_frame.pack(expand=True, fill='both')
        label = tk.Label(label_frame, text=block_name, fg="white", bg="#7289da", font=('Segoe UI', 10, 'bold'))
        label.pack(expand=True, fill='both')
        
        drag_window.lift()

        def on_drag_move(e):
            drag_window.geometry(f"+{e.x_root-BLOCK_WIDTH//2}+{e.y_root-BLOCK_HEIGHT//2}")

        def on_drag_drop(e):
            drag_window.destroy()
            self.root.unbind('<Motion>')
            self.root.unbind('<ButtonRelease-1>')
            x, y = self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)
            self.add_new_block(block_type, block_name, description, {"x": int(x), "y": int(y)})

        self.root.bind('<Motion>', on_drag_move)
        self.root.bind('<ButtonRelease-1>', on_drag_drop)

    def on_canvas_right_click(self, event):
        """Shows a context menu on the canvas."""
        menu = tk.Menu(self.root, tearoff=0, bg="#2c2f33", fg="#ffffff", activebackground="#7289da", relief=tk.FLAT)
        menu.add_command(label="New Flow...", command=self.create_new_flow)
        menu.add_command(label="Open Flow...", command=self.open_flow_dialog)
        
        if self.current_flow:
            menu.add_separator()
            menu.add_command(label="Save Flow", command=self.save_current_flow)
            menu.add_command(label=f"Validate '{self.current_flow.name}'", command=self.validate_current_flow)
        
        menu.add_separator()
        menu.add_command(label="Settings...", command=self.show_settings)
        menu.tk_popup(event.x_root, event.y_root)

    # --- Flow Management ---

    def load_initial_flow(self):
        """Loads the first available flow or creates a default one."""
        flows = self.config_manager.list_flows()
        if flows:
            self.load_flow_by_name(flows[0])
        else:
            self.create_new_flow(from_template="Simple Data Pipeline")

    def create_new_flow(self, from_template=None):
        """Creates a new, empty flow or loads one from a template."""
        if self.is_running_flow:
            messagebox.showwarning("Flow Running", "Cannot create a new flow while one is running.")
            return

        if from_template:
            try:
                self.current_flow = self.config_manager.load_template(from_template)
            except FileNotFoundError:
                messagebox.showerror("Error", f"Template '{from_template}' not found. Creating empty flow.")
                self.current_flow = FlowConfig(name="New Untitled Flow")
        else:
            self.current_flow = FlowConfig(name="New Untitled Flow")
        
        self.redraw_canvas()
        self.update_flow_title()

    def open_flow_dialog(self):
        """Shows a dialog to open an existing flow."""
        flows = self.config_manager.list_flows()
        if not flows:
            messagebox.showinfo("No Flows", "No saved flows found. Create a new one!")
            return

        # Simple dialog to choose a flow
        win = tk.Toplevel(self.root)
        win.title("Open Flow")
        ttk.Label(win, text="Select a flow to open:").pack(pady=10)
        listbox = tk.Listbox(win, bg="#23272a", fg="white", selectbackground="#7289da")
        listbox.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        for flow_name in flows:
            listbox.insert(tk.END, flow_name)
        
        def on_select():
            selected = listbox.get(listbox.curselection())
            self.load_flow_by_name(selected)
            win.destroy()

        ttk.Button(win, text="Open", command=on_select).pack(pady=10)

    def load_flow_by_name(self, name: str):
        """Loads a flow and displays it on the canvas."""
        try:
            self.current_flow = self.config_manager.load_flow(name)
            self.redraw_canvas()
            self.update_flow_title()
            self.show_inspector_message("Flow loaded successfully. Select a block to view details.")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load flow '{name}':\n{e}")

    def save_current_flow(self):
        """Saves the currently active flow."""
        if not self.current_flow:
            messagebox.showwarning("Warning", "No flow is currently loaded.")
            return
        
        try:
            self.config_manager.save_flow(self.current_flow)
            self.update_flow_title() # In case the name was changed
            messagebox.showinfo("Success", f"Flow '{self.current_flow.name}' saved successfully.")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save flow:\n{e}")
    
    def validate_current_flow(self):
        """Validates the current flow for errors."""
        if not self.current_flow:
            return
        errors = ConfigManager.validate_flow(self.current_flow)
        if not errors:
            messagebox.showinfo("Validation", "✅ This flow is valid!")
        else:
            messagebox.showwarning("Validation Issues", "Found issues:\n\n" + "\n".join(errors))

    # --- Block Management ---

    def add_new_block(self, block_type: str, name: str, desc: str, pos: Dict[str, int]):
        """Adds a new block to the current flow."""
        if not self.current_flow:
            return
        
        new_id = f"block_{os.urandom(4).hex()}"
        new_block = BlockConfig(id=new_id, name=name, description=desc, block_type=block_type, position=pos)
        self.current_flow.blocks.append(new_block)
        self.draw_block(new_block)
        self.select_block(new_id)

    def draw_block(self, block: BlockConfig):
        """Draws a single block on the canvas."""
        x, y = block.position['x'], block.position['y']
        tag = f"block_{block.id}"
        
        self.canvas.create_rectangle(x, y, x + BLOCK_WIDTH, y + BLOCK_HEIGHT, fill="#40444b", outline="#7289da", width=2, tags=(tag, "block_body"))
        self.canvas.create_text(x + BLOCK_WIDTH/2, y + 20, text=block.name, fill="white", font=('Segoe UI', 10, 'bold'), tags=tag)
        
        self.canvas.create_oval(x - PIN_RADIUS, y + BLOCK_HEIGHT/2 - PIN_RADIUS, x + PIN_RADIUS, y + BLOCK_HEIGHT/2 + PIN_RADIUS, fill="cyan", tags=(tag, "pin", "input_pin", f"pin_{block.id}_in"))
        self.canvas.create_oval(x + BLOCK_WIDTH - PIN_RADIUS, y + BLOCK_HEIGHT/2 - PIN_RADIUS, x + BLOCK_WIDTH + PIN_RADIUS, y + BLOCK_HEIGHT/2 + PIN_RADIUS, fill="magenta", tags=(tag, "pin", "output_pin", f"pin_{block.id}_out"))

        self.canvas.tag_bind(tag, '<ButtonPress-1>', lambda e, b_id=block.id: self.on_block_press(e, b_id))
        self.canvas.tag_bind(tag, '<B1-Motion>', self.on_block_drag)
        self.canvas.tag_bind(tag, '<ButtonRelease-1>', self.on_block_release)

    def on_block_press(self, event, block_id):
        """Handles when a block is clicked."""
        self.select_block(block_id)
        self._drag_data = {'x': event.x, 'y': event.y, 'id': block_id}
        self.canvas.lift(f"block_{block_id}") # Bring to front

    def on_block_drag(self, event):
        """Handles dragging a block."""
        if not self._drag_data: return
        dx = event.x - self._drag_data['x']
        dy = event.y - self._drag_data['y']
        
        tag = f"block_{self._drag_data['id']}"
        self.canvas.move(tag, dx, dy)
        
        self._drag_data['x'] = event.x
        self._drag_data['y'] = event.y
        self.redraw_connections()

    def on_block_release(self, event):
        """Handles when a block is released after dragging."""
        if not self._drag_data: return
        block_id = self._drag_data['id']
        block = self.get_block_by_id(block_id)
        if block:
            body_items = self.canvas.find_withtag(f"block_{block_id} and block_body")
            if body_items:
                coords = self.canvas.coords(body_items[0])
                block.position['x'] = int(coords[0])
                block.position['y'] = int(coords[1])
        self._drag_data = {}
        self.redraw_canvas()

    def get_block_by_id(self, block_id: str) -> Optional[BlockConfig]:
        """Finds a block in the current flow by its ID."""
        if self.current_flow:
            return next((b for b in self.current_flow.blocks if b.id == block_id), None)
        return None

    def select_block(self, block_id: str):
        """Selects a block and updates the inspector panel."""
        self.selected_block_id = block_id
        self.redraw_canvas() # Redraw to handle selection highlight
        self.update_inspector()

    # --- Canvas & UI Redrawing ---
    
    def redraw_canvas(self):
        """Clears and redraws the entire canvas from the current flow data."""
        self.canvas.delete("all")
        if not self.current_flow:
            return
            
        for block in self.current_flow.blocks:
            self.draw_block(block)
        
        if self.selected_block_id:
            tag = f"block_{self.selected_block_id}"
            body_items = self.canvas.find_withtag(f"{tag} and block_body")
            if body_items:
                self.canvas.itemconfig(body_items[0], outline="yellow", width=3)
        
        self.redraw_connections()

    def redraw_connections(self):
        """Draws the lines representing dependencies between blocks."""
        self.canvas.delete("connection")
        if not self.current_flow:
            return

        for block in self.current_flow.blocks:
            for dep_id in block.dependencies:
                out_pin_items = self.canvas.find_withtag(f"pin_{dep_id}_out")
                in_pin_items = self.canvas.find_withtag(f"pin_{block.id}_in")

                if out_pin_items and in_pin_items:
                    out_coords = self.canvas.coords(out_pin_items[0])
                    in_coords = self.canvas.coords(in_pin_items[0])
                    
                    x1, y1 = (out_coords[0] + out_coords[2]) / 2, (out_coords[1] + out_coords[3]) / 2
                    x2, y2 = (in_coords[0] + in_coords[2]) / 2, (in_coords[1] + in_coords[3]) / 2
                    
                    self.canvas.create_line(x1, y1, x1 + 50, y1, x2 - 50, y2, x2, y2, smooth=True, arrow=tk.LAST, fill="white", width=2, tags="connection")

    def update_flow_title(self):
        """Updates the flow title label."""
        if self.current_flow:
            self.flow_title_label.config(text=self.current_flow.name)
        else:
            self.flow_title_label.config(text="No Flow Loaded")

    def update_inspector(self):
        """Updates the inspector panel based on the selected block."""
        for tab in self.inspector_notebook.tabs():
            for widget in self.inspector_notebook.nametowidget(tab).winfo_children():
                widget.destroy()

        block = self.get_block_by_id(self.selected_block_id)
        if not block:
            self.show_inspector_message("No block selected.")
            return

        # --- Populate Settings Tab ---
        ttk.Label(self.settings_tab, text=f"Settings: {block.name}", style='Header.TLabel').pack(pady=10, fill=tk.X, padx=5)
        
        # --- Populate AI Tab ---
        ttk.Button(self.ai_tab, text="🔍 Analyze Code", command=self.analyze_selected_block).pack(pady=10, fill=tk.X, padx=10)
        ttk.Button(self.ai_tab, text="✨ Get AI Suggestion", command=self.get_ai_suggestion).pack(pady=5, fill=tk.X, padx=10)
        ttk.Button(self.ai_tab, text="🪄 Auto-Heal Error", command=self.auto_heal_block).pack(pady=5, fill=tk.X, padx=10)

    def show_inspector_message(self, message):
        """Displays a message in the inspector panel when no block is selected."""
        for tab in self.inspector_notebook.tabs():
            for widget in self.inspector_notebook.nametowidget(tab).winfo_children():
                widget.destroy()
        label = ttk.Label(self.settings_tab, text=message, wraplength=300, justify=tk.CENTER, anchor='center')
        label.pack(expand=True, padx=20, pady=20)

    # --- AI & Healing Integration ---

    def analyze_selected_block(self):
        messagebox.showinfo("Analyze", "Code analysis would be implemented here.")

    def get_ai_suggestion(self):
        messagebox.showinfo("AI Suggestion", "AI suggestions would be implemented here.")

    def auto_heal_block(self):
        messagebox.showinfo("Auto-Heal", "Auto-healing would be implemented here.")

    def show_settings(self):
        """Opens a dialog for application-level settings like the API key."""
        key = simpledialog.askstring("Settings", "Enter your OpenAI API Key:", parent=self.root)
        if key:
            save_api_key(key)
            messagebox.showinfo("Success", "API Key saved. AI features are now available.")

    # --- Flow Execution ---

    def run_flow(self):
        """Executes the entire flow based on dependencies."""
        if self.is_running_flow:
            messagebox.showwarning("Warning", "A flow is already running.")
            return
        if not self.current_flow: return
        
        messagebox.showinfo("Run Flow", "Flow execution would start here, processing blocks in order.")
        self.is_running_flow = True

    def stop_flow(self):
        """Stops all running blocks in the flow."""
        messagebox.showinfo("Stop Flow", "Stopping all running blocks.")
        self.is_running_flow = False
        
    def process_output_queue(self):
        """Processes messages from subprocesses to update the UI."""
        try:
            while True:
                block_id, tag, line = self.output_queue.get_nowait()
                if self.selected_block_id == block_id:
                    self.log_text.config(state=tk.NORMAL)
                    self.log_text.insert(tk.END, line)
                    self.log_text.see(tk.END)
                    self.log_text.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_output_queue)

    # --- App Lifecycle ---

    def on_close(self):
        """Handles the application close event."""
        if messagebox.askokcancel("Quit", "Do you want to exit Neural Weaver?"):
            self.profiler.stop_monitoring()
            self.root.destroy()

# --- Main Entry Point ---

def main():
    """The main function to create and run the Neural Weaver application."""
    root = ThemedTk(theme="clam")
    app = NeuralWeaverApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()

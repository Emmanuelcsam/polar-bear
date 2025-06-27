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
        self.process_manager = {} # Maps block_id to its subprocess and psutil process
        self.output_queue = queue.Queue() # For thread-safe UI updates from subprocesses
        self.is_running_flow = False
        
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

        # Configure styles for a dark, glassy feel
        self.style.configure('.', background=BG_COLOR, foreground=FG_COLOR, bordercolor=BORDER_COLOR)
        self.style.configure('TFrame', background=BG_COLOR)
        self.style.configure('TLabel', background=BG_COLOR, foreground=FG_COLOR, font=('Segoe UI', 10))
        self.style.configure('TButton', background="#40444b", foreground=FG_COLOR, borderwidth=1, focusthickness=3, focuscolor=ACCENT_COLOR)
        self.style.map('TButton', background=[('active', ACCENT_COLOR)])
        self.style.configure('Treeview', background="#23272a", foreground=FG_COLOR, fieldbackground="#23272a", rowheight=25)
        self.style.map('Treeview', background=[('selected', ACCENT_COLOR)])
        self.style.configure('Treeview.Heading', background="#40444b", font=('Segoe UI', 10, 'bold'))
        self.style.configure('Vertical.TScrollbar', background=BG_COLOR, troughcolor="#40444b")
        self.style.configure('Inspector.TFrame', background=INACTIVE_BG_COLOR, relief='solid', borderwidth=1)
        self.style.configure('Inspector.TLabel', background=INACTIVE_BG_COLOR, font=('Segoe UI', 11, 'bold'))
        self.style.configure('Header.TLabel', font=('Segoe UI', 12, 'bold'), foreground=ACCENT_COLOR)
        
        self.root.configure(bg=BG_COLOR)

    def build_ui(self):
        """Constructs the main UI layout with Palette, Canvas, and Inspector."""
        # Main Paned Window
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Left Panel: Block Palette ---
        palette_frame = ttk.Frame(main_pane, width=250, style='Inspector.TFrame')
        main_pane.add(palette_frame, weight=0)
        ttk.Label(palette_frame, text="Block Palette", style='Header.TLabel').pack(pady=10)

        self.block_palette = ttk.Treeview(palette_frame, show="tree")
        self.block_palette.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.populate_block_palette()
        self.block_palette.bind('<ButtonPress-1>', self.on_palette_drag_start)

        # --- Center Panel: Canvas ---
        canvas_container = ttk.Frame(main_pane)
        main_pane.add(canvas_container, weight=1)
        
        toolbar = ttk.Frame(canvas_container)
        toolbar.pack(fill=tk.X, pady=5)
        self.build_toolbar(toolbar)

        self.canvas = tk.Canvas(canvas_container, bg="#1e1f22", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Button-3>', self.on_canvas_right_click) # Right click for context menu

        # --- Right Panel: Inspector ---
        self.inspector_frame = ttk.Frame(main_pane, width=350, style='Inspector.TFrame')
        main_pane.add(self.inspector_frame, weight=0)
        
        self.inspector_notebook = ttk.Notebook(self.inspector_frame)
        self.inspector_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Build the individual tabs of the inspector
        self.build_inspector_tabs()
        self.show_inspector_message("Select a block to see its details or right-click the canvas for flow options.")

    def build_toolbar(self, parent):
        """Builds the main toolbar for flow control."""
        ttk.Button(parent, text="▶ Run Flow", command=self.run_flow).pack(side=tk.LEFT, padx=5)
        ttk.Button(parent, text="⏹ Stop Flow", command=self.stop_flow).pack(side=tk.LEFT, padx=5)
        ttk.Separator(parent, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill='y')
        ttk.Button(parent, text="💾 Save Flow", command=self.save_current_flow).pack(side=tk.LEFT, padx=5)
        ttk.Button(parent, text="➕ New Block", command=self.add_new_block).pack(side=tk.LEFT, padx=5)
        self.flow_title_label = ttk.Label(parent, text="No Flow Loaded", font=('Segoe UI', 12, 'bold'))
        self.flow_title_label.pack(side=tk.RIGHT, padx=20)

    def build_inspector_tabs(self):
        """Creates the tabs within the inspector panel."""
        # Settings Tab
        self.settings_tab = ttk.Frame(self.inspector_notebook)
        self.inspector_notebook.add(self.settings_tab, text="Settings")
        
        # Logs Tab
        self.logs_tab = ttk.Frame(self.inspector_notebook)
        self.inspector_notebook.add(self.logs_tab, text="Logs")
        self.log_text = tk.Text(self.logs_tab, wrap=tk.WORD, bg="#1e1f22", fg="lightgrey", relief=tk.FLAT)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)

        # Performance Tab
        self.perf_tab = ttk.Frame(self.inspector_notebook)
        self.inspector_notebook.add(self.perf_tab, text="Performance")

        # AI Assist Tab
        self.ai_tab = ttk.Frame(self.inspector_notebook)
        self.inspector_notebook.add(self.ai_tab, text="AI Assist ✨")

    def populate_block_palette(self):
        """Fills the block palette with predefined block types."""
        # Categories
        data_cat = self.block_palette.insert("", "end", text="Data")
        proc_cat = self.block_palette.insert("", "end", text="Processing")
        nn_cat = self.block_palette.insert("", "end", text="Neural Network")
        
        # Blocks
        self.block_palette.insert(data_cat, "end", text="File Input", values=("data_input", "Read data from a file (CSV, TXT)."))
        self.block_palette.insert(data_cat, "end", text="File Output", values=("data_output", "Save data to a file."))
        self.block_palette.insert(proc_cat, "end", text="Custom Python", values=("custom_python", "Write your own Python logic."))
        self.block_palette.insert(nn_cat, "end", text="Dense Layer", values=("nn_layer", "A fully connected neural network layer."))

    def on_palette_drag_start(self, event):
        """Initiates a drag-and-drop operation from the palette."""
        item_id = self.block_palette.identify_row(event.y)
        if not item_id:
            return

        item_values = self.block_palette.item(item_id, "values")
        if not item_values:
            return # It's a category, not a block
        
        block_type, description = item_values
        
        # Create a semi-transparent temporary window to represent the dragged block
        drag_window = tk.Toplevel(self.root)
        drag_window.overrideredirect(True)
        drag_window.attributes('-alpha', 0.7)
        drag_window.geometry(f"{BLOCK_WIDTH}x{BLOCK_HEIGHT}+{event.x_root}+{event.y_root}")
        
        label = ttk.Label(drag_window, text=self.block_palette.item(item_id, "text"), style='Header.TLabel', anchor="center")
        label.pack(expand=True, fill='both', ipady=20)
        label.configure(background=ACCENT_COLOR)
        
        self.canvas.bind('<Motion>', lambda e, dw=drag_window: self.on_palette_drag_move(e, dw))
        self.canvas.bind('<ButtonRelease-1>', lambda e, dw=drag_window, bt=block_type, bn=self.block_palette.item(item_id, "text"), bd=description: self.on_palette_drag_drop(e, dw, bt, bn, bd))

    def on_palette_drag_move(self, event, drag_window):
        """Updates the position of the temporary drag window."""
        drag_window.geometry(f"+{event.x_root}+{event.y_root}")

    def on_palette_drag_drop(self, event, drag_window, block_type, block_name, block_description):
        """Handles the creation of a new block when dropped on the canvas."""
        drag_window.destroy()
        self.canvas.unbind('<Motion>')
        self.canvas.unbind('<ButtonRelease-1>')

        if self.current_flow:
            x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
            self.add_new_block(block_type, block_name, block_description, {"x": int(x), "y": int(y)})

    def on_canvas_right_click(self, event):
        """Shows a context menu on the canvas."""
        menu = tk.Menu(self.root, tearoff=0, bg="#2c2f33", fg="#ffffff", activebackground="#7289da")
        menu.add_command(label="New Flow...", command=self.create_new_flow)
        menu.add_command(label="Open Flow...", command=self.open_flow)
        menu.add_separator()
        if self.current_flow:
            menu.add_command(label="Save Flow", command=self.save_current_flow)
            menu.add_command(label=f"Validate '{self.current_flow.name}'", command=self.validate_current_flow)
            menu.add_separator()
            menu.add_command(label="Add New Block", command=self.add_new_block)
        menu.add_separator()
        menu.add_command(label="Settings...", command=self.show_settings)
        menu.tk_popup(event.x_root, event.y_root)

    # --- Flow Management ---

    def load_initial_flow(self):
        """Loads the first available flow or shows a welcome message."""
        flows = self.config_manager.list_flows()
        if flows:
            self.load_flow_by_name(flows[0])
        else:
            self.create_new_flow(from_template="Simple Data Pipeline")

    def create_new_flow(self, from_template=None):
        """Creates a new, empty flow or loads one from a template."""
        if from_template:
            try:
                self.current_flow = self.config_manager.load_template(from_template)
            except FileNotFoundError:
                messagebox.showerror("Error", f"Template '{from_template}' not found. Creating empty flow.")
                self.current_flow = FlowConfig(name="New Flow")
        else:
             self.current_flow = FlowConfig(name="New Untitled Flow")
        
        self.redraw_canvas()
        self.update_flow_title()

    def open_flow(self):
        """Shows a dialog to open an existing flow."""
        flows = self.config_manager.list_flows()
        if not flows:
            messagebox.showinfo("No Flows", "No saved flows found. Create a new one!")
            return

        flow_name = simpledialog.askstring("Open Flow", "Enter the name of the flow to open:", parent=self.root)
        if flow_name and flow_name.title() in flows:
            self.load_flow_by_name(flow_name.title())
        elif flow_name:
            messagebox.showerror("Error", f"Flow '{flow_name}' not found.")

    def load_flow_by_name(self, name: str):
        """Loads a flow and displays it on the canvas."""
        try:
            self.current_flow = self.config_manager.load_flow(name)
            self.redraw_canvas()
            self.update_flow_title()
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load flow '{name}':\n{e}")

    def save_current_flow(self):
        """Saves the currently active flow."""
        if not self.current_flow:
            messagebox.showwarning("Warning", "No flow is currently loaded.")
            return
        
        try:
            self.config_manager.save_flow(self.current_flow)
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

    def add_new_block(self, block_type="custom_python", name="New Block", desc="", pos=None):
        """Adds a new block to the current flow."""
        if not self.current_flow:
            return

        if pos is None:
            pos = {"x": 100, "y": 100}
        
        new_id = f"block_{int(os.urandom(4).hex(), 16)}"
        new_block = BlockConfig(
            id=new_id,
            name=name,
            description=desc,
            block_type=block_type,
            position=pos
        )
        self.current_flow.blocks.append(new_block)
        self.draw_block(new_block)
        self.select_block(new_id)

    def draw_block(self, block: BlockConfig):
        """Draws a single block on the canvas."""
        x, y = block.position['x'], block.position['y']
        
        # Create a group of items for the block using tags
        tag = f"block_{block.id}"
        
        # Block body
        self.canvas.create_rectangle(x, y, x + BLOCK_WIDTH, y + BLOCK_HEIGHT,
                                     fill="#40444b", outline="#7289da", width=2, tags=(tag, "block_body"))
        
        # Block name
        self.canvas.create_text(x + BLOCK_WIDTH/2, y + 20, text=block.name,
                                fill="white", font=('Segoe UI', 10, 'bold'), tags=tag)
        
        # Input/Output Pins
        # For simplicity, one input on the left, one output on the right.
        self.canvas.create_oval(x - PIN_RADIUS, y + BLOCK_HEIGHT/2 - PIN_RADIUS,
                                x + PIN_RADIUS, y + BLOCK_HEIGHT/2 + PIN_RADIUS,
                                fill="cyan", tags=(tag, "pin", "input_pin", f"pin_{block.id}_in"))
                                
        self.canvas.create_oval(x + BLOCK_WIDTH - PIN_RADIUS, y + BLOCK_HEIGHT/2 - PIN_RADIUS,
                                x + BLOCK_WIDTH + PIN_RADIUS, y + BLOCK_HEIGHT/2 + PIN_RADIUS,
                                fill="magenta", tags=(tag, "pin", "output_pin", f"pin_{block.id}_out"))

        # Bind events for dragging
        self.canvas.tag_bind(tag, '<ButtonPress-1>', lambda e, b_id=block.id: self.on_block_press(e, b_id))
        self.canvas.tag_bind(tag, '<B1-Motion>', self.on_block_drag)
        self.canvas.tag_bind(tag, '<ButtonRelease-1>', self.on_block_release)

    def on_block_press(self, event, block_id):
        """Handles when a block is clicked."""
        self.select_block(block_id)
        # For dragging
        self._drag_data = {'x': event.x, 'y': event.y, 'id': block_id}

    def on_block_drag(self, event):
        """Handles dragging a block."""
        dx = event.x - self._drag_data['x']
        dy = event.y - self._drag_data['y']
        
        # Move all parts of the block
        tag = f"block_{self._drag_data['id']}"
        self.canvas.move(tag, dx, dy)
        
        self._drag_data['x'] = event.x
        self._drag_data['y'] = event.y

        self.redraw_connections() # Redraw connections as block moves

    def on_block_release(self, event):
        """Handles when a block is released after dragging."""
        block_id = self._drag_data['id']
        block = self.get_block_by_id(block_id)
        if block:
            # Get the new top-left corner of the rectangle
            body = self.canvas.find_withtag(f"block_{block_id}") and self.canvas.find_withtag("block_body")
            coords = self.canvas.coords(body[0])
            block.position['x'] = int(coords[0])
            block.position['y'] = int(coords[1])
        self._drag_data = {}
        self.redraw_canvas()

    def get_block_by_id(self, block_id: str) -> Optional[BlockConfig]:
        """Finds a block in the current flow by its ID."""
        if self.current_flow:
            for block in self.current_flow.blocks:
                if block.id == block_id:
                    return block
        return None

    def select_block(self, block_id: str):
        """Selects a block and updates the inspector panel."""
        self.selected_block_id = block_id
        
        # Deselect others visually
        self.canvas.itemconfig('block_body', outline="#7289da", width=2)
        # Highlight selected
        tag = f"block_{block_id}"
        self.canvas.itemconfig(self.canvas.find_withtag(tag) and self.canvas.find_withtag("block_body"), outline="yellow", width=3)
        
        self.update_inspector()

    # --- Canvas & UI Redrawing ---
    
    def redraw_canvas(self):
        """Clears and redraws the entire canvas from the current flow data."""
        self.canvas.delete("all")
        if not self.current_flow:
            return
            
        for block in self.current_flow.blocks:
            self.draw_block(block)
        
        self.redraw_connections()

    def redraw_connections(self):
        """Draws the lines representing dependencies between blocks."""
        self.canvas.delete("connection")
        if not self.current_flow:
            return

        for block in self.current_flow.blocks:
            for dep_id in block.dependencies:
                # Get coordinates of output pin of dependency and input pin of current block
                out_pin_tag = f"pin_{dep_id}_out"
                in_pin_tag = f"pin_{block.id}_in"
                
                out_pin_items = self.canvas.find_withtag(out_pin_tag)
                in_pin_items = self.canvas.find_withtag(in_pin_tag)

                if out_pin_items and in_pin_items:
                    out_coords = self.canvas.coords(out_pin_items[0])
                    in_coords = self.canvas.coords(in_pin_items[0])
                    
                    # Center of pins
                    x1 = (out_coords[0] + out_coords[2]) / 2
                    y1 = (out_coords[1] + out_coords[3]) / 2
                    x2 = (in_coords[0] + in_coords[2]) / 2
                    y2 = (in_coords[1] + in_coords[3]) / 2
                    
                    # Draw a curved line
                    self.canvas.create_line(x1, y1, x1 + 50, y1, x2 - 50, y2, x2, y2,
                                            smooth=True, arrow=tk.LAST, fill="white", width=2, tags="connection")

    def update_flow_title(self):
        """Updates the flow title label."""
        if self.current_flow:
            self.flow_title_label.config(text=self.current_flow.name)
        else:
            self.flow_title_label.config(text="No Flow Loaded")

    def update_inspector(self):
        """Updates the inspector panel based on the selected block."""
        # Clear previous inspector content
        for tab in [self.settings_tab, self.logs_tab, self.perf_tab, self.ai_tab]:
             for widget in tab.winfo_children():
                 widget.destroy()

        block = self.get_block_by_id(self.selected_block_id)
        if not block:
            self.show_inspector_message("No block selected.")
            return

        # --- Populate Settings Tab ---
        ttk.Label(self.settings_tab, text=f"Settings: {block.name}", style='Header.TLabel').pack(pady=10, fill=tk.X)
        # You would build a dynamic form here based on block.settings
        
        # --- Populate AI Tab ---
        ttk.Button(self.ai_tab, text="🔍 Analyze Code", command=self.analyze_selected_block).pack(pady=10, fill=tk.X)
        ttk.Button(self.ai_tab, text="✨ Get AI Suggestion", command=self.get_ai_suggestion).pack(pady=5, fill=tk.X)
        ttk.Button(self.ai_tab, text="🪄 Auto-Heal Error", command=self.auto_heal_block).pack(pady=5, fill=tk.X)

    def show_inspector_message(self, message):
        """Displays a message in the inspector panel when no block is selected."""
        self.update_inspector() # Clears tabs
        label = ttk.Label(self.settings_tab, text=message, wraplength=300, justify=tk.CENTER)
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
        api_key = simpledialog.askstring("Settings", "Enter your OpenAI API Key:", parent=self.root)
        if api_key:
            save_api_key(api_key)
            messagebox.showinfo("Success", "API Key saved. AI features are now enabled.")

    # --- Flow Execution ---

    def run_flow(self):
        """Executes the entire flow based on dependencies."""
        if self.is_running_flow:
            messagebox.showwarning("Warning", "A flow is already running.")
            return
        
        if not self.current_flow:
            return
        
        # Simplified execution - would need full topological sort and process management
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
            pass # Queue is empty, do nothing
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
    # Use ThemedTk for better styling capabilities
    root = ThemedTk(theme="clam")
    app = NeuralWeaverApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()

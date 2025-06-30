#!/usr/bin/env python3
"""
Test script for VS Code Copilot Auto-Continue Clicker
Creates a window with a blue Continue button that mimics VS Code's style
"""

import tkinter as tk
from tkinter import ttk, messagebox
import random
import time
import sys

class ContinueButtonTest:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VS Code Copilot Continue Button Test")
        self.root.geometry("800x600")
        self.root.configure(bg='#1e1e1e')  # VS Code dark background
        
        # Statistics
        self.button_shown_count = 0
        self.button_clicked_count = 0
        self.auto_clicked_count = 0
        self.start_time = time.time()
        
        # Create UI
        self.setup_ui()
        
        # Start button positioning
        self.reposition_button()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title = tk.Label(
            self.root, 
            text="VS Code Copilot Auto-Clicker Test",
            font=("Arial", 20, "bold"),
            fg='white',
            bg='#1e1e1e'
        )
        title.pack(pady=20)
        
        # Instructions
        instructions = tk.Label(
            self.root,
            text="A blue 'Continue' button will appear randomly on the screen.\n" +
                 "The auto-clicker should detect and click it automatically.\n" +
                 "The button will reappear in a new location after being clicked.",
            font=("Arial", 12),
            fg='#cccccc',
            bg='#1e1e1e',
            justify=tk.CENTER
        )
        instructions.pack(pady=10)
        
        # Stats frame
        stats_frame = tk.Frame(self.root, bg='#1e1e1e')
        stats_frame.pack(side=tk.TOP, pady=20)
        
        # Stats labels
        self.stats_labels = {}
        stats = [
            ("Buttons Shown:", "shown"),
            ("Manual Clicks:", "manual"),
            ("Auto Clicks:", "auto"),
            ("Success Rate:", "rate"),
            ("Uptime:", "uptime")
        ]
        
        for i, (label_text, key) in enumerate(stats):
            label = tk.Label(
                stats_frame,
                text=label_text,
                font=("Arial", 10),
                fg='#999999',
                bg='#1e1e1e'
            )
            label.grid(row=i, column=0, sticky='w', padx=10)
            
            value_label = tk.Label(
                stats_frame,
                text="0",
                font=("Arial", 10, "bold"),
                fg='white',
                bg='#1e1e1e'
            )
            value_label.grid(row=i, column=1, sticky='w', padx=10)
            self.stats_labels[key] = value_label
        
        # Control buttons frame
        control_frame = tk.Frame(self.root, bg='#1e1e1e')
        control_frame.pack(side=tk.BOTTOM, pady=20)
        
        # Control buttons
        tk.Button(
            control_frame,
            text="Reset Stats",
            command=self.reset_stats,
            bg='#3c3c3c',
            fg='white',
            font=("Arial", 10),
            relief=tk.FLAT,
            padx=20,
            pady=5
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            control_frame,
            text="Exit",
            command=self.root.quit,
            bg='#3c3c3c',
            fg='white',
            font=("Arial", 10),
            relief=tk.FLAT,
            padx=20,
            pady=5
        ).pack(side=tk.LEFT, padx=5)
        
        # The Continue button (initially hidden)
        self.continue_button = tk.Button(
            self.root,
            text="Continue",
            font=("Arial", 11),
            fg='white',
            bg='#0e639c',  # VS Code blue
            activebackground='#1177bb',
            activeforeground='white',
            relief=tk.FLAT,
            padx=20,
            pady=6,
            cursor='hand2',
            command=self.button_clicked
        )
        
        # Update stats periodically
        self.update_stats()
        
    def create_continue_button_style(self):
        """Create a button that looks like VS Code's Continue button"""
        # Place the button
        self.continue_button.place(x=400, y=300)
        self.button_shown_count += 1
        
    def button_clicked(self, auto=False):
        """Handle button click"""
        if auto:
            self.auto_clicked_count += 1
            print(f"✅ AUTO-CLICKED at position ({self.button_x}, {self.button_y})")
        else:
            self.button_clicked_count += 1
            print(f"🖱️  Manual click at position ({self.button_x}, {self.button_y})")
        
        # Hide button briefly
        self.continue_button.place_forget()
        
        # Reposition after delay
        self.root.after(1500, self.reposition_button)
        
    def reposition_button(self):
        """Move the button to a random position"""
        # Get window dimensions
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()
        
        # Button dimensions (approximate)
        button_width = 100
        button_height = 35
        
        # Random position (keep button fully visible)
        self.button_x = random.randint(50, max(51, window_width - button_width - 50))
        self.button_y = random.randint(150, max(151, window_height - button_height - 100))
        
        # Place button
        self.continue_button.place(x=self.button_x, y=self.button_y)
        self.button_shown_count += 1
        
        # Simulate auto-click detection
        self.check_for_auto_click()
        
    def check_for_auto_click(self):
        """Check if auto-clicker clicked the button"""
        # This will be called periodically to check if the button is still visible
        # If the auto-clicker clicked it, the button would be hidden
        def check():
            if self.continue_button.winfo_viewable():
                # Button still visible, check again
                self.root.after(100, check)
            else:
                # Button was clicked (either manually or by auto-clicker)
                pass
        
        # Start checking after a small delay
        self.root.after(500, check)
        
    def reset_stats(self):
        """Reset statistics"""
        self.button_shown_count = 0
        self.button_clicked_count = 0
        self.auto_clicked_count = 0
        self.start_time = time.time()
        print("📊 Stats reset!")
        
    def update_stats(self):
        """Update statistics display"""
        # Calculate stats
        total_clicks = self.button_clicked_count + self.auto_clicked_count
        success_rate = (self.auto_clicked_count / max(1, self.button_shown_count)) * 100
        uptime = int(time.time() - self.start_time)
        uptime_str = f"{uptime // 60}m {uptime % 60}s"
        
        # Update labels
        self.stats_labels['shown'].config(text=str(self.button_shown_count))
        self.stats_labels['manual'].config(text=str(self.button_clicked_count))
        self.stats_labels['auto'].config(text=str(self.auto_clicked_count))
        self.stats_labels['rate'].config(text=f"{success_rate:.1f}%")
        self.stats_labels['uptime'].config(text=uptime_str)
        
        # Color code success rate
        if success_rate >= 80:
            self.stats_labels['rate'].config(fg='#4ec9b0')  # Green
        elif success_rate >= 50:
            self.stats_labels['rate'].config(fg='#dcdcaa')  # Yellow
        else:
            self.stats_labels['rate'].config(fg='#f44747')  # Red
        
        # Schedule next update
        self.root.after(1000, self.update_stats)
    
    def run(self):
        """Start the test application"""
        print("🎯 VS Code Copilot Continue Button Test")
        print("=" * 50)
        print("A blue 'Continue' button will appear in the window.")
        print("Run your auto-clicker script to test detection.")
        print("=" * 50)
        
        # Override window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start
        self.root.mainloop()
        
    def on_closing(self):
        """Handle window closing"""
        print("\n📊 Final Statistics:")
        print(f"   Buttons shown: {self.button_shown_count}")
        print(f"   Manual clicks: {self.button_clicked_count}")
        print(f"   Auto clicks: {self.auto_clicked_count}")
        print(f"   Success rate: {(self.auto_clicked_count / max(1, self.button_shown_count)) * 100:.1f}%")
        self.root.destroy()

class SimpleButtonTest:
    """Simpler test with just a button"""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Simple Continue Button Test")
        self.root.geometry("400x300")
        self.root.configure(bg='#2d2d2d')
        
        # Create a frame to hold the button
        frame = tk.Frame(self.root, bg='#252526', relief=tk.RAISED, bd=1)
        frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Create the Continue button
        self.button = tk.Button(
            frame,
            text="Continue",
            font=("Segoe UI", 11),
            fg='white',
            bg='#0e639c',
            activebackground='#1177bb',
            activeforeground='white',
            relief=tk.FLAT,
            bd=0,
            padx=24,
            pady=6,
            command=self.clicked
        )
        self.button.pack(padx=20, pady=20)
        
        # Click counter
        self.clicks = 0
        
        # Status label
        self.status = tk.Label(
            self.root,
            text="Waiting for auto-click...",
            font=("Arial", 10),
            fg='#cccccc',
            bg='#2d2d2d'
        )
        self.status.pack(side=tk.BOTTOM, pady=10)
        
    def clicked(self):
        """Button was clicked"""
        self.clicks += 1
        print(f"✅ Button clicked! (Total: {self.clicks})")
        self.status.config(text=f"Clicked {self.clicks} times")
        
        # Flash effect
        original_bg = self.button['bg']
        self.button.config(bg='#1177bb')
        self.root.after(100, lambda: self.button.config(bg=original_bg))
        
    def run(self):
        """Run the simple test"""
        print("🎯 Simple Continue Button Test")
        print("A static blue 'Continue' button is displayed.")
        print("Run your auto-clicker to test if it detects and clicks it.")
        print("-" * 50)
        self.root.mainloop()

def main():
    """Main entry point"""
    print("Choose test mode:")
    print("1. Dynamic test (button moves after each click)")
    print("2. Simple test (static button)")
    
    choice = input("\nSelect mode [1]: ").strip()
    
    if choice == '2':
        test = SimpleButtonTest()
    else:
        test = ContinueButtonTest()
    
    try:
        test.run()
    except KeyboardInterrupt:
        print("\nTest interrupted.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check if tkinter is available
    try:
        import tkinter
    except ImportError:
        print("❌ tkinter is required for the test script")
        print("Install with:")
        print("  Ubuntu/Debian: sudo apt-get install python3-tk")
        print("  Fedora: sudo dnf install python3-tkinter")
        print("  macOS: tkinter should be included with Python")
        sys.exit(1)
    
    main()

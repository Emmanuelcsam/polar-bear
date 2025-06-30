"""
BlockNet Builder – main GUI wrapper around the existing Neural‑IDE engine.
All heavy logic stays in app/engine; here we only provide a
block‑based editor plus simple run / stop controls.
"""
from __future__ import annotations

from pathlib import Path
import importlib
import tkinter as tk
from tkinter import ttk, messagebox

from ui_dark_glass import apply as apply_theme
from block_canvas import BlockCanvas
from smart_ai import SmartAI
from code_check import CodeCheck
from auto_fix import AutoFix

# --- bring original engine into the same process -----------------------
_engine = importlib.import_module("engine.neural_ide_tools")        # your code
_orchestrator = _engine.ScriptOrchestrator                          # type: ignore
_message_bus = _engine.MessageBroker()                              # type: ignore
_message_bus.start()


# ----------------------------------------------------------------------
def run() -> None:
    root = ttk.Window(themename="darkly")           # ttkbootstrap window
    apply_theme(root)
    App(root)
    root.mainloop()


class App:
    def __init__(self, root: tk.Tk):
        self.root, self.ai = root, None

        # ─── topbar ───────────────────────────────────────────────────
        bar = ttk.Frame(root)
        bar.pack(fill="x")
        ttk.Button(bar, text="＋ New Block", command=self.new_block).pack(side="left")
        ttk.Button(bar, text="▶ Run", command=self.run_blocks).pack(side="left")
        ttk.Button(bar, text="■ Stop", command=lambda: _message_bus.stop()).pack(side="left")

        self.ai_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="💡 Smart‑AI Help",
                        variable=self.ai_var,
                        command=self._toggle_ai, bootstyle="success").pack(side="right")

        # ─── centre workspace ───────────────────────────────────────
        self.canvas = BlockCanvas(root)
        self.canvas.pack(fill="both", expand=True)

        # ─── right mini‑console ─────────────────────────────────────
        self.log = tk.Text(root, height=8, width=50, bg="#1a1a1a", fg="#e0e0e0")
        self.log.pack(side="bottom", fill="x")
        self._write("Ready.  Build your net with blocks and press ▶ Run.\n")

    # ----------------------- helpers ---------------------------------
    def _toggle_ai(self):
        if self.ai_var.get():
            self.ai = SmartAI()
            self._write("Smart‑AI enabled.\n")
        else:
            self.ai = None
            self._write("Smart‑AI disabled.\n")

    def new_block(self):
        self.canvas.add_block()

    # ----------------------- execution -------------------------------
    def run_blocks(self):
        files = self.canvas.export_python()      # {name: path}
        for fname, path in files.items():
            CodeCheck(path).quick_check()        # lints -> stdout

            if self.ai:                          # optional refactor
                suggestion = self.ai.refactor(Path(path).read_text())
                AutoFix(path).apply(suggestion)

        # pass the set of scripts to the original orchestrator
        orch = _orchestrator(_message_bus)
        for name, path in files.items():
            node = _engine.ScriptNode(name, path)         # type: ignore
            orch.add_script(node)
        orch.execute_all(parallel=True)
        self._write("Execution kicked‑off …\n")

    # ---------------------- ui utils ---------------------------------
    def _write(self, txt: str):
        self.log.configure(state="normal")
        self.log.insert("end", txt)
        self.log.configure(state="disabled")
        self.log.see("end")

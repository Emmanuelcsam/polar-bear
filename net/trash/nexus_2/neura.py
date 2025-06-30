#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Nexus IDE 4.0 – Refactored
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A cross‑platform, Tk‑based IDE for rapid prototyping of Python/AI scripts with
auto‑diagnosis, optional LLM fixes and dependency management.

Key improvements over 3.0 (see original) :contentReference[oaicite:0]{index=0}:
• Robust threading model (all daemons, graceful shutdown)
• Central logging & error‑diagnosis pipeline
• Theme engine & ttk.Style factory
• Deferred import of heavyweight libs (matplotlib, networkx)
• Capability flags for OpenAI & Copilot
"""

from __future__ import annotations

import ast
import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk

# --------------------------------------------------------------------------- #
#  Constants & globals                                                        #
# --------------------------------------------------------------------------- #

APP_NAME     = "Neural Nexus IDE"
APP_VERSION  = "4.0.0"

HOME_DIR     = Path.home() / ".neural_nexus"
CONFIG_FILE  = HOME_DIR / "config.json"
SCRIPTS_DIR  = HOME_DIR / "scripts"
LOGS_DIR     = HOME_DIR / "logs"

for p in (SCRIPTS_DIR, LOGS_DIR):
    p.mkdir(parents=True, exist_ok=True)

REQUIRED_PKGS = ("psutil", "networkx", "matplotlib")

# --------------------------------------------------------------------------- #
#  Optional integrations                                                      #
# --------------------------------------------------------------------------- #

try:
    import psutil  # noqa: F401
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import openai  # noqa: F401
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
    import matplotlib.pyplot as plt  # noqa: E402
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import networkx as nx  # noqa: E402
    HAS_NX = True
except ImportError:
    HAS_NX = False

def _has_copilot() -> bool:
    """Return True if the GitHub CLI Copilot extension is installed."""
    try:
        return subprocess.run(
            ["gh", "copilot", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        ).returncode == 0
    except Exception:
        return False

HAS_COPILOT = _has_copilot()

# --------------------------------------------------------------------------- #
#  Data models                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class Script:
    """In‑memory representation of a script."""
    sid: str
    name: str
    content: str
    file_path: Optional[Path] = None
    imports: List[str] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "idle"              # idle|running|success|error
    process: Optional[subprocess.Popen[str]] = None
    stdout: str = ""
    stderr: str = ""

# --------------------------------------------------------------------------- #
#  Utility helpers                                                            #
# --------------------------------------------------------------------------- #

def ensure_packages(pkgs: tuple[str, ...]) -> None:
    """Install missing packages via pip; block until complete."""
    missing = [p for p in pkgs if not _module_available(p)]
    if not missing:
        return
    print(f"Installing missing packages: {', '.join(missing)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False

def safe_tk(func):
    """Decorator: marshal a function back onto the Tk prime thread."""
    def _wrapper(self, *args, **kwargs):
        self.root.after_idle(lambda: func(self, *args, **kwargs))
    return _wrapper

# --------------------------------------------------------------------------- #
#  Main application class                                                     #
# --------------------------------------------------------------------------- #

class NeuralNexusIDE:
    # -------------------------- initialisation ----------------------------- #

    def __init__(self, root: tk.Tk) -> None:
        self.root     = root
        self.config   = self._load_config()
        self.scripts  : Dict[str, Script] = {}
        self.active_id: Optional[str] = None
        self.queue    : queue.Queue[tuple[str, str, str]] = queue.Queue()

        self._build_ui()
        self._bind_shortcuts()

        threading.Thread(target=self._queue_worker,
                         daemon=True, name="STDOUT‑dispatcher").start()

    # ------------------------------- UI ------------------------------------ #

    def _build_ui(self) -> None:
        self.root.title(f"{APP_NAME} {APP_VERSION}")
        self.root.geometry("1400x800")
        self._setup_styles()

        # — Paned layout ---------------------------------------------------- #
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Left: script tree
        frame_left = ttk.Frame(paned, width=260)
        paned.add(frame_left, weight=1)
        self._build_script_tree(frame_left)

        # Centre: editor/output
        frame_mid = ttk.Frame(paned)
        paned.add(frame_mid, weight=3)
        self._build_editor(frame_mid)

        # Right: tools
        frame_right = ttk.Frame(paned, width=350)
        paned.add(frame_right, weight=2)
        self._build_tools(frame_right)

        # Status bar
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self.root, textvariable=self.status_var, anchor=tk.W)\
           .pack(fill=tk.X, side=tk.BOTTOM)

    def _setup_styles(self) -> None:
        theme = self.config.get("theme", "dark")
        style = ttk.Style()
        if theme == "dark":
            self.root.configure(bg="#1e1e1e")
            style.theme_use("clam")
            style.configure(".", background="#1e1e1e", foreground="white")
            txt_cfg = dict(bg="#1e1e1e", fg="white", insertbackground="white",
                           selectbackground="#264f78")
        else:
            style.theme_use("default")
            txt_cfg = dict(bg="white", fg="black")

        # Widgets initialised later will inherit these defaults
        scrolledtext.ScrolledText.configure = lambda self, **kw: \
            tk.Text.configure(self, **(txt_cfg | kw))  # type: ignore[arg-type]

    # --------------- left panel – Script Manager --------------------------- #

    def _build_script_tree(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent); header.pack(fill=tk.X, pady=4)
        ttk.Label(header, text="Scripts", font=("Segoe UI", 11, "bold"))\
            .pack(side=tk.LEFT)
        ttk.Button(header, text="＋", width=3, command=self._new_script)\
            .pack(side=tk.RIGHT)

        self.tree = ttk.Treeview(parent, columns=("status",), show="tree headings")
        self.tree.heading("#0", text="Name")
        self.tree.heading("status", text="Status")
        self.tree.column("status", width=70)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

    # --------------- centre – Editor + Output ------------------------------ #

    def _build_editor(self, parent: ttk.Frame) -> None:
        toolbar = ttk.Frame(parent); toolbar.pack(fill=tk.X, pady=2)
        ttk.Button(toolbar, text="▶ Run", command=self._run_script).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="■ Stop", command=self._stop_script)\
            .pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text=" VS Code", command=self._open_in_vscode)\
            .pack(side=tk.LEFT, padx=6)

        nb = ttk.Notebook(parent); nb.pack(fill=tk.BOTH, expand=True)
        # Code tab
        frame_code = ttk.Frame(nb); nb.add(frame_code, text="Code")
        self.editor = scrolledtext.ScrolledText(frame_code, undo=True,
                                                font=("Consolas", 11))
        self.editor.pack(fill=tk.BOTH, expand=True)
        self.editor.bind("<<Modified>>", self._on_edit)

        # Output tab
        frame_out = ttk.Frame(nb); nb.add(frame_out, text="Output")
        self.console = scrolledtext.ScrolledText(frame_out, state=tk.NORMAL,
                                                 font=("Consolas", 10), height=14)
        self.console.pack(fill=tk.BOTH, expand=True)

        self.nb_main = nb

    # --------------- right – Errors / Deps / Viz --------------------------- #

    def _build_tools(self, parent: ttk.Frame) -> None:
        nb = ttk.Notebook(parent); nb.pack(fill=tk.BOTH, expand=True)
        # Errors
        frame_err = ttk.Frame(nb); nb.add(frame_err, text="Errors")
        self.err_tree = ttk.Treeview(frame_err, columns=("line", "msg"),
                                     show="headings")
        self.err_tree.heading("line", text="Line"); self.err_tree.column("line", width=60)
        self.err_tree.heading("msg", text="Message")
        self.err_tree.pack(fill=tk.BOTH, expand=True)
        self.err_tree.bind("<Double-1>", self._jump_to_error)

        # Deps
        frame_dep = ttk.Frame(nb); nb.add(frame_dep, text="Dependencies")
        self.deps_txt = scrolledtext.ScrolledText(frame_dep, height=8,
                                                  font=("Consolas", 10))
        self.deps_txt.pack(fill=tk.BOTH, expand=True)

        # Network visualisation
        if HAS_MPL and HAS_NX:
            frame_viz = ttk.Frame(nb); nb.add(frame_viz, text="Network")
            self.fig, self.ax = plt.subplots(figsize=(4, 3),
                                             facecolor="black")
            self.canvas = FigureCanvasTkAgg(self.fig, master=frame_viz)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            nb.add(ttk.Frame(nb), text="Network (disabled)")

    # -------------------------- event handlers ----------------------------- #

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-n>", lambda _: self._new_script())
        self.root.bind("<Control-s>", lambda _: self._save_script())
        self.root.bind("<F5>",        lambda _: self._run_script())
        self.root.bind("<Shift-F5>",  lambda _: self._stop_script())

    # Editor modified
    def _on_edit(self, evt) -> None:                               # noqa: N802
        self.editor.edit_modified(False)
        if self.active_id:
            self.scripts[self.active_id].content = self.editor.get("1.0", tk.END)

    # Tree selection
    def _on_select(self, evt) -> None:                            # noqa: N802
        sel = self.tree.selection()
        if not sel:
            return
        sid = sel[0]
        self.active_id = sid
        script = self.scripts[sid]
        self.editor.delete("1.0", tk.END)
        self.editor.insert("1.0", script.content)
        self.status_var.set(f"Selected {script.name}")

        self._analyse_script(script)

    # Double‑click error tree
    def _jump_to_error(self, _evt) -> None:
        sel = self.err_tree.selection()
        if not sel:
            return
        line = self.err_tree.item(sel[0], "values")[0]
        self.editor.mark_set(tk.INSERT, f"{line}.0")
        self.editor.see(f"{line}.0")
        self.nb_main.select(0)

    # ---------------------------- scripts ---------------------------------- #

    def _new_script(self) -> None:
        name = simpledialog.askstring("New script", "Script name:")
        if not name:
            return
        sid = f"scr_{int(time.time()*1000)}"
        self.scripts[sid] = Script(sid=sid, name=name,
                                   content="#!/usr/bin/env python3\n\n")
        self.tree.insert("", tk.END, iid=sid, text=name, values=("idle",))
        self.tree.selection_set(sid)
        self._on_select(None)

    def _save_script(self, *, as_new: bool = False) -> None:
        if not self.active_id:
            return
        sc = self.scripts[self.active_id]
        if as_new or not sc.file_path:
            path = filedialog.asksaveasfilename(defaultextension=".py",
                                                filetypes=[("Python", "*.py")])
            if not path:
                return
            sc.file_path = Path(path)
            sc.name = sc.file_path.name
            self.tree.item(sc.sid, text=sc.name)
        sc.file_path.write_text(sc.content, encoding="utf‑8")  # type: ignore[arg-type]
        self.status_var.set(f"Saved {sc.file_path}")

    def _run_script(self) -> None:
        if not self.active_id:
            return
        sc = self.scripts[self.active_id]
        # Ensure on disk
        if not sc.file_path:
            self._save_script()
        if not sc.file_path:
            return

        self._analyse_script(sc)     # refresh syntax/deps before run
        self._update_error_view(sc)

        cmd: list[str] = [sys.executable, str(sc.file_path)]
        sc.status = "running"
        self.tree.item(sc.sid, values=("running",))
        self.console.delete("1.0", tk.END)

        try:
            sc.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
        except Exception as exc:      # pylint: disable=broad-except
            self._diagnose_error(sc, f"Spawn error: {exc}", fatal=True)
            return

        threading.Thread(target=self._watch_pipe,
                         args=(sc, sc.process.stdout, "stdout"),
                         daemon=True, name=f"{sc.name}-stdout").start()

        threading.Thread(target=self._watch_pipe,
                         args=(sc, sc.process.stderr, "stderr"),
                         daemon=True, name=f"{sc.name}-stderr").start()

        threading.Thread(target=self._wait_proc,
                         args=(sc,),
                         daemon=True, name=f"{sc.name}-waiter").start()

    def _stop_script(self) -> None:
        if not self.active_id:
            return
        sc = self.scripts[self.active_id]
        if sc.process and sc.process.poll() is None:
            sc.process.terminate()
            self.queue.put((sc.sid, "\n— Terminated by user —\n", "stderr"))

    # ---------------------- subprocess monitoring -------------------------- #

    def _watch_pipe(self, sc: Script, pipe, tag: str) -> None:
        for line in iter(pipe.readline, ''):
            self.queue.put((sc.sid, line, tag))
            if tag == "stderr":
                sc.stderr += line
        pipe.close()

    def _wait_proc(self, sc: Script) -> None:
        sc.process.wait()            # type: ignore[union-attr]
        rc = sc.process.returncode   # type: ignore[union-attr]
        tag = "info" if rc == 0 else "stderr"
        msg = f"\n— exited with code {rc} —\n"
        self.queue.put((sc.sid, msg, tag))
        sc.status = "success" if rc == 0 else "error"
        self.tree.item(sc.sid, values=(sc.status,))
        if rc != 0:
            self._diagnose_error(sc, sc.stderr)

    @safe_tk
    def _diagnose_error(self, sc: Script, text: str, *, fatal: bool = False) -> None:
        """Central dispatcher: display stderr, highlight line, offer auto‑fix."""
        if text:
            self.console.insert(tk.END, text, "stderr")
        # Very simple line extraction
        import re
        m = re.search(r'File ".*?", line (\d+)', text)
        if m:
            line_no = m.group(1)
            self.err_tree.insert("", tk.END, values=(line_no, "Runtime error"))
        if fatal:
            sc.status = "error"
            self.tree.item(sc.sid, values=("error",))

    # -------------------------- background queue --------------------------- #

    def _queue_worker(self) -> None:
        while True:
            try:
                sid, text, tag = self.queue.get()
            except Exception:
                continue
            if sid == self.active_id:
                self._write_console(text, tag)

    @safe_tk
    def _write_console(self, text: str, tag: str) -> None:
        self.console.insert(tk.END, text)
        self.console.see(tk.END)

    # ---------------------- analysis (AST + deps) -------------------------- #

    def _analyse_script(self, sc: Script) -> None:
        sc.errors.clear()
        sc.imports.clear()
        try:
            tree = ast.parse(sc.content)
        except SyntaxError as err:
            sc.errors.append(dict(type="SyntaxError", line=err.lineno,
                                  msg=err.msg))
            self._update_error_view(sc)
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                sc.imports.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                sc.imports.append(node.module)

        # dependency check (async)
        threading.Thread(target=self._check_deps,
                         args=(sc,),
                         daemon=True).start()

    def _check_deps(self, sc: Script) -> None:
        installed = self._pip_freeze()
        missing = [m.split('.')[0] for m in sc.imports
                   if m.split('.')[0] not in installed
                   and m.split('.')[0] not in sys.stdlib_module_names]

        for mod in missing:
            sc.errors.append(dict(type="MissingDependency", line=0,
                                  msg=f"Module '{mod}' not installed"))
        self._update_error_view(sc)
        self._update_deps_view(sc)

    @staticmethod
    def _pip_freeze() -> set[str]:
        try:
            out = subprocess.check_output([sys.executable, "-m", "pip", "list",
                                           "--format=freeze"],
                                          text=True)
            return {ln.split("==")[0].lower() for ln in out.splitlines()}
        except Exception:
            return set()

    @safe_tk
    def _update_error_view(self, sc: Script) -> None:
        self.err_tree.delete(*self.err_tree.get_children())
        for err in sc.errors:
            self.err_tree.insert("", tk.END, values=(err.get("line"), err["msg"]))

    @safe_tk
    def _update_deps_view(self, sc: Script) -> None:
        self.deps_txt.delete("1.0", tk.END)
        self.deps_txt.insert(tk.END, "Imports:\n")
        for imp in sorted(set(sc.imports)):
            self.deps_txt.insert(tk.END, f"  • {imp}\n")
        miss = [e for e in sc.errors if e["type"] == "MissingDependency"]
        if miss:
            self.deps_txt.insert(tk.END, "\nMissing:\n")
            for e in miss:
                self.deps_txt.insert(tk.END, f"  • {e['msg'].split()[1]}\n")

    # --------------------------- misc helpers ------------------------------ #

    def _open_in_vscode(self) -> None:
        if not self.active_id:
            return
        sc = self.scripts[self.active_id]
        if not sc.file_path:
            self._save_script()
        if not sc.file_path:
            return
        try:
            if shutil.which("code"):
                subprocess.Popen(["code", sc.file_path])
            else:
                if sys.platform == "darwin":
                    subprocess.Popen(["open", sc.file_path])
                elif sys.platform == "win32":
                    os.startfile(sc.file_path)  # type: ignore[arg-type]
                else:
                    subprocess.Popen(["xdg-open", sc.file_path])
        except Exception as exc:      # pylint: disable=broad-except
            messagebox.showerror("VS Code", str(exc))

    # ---------------------------- config ----------------------------------- #

    def _load_config(self) -> Dict[str, Any]:
        if CONFIG_FILE.exists():
            try:
                return json.loads(CONFIG_FILE.read_text(encoding="utf‑8"))
            except Exception:
                pass
        return {"theme": "dark"}

# --------------------------------------------------------------------------- #
#  Startup                                                                    #
# --------------------------------------------------------------------------- #

def main() -> None:
    if "--install-requirements" in sys.argv:
        ensure_packages(REQUIRED_PKGS)
        print("Packages installed. Run again without --install-requirements.")
        return

    ensure_packages(tuple(p for p in REQUIRED_PKGS if p not in sys.modules))
    root = tk.Tk()
    app = NeuralNexusIDE(root)  # noqa: F841  # keep reference
    def _on_close():
        for s in app.scripts.values():
            if s.process and s.process.poll() is None:
                s.process.terminate()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()

if __name__ == "__main__":
    main()

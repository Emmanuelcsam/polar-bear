"""
Google‑Blockly hosted in a Tkinter HtmlFrame.  Converts workspace
→ individual .py files and the reverse.
"""
from __future__ import annotations
import json, tempfile
from pathlib import Path
from tkinterweb import HtmlFrame


_WWW = Path(__file__).with_suffix("")  # app/

class BlockCanvas(HtmlFrame):
    def __init__(self, master):
        super().__init__(master, messages_enabled=False)
        self.load_html((Path(__file__).parent / "web" / "blockly_host.html").read_text())

    # public helpers ---------------------------------------------------
    def add_block(self):                       # add blank python‑block
        self.evaluate_js("addNewPythonBlock()")

    def export_python(self) -> dict[str, str]:
        """Return {blockId: tmp_py_file_path}"""
        raw = self.evaluate_js("exportPython()")  # js → JSON string
        blocks = json.loads(raw)
        tmpdir = Path(tempfile.mkdtemp())
        out = {}
        for blk in blocks:
            p = tmpdir / f"{blk['name']}.py"
            p.write_text(blk['code'])
            out[blk['name']] = str(p)
        return out

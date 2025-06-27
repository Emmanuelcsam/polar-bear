"""
Single source of truth for the glossy‑dark theme used everywhere.
"""
from ttkbootstrap.style import Style


def apply(root) -> None:
    style = Style(theme="darkly")                      # dark palette
    style.configure(".", relief="flat", borderwidth=0)
    style.configure("TFrame", background="#121212")
    style.configure("TLabel", background="#e0e0e0")
    style.configure("TButton", padding=6)
    style.configure("TNotebook", background="#1a1a1a")
    root.option_add("*tearOff", False)

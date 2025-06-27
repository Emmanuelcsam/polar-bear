#!/usr/bin/env python3
"""
BlockNet Builder bootstrapper.
Installs any missing *required* libraries the **first** time you run it,
offers to install AI helpers, then launches the GUI.
"""

from __future__ import annotations
import sys, subprocess, importlib.util, os
from pathlib import Path

REQ = ["tkinterweb", "ttkbootstrap", "networkx", "matplotlib",
       "psutil", "numpy", "pillow", "pyyaml"]
OPT = ["openai", "beautifulsoup4", "requests"]          # for Smart‑AI
APP_DIR = Path(__file__).parent / "app"


def _missing(pkgs: list[str]) -> list[str]:
    return [p for p in pkgs if importlib.util.find_spec(p) is None]


def _pip_install(pkgs: list[str]) -> None:
    if not pkgs:
        return
    print("⏳ installing:", ", ".join(pkgs))
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", *pkgs],
        stdout=subprocess.DEVNULL
    )


def main() -> None:
    _pip_install(_missing(REQ))                        # always present
    if _missing(OPT):
        ans = input("\nEnable Smart‑AI helpers (needs openai)? [y/N] › ")
        if ans.lower().startswith("y"):
            _pip_install(_missing(OPT))

    sys.path.insert(0, str(APP_DIR))
    from builder_gui import run                        # noqa: delayed import
    run()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
auto_click_scroll.py
--------------------
Universal auto‑clicker + auto‑scroller for Ubuntu (Linux X11).

Features
~~~~~~~~
✔ Global hot‑keys (no focus required)
    • <F6>  toggle auto‑click           (left button by default)
    • <F7>  toggle auto‑scroll          (downward by default)
    • <ESC> immediate quit (failsafe)

✔ Configurable at the top of this file *or* via CLI flags:
    --click-interval   seconds between clicks      (default 0.05)
    --scroll-step      lines per scroll event      (default -3)
    --scroll-interval  seconds between scrolls     (default 0.02)
    --button           left|right|middle           (default left)

✔ Independent threads – click and scroll can run simultaneously.

✔ Clean shutdown on Ctrl‑C, SIGINT or <ESC>.

Dependencies
~~~~~~~~~~~~
$ sudo apt install python3-pip    # if pip not present
$ pip3 install pynput

Under Wayland, run your session under Xwayland *or* use xdotool instead.

Usage examples
~~~~~~~~~~~~~~
$ python3 auto_click_scroll.py                # use defaults
$ python3 auto_click_scroll.py --click-interval 0.1 --scroll-step -6
"""
import argparse
import signal
import sys
import threading
import time
from enum import Enum

from pynput import keyboard, mouse


# --------------------------------------------------------------------------- #
# Configuration defaults (can be overridden by CLI flags)
# --------------------------------------------------------------------------- #
DEFAULT_CLICK_INTERVAL = 0.05     # seconds
DEFAULT_SCROLL_INTERVAL = 0.02    # seconds
DEFAULT_SCROLL_STEP = -3          # negative = scroll down
DEFAULT_BUTTON = mouse.Button.left


# --------------------------------------------------------------------------- #
# Worker threads
# --------------------------------------------------------------------------- #
class ToggleState(Enum):
    STOPPED = 0
    RUNNING = 1


class ClickThread(threading.Thread):
    def __init__(self, button, interval):
        super().__init__(daemon=True)
        self.button = button
        self.interval = interval
        self.state = ToggleState.STOPPED
        self._mouse = mouse.Controller()

    def run(self):
        while True:
            if self.state is ToggleState.RUNNING:
                self._mouse.click(self.button)
                time.sleep(self.interval)
            else:
                time.sleep(0.1)


class ScrollThread(threading.Thread):
    def __init__(self, step, interval):
        super().__init__(daemon=True)
        self.step = step
        self.interval = interval
        self.state = ToggleState.STOPPED
        self._mouse = mouse.Controller()

    def run(self):
        while True:
            if self.state is ToggleState.RUNNING:
                self._mouse.scroll(0, self.step)
                time.sleep(self.interval)
            else:
                time.sleep(0.1)


# --------------------------------------------------------------------------- #
# Hot‑key bindings
# --------------------------------------------------------------------------- #
class HotKeys:
    """Registers global hot‑keys and ties them to worker‑thread state."""

    def __init__(self, clicker: ClickThread, scroller: ScrollThread):
        self.clicker = clicker
        self.scroller = scroller
        self._kb_listener = keyboard.GlobalHotKeys({
            '<f6>': self._toggle_click,
            '<f7>': self._toggle_scroll,
            '<esc>': self._exit_program,
        })

    # ---------- callbacks ----------
    def _toggle_click(self):
        self.clicker.state = (ToggleState.STOPPED if self.clicker.state is ToggleState.RUNNING
                              else ToggleState.RUNNING)
        print(f"[CLICK] {'ON' if self.clicker.state is ToggleState.RUNNING else 'OFF'}")

    def _toggle_scroll(self):
        self.scroller.state = (ToggleState.STOPPED if self.scroller.state is ToggleState.RUNNING
                               else ToggleState.RUNNING)
        print(f"[SCROLL] {'ON' if self.scroller.state is ToggleState.RUNNING else 'OFF'}")

    def _exit_program(self):
        print("Exiting…")
        self.clicker.state = ToggleState.STOPPED
        self.scroller.state = ToggleState.STOPPED
        self._kb_listener.stop()
        # Let main thread handle the actual sys.exit()

    # ---------- public ----------
    def start(self):
        self._kb_listener.start()

    def wait(self):
        self._kb_listener.join()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Auto click & scroll utility")
    parser.add_argument("--click-interval", type=float,
                        default=DEFAULT_CLICK_INTERVAL,
                        help="seconds between clicks")
    parser.add_argument("--scroll-interval", type=float,
                        default=DEFAULT_SCROLL_INTERVAL,
                        help="seconds between scroll events")
    parser.add_argument("--scroll-step", type=int, default=DEFAULT_SCROLL_STEP,
                        help="vertical scroll amount (positive=up, negative=down)")
    parser.add_argument("--button", choices=['left', 'right', 'middle'],
                        default='left', help="mouse button for clicking")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve button choice
    button_map = {
        'left': mouse.Button.left,
        'right': mouse.Button.right,
        'middle': mouse.Button.middle,
    }
    button = button_map[args.button]

    # Worker threads
    clicker = ClickThread(button, args.click_interval)
    scroller = ScrollThread(args.scroll_step, args.scroll_interval)
    clicker.start()
    scroller.start()

    # Hot‑key listener
    hk = HotKeys(clicker, scroller)
    hk.start()

    # Friendly banner
    print("Auto‑click/scroll is armed.")
    print("• F6  → toggle clicking")
    print("• F7  → toggle scrolling")
    print("• ESC → quit\n")

    # Ensure clean exit on Ctrl‑C
    def _sigint(_sig, _frame):
        hk._exit_program()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint)

    # Block until user quits
    hk.wait()


if __name__ == "__main__":
    try:
        main()
    finally:
        print("Goodbye!")

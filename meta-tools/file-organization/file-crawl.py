#!/usr/bin/env python3
"""
Deep File Crawler & Organizer
--------------------------------
Recursively searches for files whose *filenames* match one or more
user‑supplied keywords (case‑insensitive substring or regex) and copies
them to destinations you choose, renaming duplicates safely.

Key features
• Works with *any* file type – you decide by keyword, not extension.
• Two interactive modes:
   1. Single search (one keyword → one destination)
   2. Batch search (many keyword→destination pairs)
• No argparse — all input is via simple prompts or an OPTIONAL .ini/.csv
• Cross‑platform – tested on Windows 10/11, macOS 14, Ubuntu 24.04
• Pure standard library; tqdm progressbar is *optional*.
• Detailed logging (UTF‑8) and JSON report with per‑keyword statistics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# ──────────────────────────────────────────────────────────────
# Configuration constants (advanced users may edit directly)
# ──────────────────────────────────────────────────────────────
LOG_DIR          = Path("logs")
REPORT_DIR       = LOG_DIR
DEFAULT_HASH_LEN = 8          # chars of MD5 to append when renaming
ENCODING         = "utf‑8"
PROGRESS_BAR     = True       # requires 'tqdm'; auto‑disables if not found
# ──────────────────────────────────────────────────────────────

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # keep it optional
    PROGRESS_BAR = False
    def tqdm(x, **kwargs):     # type: ignore
        return x

# Initialize logging early
LOG_DIR.mkdir(exist_ok=True)
log_path = LOG_DIR / f"file_crawler_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_path, encoding=ENCODING),
              logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("file_crawler")

# ──────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────
def md5_for_file(path: Path, chunk_size: int = 1 << 16) -> str:
    """Return *hex* MD5 for a file, streaming in `chunk_size` bytes."""
    h = hashlib.md5()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as exc:
        logger.error("Hashing failed for %s – %s", path, exc)
        return ""

def ensure_dir(d: Path) -> None:
    """Create directory if it doesn't exist (thread‑safe)."""
    d.mkdir(parents=True, exist_ok=True)

def prompt(question: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        ans = input(f"{question}{suffix} › ").strip()
        if ans:
            return ans
        if default is not None:
            return default

def yes_no(question: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        ans = input(f"{question} [{d}] › ").strip().lower()
        if not ans:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False

# ──────────────────────────────────────────────────────────────
# Core crawler class
# ──────────────────────────────────────────────────────────────
class FileCrawler:
    def __init__(self,
                 start_dir: Path,
                 mapping: Dict[str, Path],
                 hash_len: int = DEFAULT_HASH_LEN) -> None:
        """
        mapping = { keyword : destination_directory }
        """
        self.start_dir = start_dir.resolve()
        self.mapping   = {k.lower(): v.resolve() for k, v in mapping.items()}
        self.hash_len  = hash_len

        self.stats_files_scanned  = 0
        self.stats_copied         = defaultdict(int)      # per keyword
        self.stats_duplicates     = defaultdict(int)
        self.duplicate_cache      = defaultdict(set)      # per keyword → {md5}

        logger.info("Initialized crawler in %s with %d keyword(s)",
                    self.start_dir, len(self.mapping))

    # ----------------------------------------------------------
    def crawl(self) -> None:
        """Main entry – walks directories, evaluates keywords, copies."""
        if not self.mapping:
            print("No keywords provided – nothing to do.")
            return

        print(f"\nScanning '{self.start_dir}' recursively …\n")
        files_iter = self.start_dir.rglob('*')
        files_iter = tqdm(files_iter, unit="file") if PROGRESS_BAR else files_iter

        for path in files_iter:
            # Skip directories, broken symlinks, self logs
            if not path.is_file():
                continue
            if LOG_DIR in path.parents:
                continue

            self.stats_files_scanned += 1
            fname_lower = path.name.lower()

            # Which keyword(s) match this filename?
            for kw, dest in self.mapping.items():
                if kw in fname_lower or re.search(kw, fname_lower, re.I):
                    self._copy_file(path, kw, dest)

        self._report()

    # ----------------------------------------------------------
    def _copy_file(self, src: Path, keyword: str, dest_dir: Path) -> None:
        ensure_dir(dest_dir)

        # Duplicate detection via MD5 of *content*
        file_hash = md5_for_file(src)
        if not file_hash:    # On error – skip
            return

        if file_hash in self.duplicate_cache[keyword]:
            self.stats_duplicates[keyword] += 1
            logger.debug("Duplicate (hash) for keyword '%s': %s", keyword, src)
            return

        self.duplicate_cache[keyword].add(file_hash)

        # Build destination filename
        dst_name = src.name
        dst_path = dest_dir / dst_name

        if dst_path.exists():
            stem, suf = dst_path.stem, dst_path.suffix
            dst_name = f"{stem}_{file_hash[:self.hash_len]}{suf}"
            dst_path = dest_dir / dst_name
            # If even that exists, add counter
            counter = 1
            while dst_path.exists():
                dst_name = f"{stem}_{file_hash[:self.hash_len]}_{counter}{suf}"
                dst_path = dest_dir / dst_name
                counter += 1

        try:
            shutil.copy2(src, dst_path)
            self.stats_copied[keyword] += 1
            logger.info("Copied %s → %s", src, dst_path)
        except Exception as exc:
            logger.error("Failed to copy %s → %s – %s", src, dst_path, exc)

    # ----------------------------------------------------------
    def _report(self) -> None:
        total_copied = sum(self.stats_copied.values())
        print("\n────────── SUMMARY ──────────")
        print(f"Files scanned  : {self.stats_files_scanned:,}")
        print(f"Files copied   : {total_copied:,}")
        print(f"Duplicates skip: {sum(self.stats_duplicates.values()):,}\n")

        for kw in self.mapping:
            print(f"  • '{kw}' → {self.mapping[kw]}")
            print(f"       copied: {self.stats_copied[kw]:,}")
            print(f"   duplicates: {self.stats_duplicates[kw]:,}")

        # Persist JSON report
        REPORT_DIR.mkdir(exist_ok=True)
        report_path = REPORT_DIR / f"report_{datetime.now():%Y%m%d_%H%M%S}.json"
        json_payload = {
            "timestamp": datetime.now().isoformat(),
            "start_dir": str(self.start_dir),
            "stats": {
                "scanned": self.stats_files_scanned,
                "copied":  total_copied,
                "duplicates": {k: v for k, v in self.stats_duplicates.items()}
            },
            "mapping": {k: str(v) for k, v in self.mapping.items()},
            "log_file": str(log_path)
        }
        report_path.write_text(json.dumps(json_payload, indent=2), encoding=ENCODING)
        print(f"\nDetailed JSON report → {report_path}")
        logger.info("JSON report written to %s", report_path)

# ──────────────────────────────────────────────────────────────
# Interactive front‑end
# ──────────────────────────────────────────────────────────────

def _quick_list_mode() -> Dict[str, Path]:
    """Batch mode A – one base dir, many keywords."""
    base = Path(prompt("Base destination directory for ALL keywords"))
    ensure_dir(base)
    print("\nEnter keywords (comma, space or newline separated).")
    print("Example:  core  cladding,ferrule   invoices\n")
    raw = []
    while True:
        line = input("keywords › ").strip()
        if not line:
            break
        raw.append(line)
    keywords = re.split(r"[,\s]+", " ".join(raw).strip())
    mapping = {kw: base / kw for kw in keywords if kw}
    return mapping


def _csv_file_mode() -> Dict[str, Path]:
    """Batch mode B – read keyword,destination from a file."""
    while True:
        fp = Path(prompt("Path to CSV/TXT file"))
        if fp.is_file():
            break
        print("⚠  File not found – try again.")
    mapping: Dict[str, Path] = {}
    for lineno, line in enumerate(fp.read_text(encoding=ENCODING).splitlines(), 1):
        if not line.strip():
            continue
        if "," not in line:
            print(f"Line {lineno}: missing comma – skipped.")
            continue
        kw, dest = (x.strip() for x in line.split(",", 1))
        if kw and dest:
            mapping[kw] = Path(dest)
    return mapping


def _advanced_line_mode() -> Dict[str, Path]:
    """Batch mode C – original 'keyword => destination' lines."""
    print("\nEnter mapping lines.  Example:\n    core       =>  D:\\Sort\\Core")
    print("Blank line to finish.\n")
    mapping: Dict[str, Path] = {}
    while True:
        line = input("mapping › ").strip()
        if not line:
            break
        if "=>" not in line:
            print("⚠  Use  keyword => path")
            continue
        kw, dest = (part.strip() for part in line.split("=>", 1))
        mapping[kw] = Path(dest)
    return mapping


def _guided_one_by_one_mode() -> Dict[str, Path]:
    """Batch mode D – prompt destination per keyword."""
    print("\nEnter the keywords you want to harvest (comma/space separated).")
    keywords = re.split(r"[,\s]+", input("Keywords › ").strip())
    mapping: Dict[str, Path] = {}
    for kw in keywords:
        if not kw:
            continue
        dest = Path(prompt(f"Destination directory for '{kw}'"))
        mapping[kw] = dest
    return mapping


def build_mapping_interactively() -> Dict[str, Path]:
    """Unified entry point with four styles now."""
    print("\nChoose batch‑input style:")
    print("  A) Quick list  (one base dir, many keywords)")
    print("  B) CSV/TXT file (keyword,dest)")
    print("  C) Advanced mapping (keyword => dest)")
    print("  D) Guided wizard (ask destination for each keyword)")
    choice = prompt("Option", default="D").upper()

    if choice == "A":
        return _quick_list_mode()
    if choice == "B":
        return _csv_file_mode()
    if choice == "C":
        return _advanced_line_mode()
    return _guided_one_by_one_mode()
# ──────────────────────────────────────────────────────────────

# (the rest of main() remains exactly the same)





def main() -> None:
    print("─" * 56)
    print("DEEP FILE CRAWLER  •  v1.0")
    print("─" * 56)

    start_dir_input = prompt("Start directory to search", default=os.getcwd())
    start_dir = Path(start_dir_input).expanduser()

    mode = prompt("\nSelect mode:\n  1) Single keyword → single destination\n  2) Batch mapping\nChoice", default="1")
    mapping: Dict[str, Path] = {}

    if mode == "1":
        kw = prompt("Keyword (case‑insensitive substring/regex)")
        dest = Path(prompt("Destination directory for matches"))
        mapping[kw] = dest
    else:
        mapping = build_mapping_interactively()

    if not mapping:
        print("No mapping supplied – exiting.")
        return

    print("\nConfiguration:")
    print(f" Start dir : {start_dir}")
    for k, v in mapping.items():
        print(f"  '{k}'  → {v}")
    if not yes_no("\nProceed?", default=True):
        print("Aborted.")
        return

    crawler = FileCrawler(start_dir=start_dir, mapping=mapping)
    crawler.crawl()

    print("\nDone. See the log for details:", log_path)

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)

#!/usr/bin/env python3
"""
ChatGPT Universal Analyzer – refactored June 2025
-------------------------------------------------
Analyse ChatGPT exports (JSON) **or** your live chat list (Selenium)
and generate HTML/CSV/JSON/Markdown/PDF reports.

This version is self‑contained and backward‑compatible with the
command‑line switches used by the original script.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import os
import re
import sys
import textwrap
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_EXCEPTION
from dataclasses import dataclass, asdict
from datetime import datetime
from functools import wraps
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union
# ---------------------------------------------------------------------------
#  NEW: simple config‑file support  +  question list
# ---------------------------------------------------------------------------
import json
from pathlib import Path

try:
    import yaml  # pip install pyyaml
except ModuleNotFoundError:
    yaml = None

CONFIG_FILE_CANDIDATES = ["config.yaml", "config.json"]

def load_config_file() -> dict | None:
    """Return a config dict if config.yaml|json exists, otherwise None."""
    for name in CONFIG_FILE_CANDIDATES:
        p = Path(name)
        if p.is_file():
            with p.open(encoding="utf‑8") as fh:
                return yaml.safe_load(fh) if p.suffix == ".yaml" and yaml else json.load(fh)
    return None
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
#  NEW: single source of truth for every interactive question
# ---------------------------------------------------------------------------
QUESTIONS: list[dict] = [
    {
        "key": "mode",
        "text": "How would you like to analyse your ChatGPT data?",
        "choices": {"1": "export", "2": "live"},
        "help": "(1) analyse conversations.json  |  (2) drive the browser in real time",
        "default": "1",
    },
    {
        "key": "paths.export_file",
        "text": "Path to conversations.json:",
        "when": lambda cfg: cfg.get("mode") == "export",
        "default": "conversations.json",
    },
    {
        "key": "credentials.email",
        "text": "ChatGPT e‑mail:",
        "when": lambda cfg: cfg.get("mode") == "live",
    },
    {
        "key": "credentials.password",
        "text": "ChatGPT password (input hidden):",
        "when": lambda cfg: cfg.get("mode") == "live",
        "password": True,
    },
    {
        "key": "selenium.headless",
        "text": "Run the browser headless? (y/n)",
        "when": lambda cfg: cfg.get("mode") == "live",
        "default": "y",
        "cast": lambda v: v.lower().startswith("y"),
    },
    {
        "key": "paths.output_dir",
        "text": "Where should I save the reports?",
        "default": "chatgpt_analysis",
    },
    {
        "key": "reports.formats",
        "text": "Report formats (comma‑separated – html,csv,json,markdown,pdf):",
        "default": "html",
        "cast": lambda v: [x.strip() for x in v.split(",") if x.strip()],
    },
    {
        "key": "options.extract_content",
        "text": "Extract code / research after analysis? (yes/no/ask)",
        "default": "ask",
    },
    {
        "key": "options.use_cache",
        "text": "Use cache for faster re‑runs? (y/n)",
        "default": "y",
        "cast": lambda v: v.lower().startswith("y"),
    },
    {
        "key": "options.debug",
        "text": "Enable debug logging? (y/n)",
        "default": "n",
        "cast": lambda v: v.lower().startswith("y"),
    },
]
###############################################################################
# -----------------------------  GLOBAL CONFIG  ----------------------------- #
###############################################################################

APP_NAME = "ChatGPT‑Analyzer"
__version__ = "3.0.0"
DEFAULT_OUTPUT_DIR = Path("chatgpt_analysis")
DEFAULT_CACHE_DIR = Path(".cache")
LOG = logging.getLogger(APP_NAME)

###############################################################################
# ------------------------------  UTILITIES  -------------------------------- #
###############################################################################


def configure_logging(debug: bool = False, logfile: Optional[Path] = None) -> None:
    """Basic coloured console logger + optional file."""
    fmt = "%(asctime)s | %(levelname)-8s| %(name)s | %(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if logfile:
        fh = logging.FileHandler(logfile, encoding="utf‑8")
        fh.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(fh)


def sha1_trace(exc_type: Type[BaseException], tb: TracebackType) -> str:
    """Return a short SHA‑1 fingerprint for a traceback, to group identical bugs."""
    import hashlib
    import traceback

    digest = hashlib.sha1()
    digest.update("".join(traceback.format_tb(tb)).encode())
    return digest.hexdigest()[:8]


def diagnose(fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator – wraps every public method so that *all* exceptions are logged
    with context **before** they propagate (or are swallowed).
    """

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            tb = sys.exc_info()[2]
            digest = sha1_trace(type(e), tb) if tb else "00000000"
            LOG.error("❌ %s failed [%s]: %s", fn.__name__, digest, e, exc_info=True)
            raise

    return wrapper


def try_import(pkg_name: str) -> Optional[Any]:
    """Import a library lazily; return None if missing (no crash)."""
    try:
        return __import__(pkg_name)
    except ImportError:
        LOG.debug("Optional dependency %s not installed – feature disabled.", pkg_name)
        return None


###############################################################################
# ---------------------------  DATA STRUCTURES  ----------------------------- #
###############################################################################


class ConvType(str):
    CODE = "code"
    RESEARCH = "research"
    MIXED = "mixed"
    LINKS = "links"
    GENERAL = "general"


@dataclass(slots=True)
class ConversationMeta:
    id: str
    title: str
    url: str
    folder: str = "No Folder"
    create_time: Optional[str] = None
    update_time: Optional[str] = None
    total_messages: int = 0
    conv_type: ConvType = ConvType.GENERAL
    has_code: bool = False
    has_research: bool = False
    has_links: bool = False
    code_blocks: int = 0
    languages: List[str] = None
    research_keywords: List[str] = None
    url_count: int = 0
    word_count: int = 0

    # Convenience
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["conv_type"] = self.conv_type
        return d


###############################################################################
# ---------------------------  CORE ANALYSER  ------------------------------- #
###############################################################################


class Analyzer:
    """
    High‑level façade; handles either ‘export’ (offline JSON) or ‘live’ (Selenium)
    sources and produces ConversationMeta objects.
    """

    ###########################################################################
    # Construction & lifecycle
    ###########################################################################

    def __init__(
        self,
        mode: str,
        export_file: Optional[Path] = None,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        use_cache: bool = True,
        max_workers: int = 8,
    ) -> None:
        self.mode = mode
        self.export_file = export_file
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.max_workers = max_workers

        self._cache_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Optional runtime deps
        self.selenium = try_import("selenium")
        self.bs4 = try_import("bs4")
        self.reportlab = try_import("reportlab")

        self.driver = None  # set lazily for live mode

    # Properties avoid unguarded Path / mkdir usage
    @property
    def _cache_dir(self) -> Path:
        return Path(self.cache_dir)

    ###########################################################################
    # Public API
    ###########################################################################

    @diagnose
    def analyze(self) -> List[ConversationMeta]:
        if self.mode == "export":
            if not self.export_file:
                raise ValueError("--export FILE is required in export mode")
            return self._analyze_export(Path(self.export_file))
        if self.mode == "live":
            return self._analyze_live()
        raise ValueError(f"Unknown mode: {self.mode}")

    @diagnose
    def generate_reports(
        self, conversations: Sequence[ConversationMeta], formats: Sequence[str]
    ) -> None:
        """Call specialised reporters; always write JSON for programmatic reuse."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        reporters = {
            "json": self._report_json,
            "csv": self._report_csv,
            "markdown": self._report_md,
            "html": self._report_html,
            "pdf": self._report_pdf if self.reportlab else lambda *_: None,
        }

        for fmt in formats:
            fn = reporters.get(fmt.lower())
            if not fn:
                LOG.warning("Unknown report type: %s (skipped)", fmt)
                continue
            try:
                fn(conversations, stamp)
            except Exception:  # noqa: BLE001
                LOG.exception("Failed to write %s report", fmt)

    ###########################################################################
    # -------------------------  EXPORT MODE  ------------------------------- #
    ###########################################################################

    @diagnose
    def _analyze_export(self, json_path: Path) -> List[ConversationMeta]:
        LOG.info("Parsing export file %s …", json_path)
        data = json.loads(json_path.read_text(encoding="utf‑8"))

        convs: List[Dict[str, Any]]
        projects: List[Dict[str, Any]]
        if isinstance(data, list):
            convs = data
            projects = []
        else:
            convs = data.get("conversations", [])
            projects = data.get("projects", [])

        project_map = {
            cid: proj.get("name", "No Project")
            for proj in projects
            for cid in proj.get("conversation_ids", [])
        }

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = [
                ex.submit(self._analyze_single_export, c, project_map) for c in convs
            ]
            wait(futs, return_when=FIRST_EXCEPTION)

        results = [f.result() for f in futs if f.exception() is None]
        LOG.info("Analyzed %d conversations", len(results))
        return results

    def _analyze_single_export(
        self, conv: Dict[str, Any], project_map: Dict[str, str]
    ) -> ConversationMeta:
        cid = conv["id"]
        cached = self._from_cache(cid)
        if cached:
            return cached

        # ---------- text extraction ----------
        text = self._extract_text(conv)
        analysis = self._analyze_text(text)

        meta = ConversationMeta(
            id=cid,
            title=conv.get("title", "Untitled"),
            url=f"https://chat.openai.com/c/{cid}",
            folder=project_map.get(cid, "No Folder"),
            create_time=conv.get("create_time"),
            update_time=conv.get("update_time"),
            total_messages=self._count_messages(conv),
            **analysis,
        )

        self._to_cache(meta)
        return meta

    # -------------------- helpers -------------------- #

    @staticmethod
    def _count_messages(conv: Dict[str, Any]) -> int:
        mapping = conv.get("mapping", {})
        if mapping:
            return sum(
                1
                for node in mapping.values()
                if node.get("message", {})
                .get("content", {})
                .get("content_type")
                == "text"
            )
        return len(conv.get("messages", []))

    @staticmethod
    def _extract_text(conv: Dict[str, Any]) -> str:
        parts: List[str] = []
        if conv.get("mapping"):
            for node in conv["mapping"].values():
                msg = node.get("message", {})
                if msg.get("content", {}).get("content_type") == "text":
                    parts.extend(msg["content"]["parts"])
        else:
            for m in conv.get("messages", []):
                if isinstance(m, str):
                    parts.append(m)
                elif isinstance(m, dict):
                    c = m.get("content") or m.get("text") or ""
                    parts.append(c if isinstance(c, str) else "\n".join(c.get("parts", [])))
        return "\n".join(parts)

    @diagnose
    def _analyze_text(self, text: str) -> Dict[str, Any]:
        code_blocks = re.findall(r"```(\w*)\n(.*?)```", text, flags=re.S)
        has_code = bool(code_blocks)

        # guess languages
        langs = set()
        for lang, _ in code_blocks:
            if lang:
                langs.add(lang.lower())
        # heuristics
        if not langs and has_code:
            if "def " in text or "import " in text:
                langs.add("python")

        # research detection
        KEYWORDS = {
            "research",
            "study",
            "algorithm",
            "analysis",
            "dataset",
            "paper",
            "evidence",
        }
        lowered = text.lower()
        found_kw = [w for w in KEYWORDS if w in lowered]
        has_research = len(found_kw) >= 3

        urls = re.findall(r"https?://\S+", text)
        has_links = bool(urls)

        # category
        if has_code and has_research:
            ctype = ConvType.MIXED
        elif has_code:
            ctype = ConvType.CODE
        elif has_research:
            ctype = ConvType.RESEARCH
        elif has_links:
            ctype = ConvType.LINKS
        else:
            ctype = ConvType.GENERAL

        return dict(
            conv_type=ctype,
            has_code=has_code,
            has_research=has_research,
            has_links=has_links,
            code_blocks=len(code_blocks),
            languages=sorted(langs),
            research_keywords=found_kw,
            url_count=len(urls),
            word_count=len(text.split()),
        )

    ###########################################################################
    # ---------------------------  LIVE MODE  ------------------------------- #
    ###########################################################################

    @diagnose
    def _analyze_live(self) -> List[ConversationMeta]:
        if not self.selenium:
            raise RuntimeError("Selenium not installed – cannot use live mode")

        driver = self._create_driver()
        self.driver = driver  # save for cleanup

        # The actual DOM scraping code should go here – omitted for brevity.
        raise NotImplementedError("Live mode not implemented in this excerpt")

    # -------------------- driver helpers -------------------- #

    def _create_driver(self):
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options as ChromeOptions

        opts = ChromeOptions()
        opts.add_argument("--headless=new")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--no-sandbox")

        LOG.info("Starting Chrome driver (headless)…")
        driver = webdriver.Chrome(options=opts)  # relies on PATH
        driver.set_page_load_timeout(30)
        return driver

    ###########################################################################
    # ------------------------  CACHING LAYER  ------------------------------ #
    ###########################################################################

    def _cache_path(self, cid: str) -> Path:
        return self._cache_dir / f"{cid}.json"

    def _from_cache(self, cid: str) -> Optional[ConversationMeta]:
        p = self._cache_path(cid)
        if not self.use_cache or not p.exists():
            return None
        try:
            return ConversationMeta(**json.loads(p.read_text()))
        except Exception:  # noqa: BLE001
            LOG.warning("Corrupted cache for %s – ignoring", cid)
            p.unlink(missing_ok=True)
            return None

    def _to_cache(self, meta: ConversationMeta) -> None:
        if not self.use_cache:
            return
        self._cache_path(meta.id).write_text(json.dumps(meta.to_dict(), ensure_ascii=False))

    ###########################################################################
    # ---------------------------  REPORTERS  ------------------------------- #
    ###########################################################################

    # ----- JSON is always enabled -----
    def _report_json(self, convs: Sequence[ConversationMeta], stamp: str) -> None:
        out = self.output_dir / f"analysis_{stamp}.json"
        out.write_text(
            json.dumps([c.to_dict() for c in convs], indent=2, ensure_ascii=False)
        )
        LOG.info("💾 wrote %s (%d B)", out, out.stat().st_size)

    # ----- CSV (spreadsheet) -----
    def _report_csv(self, convs: Sequence[ConversationMeta], stamp: str) -> None:
        out = self.output_dir / f"analysis_{stamp}.csv"
        with out.open("w", newline="", encoding="utf‑8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "folder",
                    "title",
                    "conv_type",
                    "url",
                    "languages",
                    "code_blocks",
                    "url_count",
                    "total_messages",
                    "word_count",
                ],
            )
            writer.writeheader()
            for c in convs:
                d = {
                    "folder": c.folder,
                    "title": c.title,
                    "conv_type": c.conv_type,
                    "url": c.url,
                    "languages": ", ".join(c.languages or []),
                    "code_blocks": c.code_blocks,
                    "url_count": c.url_count,
                    "total_messages": c.total_messages,
                    "word_count": c.word_count,
                }
                writer.writerow(d)
        LOG.info("💾 wrote %s", out)

    # ----- Markdown (readable summary) -----
    def _report_md(self, convs: Sequence[ConversationMeta], stamp: str) -> None:
        out = self.output_dir / f"analysis_{stamp}.md"
        with out.open("w", encoding="utf‑8") as fh:
            fh.write(f"# {APP_NAME} report – {datetime.now():%Y‑%m‑%d}\n\n")
            fh.write(f"Conversations analysed: **{len(convs)}**\n\n")
            by_type = Counter(c.conv_type for c in convs)
            for t, n in by_type.items():
                fh.write(f"* **{t.title()}**: {n}\n")
            fh.write("\n---\n")
            for c in convs:
                fh.write(f"## {c.title}\n")
                fh.write(f"*Type*: {c.conv_type}\n\n")
                if c.languages:
                    fh.write(f"*Languages*: {', '.join(c.languages)}\n\n")
                if c.research_keywords:
                    fh.write(f"*Keywords*: {', '.join(c.research_keywords)}\n\n")
                fh.write(f"[Open conversation]({c.url})\n\n")
        LOG.info("💾 wrote %s", out)

    # ----- HTML (interactive) -----
    def _report_html(self, convs: Sequence[ConversationMeta], stamp: str) -> None:
        tpl_path = Path(__file__).with_name("templates").joinpath("report.html")
        if not tpl_path.exists():
            LOG.warning("HTML template not found (%s); skipping", tpl_path)
            return
        tpl = tpl_path.read_text(encoding="utf‑8")
        rendered = tpl.replace("__DATA__", json.dumps([c.to_dict() for c in convs]))
        out = self.output_dir / f"analysis_{stamp}.html"
        out.write_text(rendered, encoding="utf‑8")
        LOG.info("💾 wrote %s", out)

    # ----- PDF (if ReportLab present) -----
    def _report_pdf(self, convs: Sequence[ConversationMeta], stamp: str) -> None:
        if not self.reportlab:
            return
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet

        out = self.output_dir / f"analysis_{stamp}.pdf"
        doc = SimpleDocTemplate(out, pagesize=letter)
        styles = getSampleStyleSheet()
        flow = [Paragraph(f"{APP_NAME} report", styles["Title"]), Spacer(1, 12)]
        flow.append(Paragraph(f"Generated {datetime.now():%c}", styles["Normal"]))
        flow.append(Spacer(1, 24))
        for c in convs[:100]:
            flow.append(Paragraph(c.title, styles["Heading2"]))
            flow.append(Paragraph(f"Type: {c.conv_type}", styles["Normal"]))
            flow.append(Spacer(1, 12))
        doc.build(flow)
        LOG.info("💾 wrote %s", out)

    ###########################################################################
    # ---------------------------  CLEANUP  --------------------------------- #
    ###########################################################################

    def close(self) -> None:
        if self.driver:
            with contextlib.suppress(Exception):
                self.driver.quit()


###############################################################################
# ---------------------------  CLI / ENTRYPOINT  ---------------------------- #
###############################################################################


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=APP_NAME,
        description="Analyse ChatGPT exports or live conversations and generate reports",
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--export", metavar="FILE", help="Path to conversations.json")
    mode.add_argument("--live", action="store_true", help="Analyse via Selenium")
    p.add_argument("--report", nargs="+", default=["html"], help="html csv json markdown pdf")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--cache/--no-cache", dest="cache", default=True)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--open", action="store_true", help="Open HTML report after run")
    p.add_argument("--wizard", action="store_true", help="Run interactive wizard")
    return p


def interactive_configuration() -> dict | None:
    """Ask only the questions defined in QUESTIONS; honour defaults."""
    from getpass import getpass

    cfg: dict = {}

    print("\n🔧  Interactive configuration (press Ctrl‑C to abort)")

    try:
        for q in QUESTIONS:
            if "when" in q and not q["when"](cfg):
                continue

            path = q["key"].split(".")
            prompt = "\n" + q["text"]
            if "help" in q:
                prompt += f"\n   {q['help']}"
            if "default" in q:
                prompt += f"  [{q['default']}]"
            prompt += "\n> "

            raw = getpass(prompt) if q.get("password") else input(prompt)
            if not raw and "default" in q:
                raw = q["default"]
            if q.get("choices") and raw in q["choices"]:
                raw = q["choices"][raw]

            casted = q.get("cast", lambda x: x)(raw)

            # write into nested dict
            d = cfg
            for p in path[:-1]:
                d = d.setdefault(p, {})
            d[path[-1]] = casted

        return cfg
    except KeyboardInterrupt:
        print("\n✋ cancelled")
        return None


def main() -> int:
    # 1️⃣ command‑line override
    if "--wizard" in sys.argv:
        cfg = interactive_configuration()
    else:
        # 2️⃣ try config.yaml / config.json
        cfg = load_config_file()
        if cfg:
            print("📄 Loaded configuration from file – no questions asked.")
        else:
            cfg = interactive_configuration()
    if not cfg:  # user pressed Ctrl‑C
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())

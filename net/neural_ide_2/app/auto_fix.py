from __future__ import annotations
import difflib, pathlib


class AutoFix:
    def __init__(self, file_path: str | pathlib.Path):
        self.file = pathlib.Path(file_path)

    def apply(self, new_code: str) -> None:
        if not new_code.strip():
            return
        old = self.file.read_text().splitlines(keepends=True)
        new = new_code.splitlines(keepends=True)
        patch = "".join(difflib.unified_diff(old, new, lineterm=""))
        # safety – only overwrite if more than 50 % lines changed
        if len(patch) > 0.5 * len("".join(old)):
            backup = self.file.with_suffix(".bak")
            backup.write_text("".join(old))
            self.file.write_text(new_code)
            print(f"Auto‑Fix applied → backup saved as {backup}")
        else:
            print("Auto‑Fix declined – patch too small.")

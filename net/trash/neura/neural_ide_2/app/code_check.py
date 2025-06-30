import subprocess, shutil, sys, pathlib, textwrap


class CodeCheck:
    """Very thin wrapper around pylint + flake8 used by the GUI."""
    def __init__(self, file_path: str | pathlib.Path):
        self.file_path = str(file_path)

    # -------- public helpers -----------------------------------------
    def quick_check(self) -> None:
        for tool in ("pylint", "flake8"):
            exe = shutil.which(tool)
            if not exe:
                continue
            res = subprocess.run([exe, self.file_path],
                                 capture_output=True, text=True)
            if res.stdout.strip():
                print(textwrap.dedent(f"""
                ── {tool.upper()} {self.file_path} ─────────────
                {res.stdout}
                """))

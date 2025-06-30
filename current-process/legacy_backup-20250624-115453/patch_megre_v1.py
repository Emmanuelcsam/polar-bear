#!/usr/bin/env python3
"""
patch_merge_v1.py
========================================================
Incremental *merging* patch – keeps every line of the
original codebase but layers the requested improvements
on top (RAM‑only processing, dynamic ndarray segmentation,
dual folder spelling, interactive RAM switch).

Run this script once from the repository root:

    python patch_merge_v1.py

You will see a   [backup] ...   line for each file moved
into the timestamped backup folder and a   [patched] ...
line for every successful in‑place modification.

Nothing is permanently deleted – just remove the new
sections or copy files back from legacy_backup‑*/ to undo.
========================================================
"""
from __future__ import annotations
import re, shutil, datetime, sys, os, stat
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TS   = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
BK   = ROOT / f"legacy_backup-{TS}"
BK.mkdir(exist_ok=True)

TARGETS = ["process.py", "separation.py", "app.py"]   # detection.py unchanged this round

# --------------------------------------------------------------------------- #
# 0. Helpers
# --------------------------------------------------------------------------- #
def backup(fname: str):
    p = ROOT / fname
    if p.exists():
        shutil.copy2(p, BK / fname)
        print(f"[backup] {fname}")
    else:
        print(f"[skip  ] {fname} not found – skipping")

def write(p: Path, text: str):
    p.write_text(text, encoding="utf-8")
    p.chmod(p.stat().st_mode | stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

def inject_after(pattern: str, addition: str, text: str, *, once=True) -> str:
    """Regex‑look for `pattern` and insert `addition` **after** the first/each match."""
    def repl(m):
        return m.group(0) + addition
    flags = 0 if once else re.DOTALL
    return re.sub(pattern, repl, text, count=1 if once else 0, flags=flags)

# --------------------------------------------------------------------------- #
# 1. Patch process.py  (add RAM‑only branch + return value)
# --------------------------------------------------------------------------- #
fname = "process.py"; backup(fname)
p = ROOT / fname
src = p.read_text()

# (a) global flag
if "FIBER_RAM_ONLY" not in src:
    head_inject = (
        "\n# === Patched by patch_merge_v1 ===\n"
        "RAM_ONLY_ENV = os.getenv('FIBER_RAM_ONLY', '0').lower() in ('1','true','yes','y')\n"
    )
    src = inject_after(r"import [^\n]*cv2[^\n]*", head_inject, src)

# (b) re‑signature + images_dict + return
if "save_intermediate" not in src:
    # change def line
    src = re.sub(
        r"def reimagine_image\(([^)]*)\):",
        r"def reimagine_image(\1, save_intermediate: bool | None = None):",
        src, count=1
    )
    # initialise save_intermediate + dict just after parameter check block
    starter = (
        "\n    # === Patched by patch_merge_v1 ===\n"
        "    if save_intermediate is None:\n"
        "        save_intermediate = not RAM_ONLY_ENV\n"
        "    images_dict = {}\n"
    )
    src = inject_after(r"# Read the image[^\n]*\n[^\n]*img = cv2.imread[^\n]*\n", starter, src)

    # patch save_image helper
    src = re.sub(
        r"def save_image\(name, image\):[^\n]*\n[^\n]*cv2\.imwrite",
        (
            "def save_image(name, image):\n"
            "        # Patched helper – always keep a RAM copy\n"
            "        images_dict[name] = image\n"
            "        if save_intermediate:\n"
            "            cv2.imwrite(os.path.join(output_folder, f\"{name}.jpg\"), image)"
        ),
        src, count=1
    )

    # add return at the very end
    if "return images_dict" not in src:
        src = inject_after(r"print\(f\"All .* saved[^\n]*\n", "\n    return images_dict\n", src)

write(p, src); print(f"[patched] {fname}")

# --------------------------------------------------------------------------- #
# 2. Patch separation.py  (add ndarray entry + dual folder spelling)
# --------------------------------------------------------------------------- #
fname = "separation.py"; backup(fname)
p = ROOT / fname
src = p.read_text()

# (a) dual directory spelling (zones_methods / zone_methods)
if "zones_methods" in src and "dual_dir_support" not in src:
    dual_dir = (
        "\n# === Patched by patch_merge_v1 (dual_dir_support) ===\n"
        "try:\n"
        "    if not self.methods_dir.is_dir():\n"
        "        alt = Path(str(self.methods_dir).replace('zone_methods','zones_methods'))\n"
        "        if alt.is_dir():\n"
        "            self.logger.info(f\"Using alternate methods directory: {alt}\")\n"
        "            self.methods_dir = alt\n"
        "except AttributeError:\n"
        "    pass\n"
    )
    src = inject_after(r"def __init__\([^\)]*\):", dual_dir, src)

# (b) add process_ndarray helper if absent
if "def process_ndarray" not in src:
    helper = (
        "\n    # === Patched by patch_merge_v1 (process_ndarray) ===\n"
        "    def process_ndarray(self, image_array, virt_name: str, output_dir_str: str):\n"
        "        \"\"\"Light helper: pipes an *in‑RAM* numpy image through the existing\n"
        "        process_image() code path without polluting the working dir.\"\"\"\n"
        "        import cv2, tempfile, numpy as _np\n"
        "        from pathlib import Path as _P\n"
        "        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)\n"
        "        cv2.imwrite(tmp.name, image_array)\n"
        "        try:\n"
        "            res = self.process_image(_P(tmp.name), output_dir_str)\n"
        "            if res and 'saved_regions' in res:\n"
        "                # stamp the virtual tag onto each path for traceability\n"
        "                res['virtual_origin'] = virt_name\n"
        "            return res\n"
        "        finally:\n"
        "            os.unlink(tmp.name)\n"
    )
    src = inject_after(r"class UnifiedSegmentationSystem[^{]*{", helper, src)

write(p, src); print(f"[patched] {fname}")

# --------------------------------------------------------------------------- #
# 3. Patch app.py  (interactive RAM question + use ndarray flow)
# --------------------------------------------------------------------------- #
fname = "app.py"; backup(fname)
p = ROOT / fname
src = p.read_text()

# (a) interactive question once at start of main()
if "FIBER_RAM_ONLY" not in src:
    ram_q = (
        "\n    # === Patched by patch_merge_v1 ===\n"
        "    if 'FIBER_RAM_ONLY' not in os.environ:\n"
        "        ans = input('Run in RAM‑only mode (no intermediate files)? [y/N]: ').strip().lower()\n"
        "        if ans in ('y','yes'):\n"
        "            os.environ['FIBER_RAM_ONLY'] = '1'\n"
    )
    src = inject_after(r"def main\([^\)]*\):", ram_q, src)

# (b) capture images_dict returned from reimagine_image
if "images_dict =" not in src and "run_processing_stage" in src:
    # find the call inside run_processing_stage
    src = re.sub(
        r"reimagine_image\(([^)]*)\)",
        r"images_dict = reimagine_image(\1)",
        src, count=1
    )
    # extend all_images_to_separate logic
    extend = (
        "\n        # === Patched by patch_merge_v1 ===\n"
        "        ram_only = os.getenv('FIBER_RAM_ONLY','0') in ('1','true','yes','y')\n"
        "        if ram_only:\n"
        "            for tag,img in images_dict.items():\n"
        "                all_images_to_separate.append(('RAM_'+tag, img))\n"
    )
    src = inject_after(r"all_images_to_separate\.extend[^\n]*\n", extend, src)

# (c) hand ndarray images to separator
if "for image_path in image_paths:" in src and ".process_ndarray" not in src:
    ndarray_branch = (
        "\n            # === Patched by patch_merge_v1 ===\n"
        "            if isinstance(image_path, tuple):   # (tag, ndarray)\n"
        "                tag, arr = image_path\n"
        "                consensus = separator.process_ndarray(arr, tag, str(image_separation_output_dir))\n"
        "                if consensus and consensus.get('saved_regions'):\n"
        "                    all_separated_regions.extend([Path(p) for p in consensus['saved_regions']])\n"
        "                continue\n"
    )
    src = inject_after(r"for image_path in image_paths:[^\n]*\n", ndarray_branch, src)

write(p, src); print(f"[patched] {fname}")

print("\nPatch complete. Original files are in:", BK,
      "\nSet  FIBER_RAM_ONLY=1  or answer “y” at the prompt to activate RAM‑only mode.")
print("Run   python app.py   as you always did – no workflow changes required.\n")

# --------------------------------------------------------------------------- #
# 4. Developer note
# --------------------------------------------------------------------------- #
print("NOTE  ➜  This patch only *adds* code.  If anything misbehaves, simply\n"
      f"        copy the originals back from {BK} .")

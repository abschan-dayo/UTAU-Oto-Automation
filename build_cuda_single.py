"""Rebuild the compact CUDA one-file application from the public source.

Requires a Windows Python 3.12 environment with CPU-only PyTorch, CuPy CUDA
12.x 13.5.1, fastrlock, PyInstaller, and algorithm/requirements.txt dependencies.
The reference EXE supplies only the four published models and four CUDA DLLs.
Generated build files stay in ignored local directories.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys

from PyInstaller.archive.readers import CArchiveReader
from PyInstaller.__main__ import run as pyinstaller_run


ROOT = Path(__file__).resolve().parent
MODELS = (
    "alignment.npz",
    "alignment_calibration.joblib",
    "existing_calibration.joblib",
    "vision.pt",
)
CUDA_DLLS = (
    "cudart64_12.dll",
    "cufft64_11.dll",
    "nvrtc64_120_0.dll",
    "nvrtc-builtins64_128.dll",
)


def extract_reference_files(reference_exe: Path, model_dir: Path, dll_dir: Path) -> None:
    archive = CArchiveReader(str(reference_exe))
    model_dir.mkdir(parents=True, exist_ok=True)
    dll_dir.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        data = archive.extract("models\\" + name)
        if not isinstance(data, bytes):
            raise ValueError("Model missing from reference EXE: " + name)
        (model_dir / name).write_bytes(data)
    for name in CUDA_DLLS:
        archive_name = "bin\\" + name if "bin\\" + name in archive.toc else name
        data = archive.extract(archive_name)
        if not isinstance(data, bytes):
            raise ValueError("CUDA DLL missing from reference EXE: " + name)
        (dll_dir / name).write_bytes(data)


def package_root(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if not spec or not spec.origin:
        raise RuntimeError(f"Required package not installed: {name}")
    return Path(spec.origin).resolve().parent.parent


def make_spec(spec_path: Path, model_dir: Path, dll_dir: Path, cupy_root: Path, device_profile: Path) -> None:
    binaries = [(str(dll_dir / name), "bin") for name in CUDA_DLLS]
    text = """# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

_deps = Path(%r)
datas = [(%r, 'models'), (%r, '.')]
binaries = %r
hiddenimports = ['graphlib']
for _package in ('tkinterdnd2', 'cupy', 'cupy_backends', 'fastrlock'):
    _data, _binary, _imports = collect_all(_package)
    datas += _data
    binaries += _binary
    hiddenimports += _imports
for _package in ('cupy', 'cupy_backends', 'cupyx'):
    for _extension in (_deps / _package).rglob('*.pyd'):
        binaries.append((str(_extension), str(_extension.parent.relative_to(_deps))))

a = Analysis(
    [%r],
    pathex=[%r, str(_deps)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
a.datas = [entry for entry in a.datas if not entry[0].lower().endswith(('.csv', 'direct_url.json'))]
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name='Autoto-CUDA',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
""" % (
        str(cupy_root), str(model_dir), str(device_profile), binaries,
        str(ROOT / "algorithm" / "v1_oto.pyw"), str(ROOT / "algorithm"),
    )
    spec_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the compact CUDA one-file EXE")
    parser.add_argument("--reference-exe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "build" / "cuda-single")
    args = parser.parse_args()
    reference = args.reference_exe.resolve(strict=True)
    work = args.work_dir.resolve()
    output = args.output_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    import torch
    if torch.version.cuda is not None:
        raise RuntimeError("Use CPU-only PyTorch for this compact build")
    cupy_root = package_root("cupy")
    package_root("cupy_backends")
    package_root("fastrlock")

    model_dir, dll_dir = work / "models", work / "cuda-dlls"
    extract_reference_files(reference, model_dir, dll_dir)
    device_profile = work / "distribution.json"
    device_profile.write_text('{"device":"cuda"}\n', encoding="utf-8")
    spec_path = work / "cuda_single.spec"
    make_spec(spec_path, model_dir, dll_dir, cupy_root, device_profile)

    tcl_base = Path(sys.base_prefix) / "tcl"
    for name, folder in (("TCL_LIBRARY", "tcl8.6"), ("TK_LIBRARY", "tk8.6")):
        path = tcl_base / folder
        if path.is_dir():
            os.environ[name] = "//?/" + str(path.resolve()).replace("\\", "/")

    pyinstaller_run([
        "--noconfirm",
        "--distpath", str(output),
        "--workpath", str(work / "pyinstaller"),
        str(spec_path),
    ])


if __name__ == "__main__":
    main()

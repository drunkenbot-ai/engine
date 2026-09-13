"""Backend package for preparing data and training small language models."""

import os
import sys
from pathlib import Path


__all__ = [
    "config",
    "data",
    "dataset_build",
    "dataset_preview",
    "dataset_mixture",
    "export",
    "gpu_discovery",
    "model",
    "resume_checks",
    "services",
    "tokenizer",
    "training",
    "training_orchestrator",
    "training_worker",
    "training_worker_protocol",
]


def find_tcc():
    """Find Triton's bundled tcc.exe on Windows safely without disk-wide traversal."""
    # 1. Check active Python's Triton installation
    try:
        import triton
        triton_path = Path(triton.__file__).resolve().parent
        tcc = triton_path / "runtime" / "tcc" / "tcc.exe"
        if tcc.is_file():
            return tcc
    except Exception:
        pass

    # 2. Check PATH
    import shutil
    w = shutil.which("tcc") or shutil.which("tcc.exe")
    if w:
        return Path(w)

    # 3. Direct site-packages candidates (exact path check only, never recursive rglob)
    python_root = Path(sys.executable).resolve().parent
    direct_candidates = [
        python_root / "Lib" / "site-packages" / "triton" / "runtime" / "tcc" / "tcc.exe",
        python_root / "Lib" / "site-packages" / "triton" / "tcc.exe",
    ]
    try:
        import site
        user_site = Path(site.getusersitepackages())
        direct_candidates.append(user_site / "triton" / "runtime" / "tcc" / "tcc.exe")
    except Exception:
        pass

    for cand in direct_candidates:
        if cand.is_file():
            return cand

    return None


def setup_tcc(permanent=False):
    """Find tcc.exe and configure CC if available."""
    tcc = find_tcc()
    if tcc is None:
        return False

    try:
        tcc = tcc.resolve()
        os.environ["CC"] = str(tcc)
        if permanent:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                "Environment",
                0,
                winreg.KEY_SET_VALUE,
            ) as key:
                winreg.SetValueEx(key, "CC", 0, winreg.REG_EXPAND_SZ, str(tcc))
    except Exception:
        pass

    return True


setup_tcc(permanent=False)

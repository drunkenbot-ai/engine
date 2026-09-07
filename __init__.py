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
    """Find Triton's bundled tcc.exe on Windows."""

    # ------------------------------------------------------------------
    # 1. Check the currently active Python's Triton installation
    # ------------------------------------------------------------------
    try:
        import triton

        triton_path = Path(triton.__file__).resolve().parent
        tcc = triton_path / "runtime" / "tcc" / "tcc.exe"

        if tcc.is_file():
            return tcc

    except ImportError:
        pass


    search_roots = []

    # Current Python
    python_root = Path(sys.executable).resolve().parent

    search_roots.extend([
        python_root / "Lib" / "site-packages",
        python_root / "Lib" / "site-packages" / "triton",
    ])

    # User site-packages
    try:
        import site

        user_site = site.getusersitepackages()
        search_roots.append(Path(user_site))

    except Exception:
        pass

    for root in search_roots:
        if not root.exists():
            continue

        # Direct Triton path
        tcc = root / "triton" / "runtime" / "tcc" / "tcc.exe"

        if tcc.is_file():
            return tcc

        # Recursive fallback
        try:
            for tcc in root.rglob("tcc.exe"):
                if (
                    tcc.parent.name == "tcc"
                    and tcc.parent.parent.name == "runtime"
                    and tcc.parent.parent.parent.name == "triton"
                ):
                    return tcc

        except (PermissionError, OSError):
            pass

    common_roots = [
        Path(os.environ.get("LOCALAPPDATA", "")),
        Path(os.environ.get("APPDATA", "")),
        Path(os.environ.get("PROGRAMFILES", "")),
        Path(os.environ.get("PROGRAMFILES(X86)", "")),
    ]

    for root in common_roots:
        if not root.exists():
            continue

        try:
            for tcc in root.rglob("triton/runtime/tcc/tcc.exe"):
                if tcc.is_file():
                    return tcc

        except (PermissionError, OSError):
            pass

    for drive in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        root = Path(f"{drive}:\\")
        if not root.exists():
            continue

        try:
            for tcc in root.rglob("triton/runtime/tcc/tcc.exe"):
                if tcc.is_file():
                    return tcc

        except (PermissionError, OSError):
            pass

    return None


def setup_tcc(permanent=False):
    """Find tcc.exe and configure CC."""

    tcc = find_tcc()

    if tcc is None:
        print("ERROR: Could not find Triton's tcc.exe")
        return False

    tcc = tcc.resolve()

    print(f"Found Triton TCC:")
    print(f"  {tcc}")

    # Set for current Python process
    os.environ["CC"] = str(tcc)

    print()
    print(f"CC={os.environ['CC']}")

    # Optionally set permanently for the current Windows user
    if permanent:
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                "Environment",
                0,
                winreg.KEY_SET_VALUE,
            ) as key:
                winreg.SetValueEx(
                    key,
                    "CC",
                    0,
                    winreg.REG_EXPAND_SZ,
                    str(tcc),
                )

            print()
            print("CC has also been saved permanently for the current user.")
            print("Restart applications/terminals for the change to take effect.")

        except Exception as exc:
            print(f"WARNING: Could not save permanent CC: {exc}")

    return True


setup_tcc(permanent=False)

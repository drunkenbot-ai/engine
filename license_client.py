"""IDE license validation: online-first, with a signed offline grace cache.

Calls the DrunkenBot cloud service's ``POST /license/validate`` at launch.
On success, caches a short-lived signed "grace receipt" locally so the app
can still launch for a limited window if the server is unreachable next
time (see ``LICENSE_SERVER_URL`` and the cloud-service README for the full
design). This module never trusts anything it did not itself verify with
:data:`LICENSE_PUBLIC_KEY_PEM` -- an unsigned or badly-signed receipt is
treated exactly like "no receipt at all."

IMPORTANT: :data:`LICENSE_PUBLIC_KEY_PEM` below is a placeholder generated
for scaffolding only. Before this ships, replace it with the real public
key printed by ``cloud-service/scripts/generate_keypair.py`` when the
production signing keypair is generated. Shipping the placeholder means
this app will only ever trust receipts signed by a throwaway key nobody
else has -- i.e. license validation will never succeed for real customers.
"""

from __future__ import annotations

import json
import platform
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import traceback
import ssl
import certifi
import sys
import base64

from cryptography.exceptions import InvalidSignature
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.serialization import load_pem_public_key

# --- REPLACE BEFORE SHIPPING: see module docstring. ---
LICENSE_PUBLIC_KEY_PEM = """-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEAledx+Yhz/kvDTBFfBscicAMUIcwwG2jI2/zrwK6VFwI=
-----END PUBLIC KEY-----
"""

LICENSE_DIR = Path.home() / ".drunkenbot_ide" / "license"
LICENSE_KEY_FILE = LICENSE_DIR / "license_key.txt"
LICENSE_VAULT_FILE = LICENSE_DIR / "license_vault.enc"
GRACE_CACHE_FILE = LICENSE_DIR / "grace_receipt.json"
MACHINE_ID_FILE = LICENSE_DIR / "machine_id.txt"

_REQUEST_TIMEOUT_SECONDS = 10.0
_ONLINE_VALIDATION_ATTEMPTS = 2


@dataclass
class LicenseCheckResult:
    """Outcome of a license check.

    Attributes:
        valid: Whether the app is licensed to launch.
        reason: Human-readable explanation, especially when not valid.
        used_offline_grace: Whether this result came from a cached grace
            receipt rather than a live server response.
        version_ceiling: Highest app version the license currently covers,
            when known.
        grace_period_until: ISO timestamp of any temporary grace extension,
            when known.
    """

    valid: bool
    reason: str
    used_offline_grace: bool = False
    version_ceiling: Optional[str] = None
    grace_period_until: Optional[str] = None


def _get_or_create_machine_id() -> str:
    """Return this machine's pseudonymous ID, generating one if needed.

    Deliberately a random value with no relationship to any real hardware
    or OS identifier (not a disk serial, not a MAC address, not tied to a
    Windows/OS username) -- it exists only to distinguish installs and spot
    abuse patterns, not to fingerprint a specific physical device or person.

    Returns:
        Stable machine ID, persisted locally after first generation.
    """

    if MACHINE_ID_FILE.exists():
        existing = MACHINE_ID_FILE.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    machine_id = uuid.uuid4().hex
    LICENSE_DIR.mkdir(parents=True, exist_ok=True)
    MACHINE_ID_FILE.write_text(machine_id, encoding="utf-8")
    return machine_id


def _collect_telemetry() -> dict:
    """Collect the minimal, privacy-conscious launch telemetry payload.

    Deliberately excludes anything directly identifying: no OS username, no
    hostname, no hardware identifiers. ISP/geolocation is derived
    server-side from the request's source IP, not reported here.

    Returns:
        Telemetry dict matching the server's ``LaunchTelemetry`` schema.
    """

    return {
        "machine_id": _get_or_create_machine_id(),
        "os": platform.system() or None,
        "os_version": platform.release() or None,
    }


def _parse_version(version: str) -> tuple[int, int, int]:
    """Parse a version string into a comparable tuple.

    Mirrors the cloud service's ``app/versioning.py`` exactly -- kept as an
    independent copy rather than a shared import, since LLM-IDE and
    cloud-service are separate applications/repos with no shared package.

    Args:
        version: Version string, e.g. ``"2.1.0"``.

    Returns:
        ``(major, minor, patch)`` tuple. Missing components default to 0.
    """

    core = version.strip().split("-", 1)[0].split("+", 1)[0]
    parts = core.split(".")
    numbers = [int(part) for part in parts[:3] if part.isdigit()]
    while len(numbers) < 3:
        numbers.append(0)
    return numbers[0], numbers[1], numbers[2]


def _is_version_within_ceiling(app_version: str, version_ceiling: str) -> bool:
    """Return whether an app version is covered by a license's ceiling.

    Args:
        app_version: Version of the running app.
        version_ceiling: Highest version the license entitles the holder to.

    Returns:
        True if ``app_version <= version_ceiling``.
    """

    return _parse_version(app_version) <= _parse_version(version_ceiling)


def _load_public_key() -> Ed25519PublicKey:
    """Load the embedded Ed25519 public key.

    Returns:
        Loaded public key object.
    """

    key = load_pem_public_key(LICENSE_PUBLIC_KEY_PEM.encode("utf-8"))
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("Embedded license public key is not Ed25519.")
    return key


def _verify_receipt(receipt: str, signature_b64: str) -> dict:
    """Verify a signed receipt and return its parsed payload.

    Args:
        receipt: Exact canonical JSON string that was signed.
        signature_b64: Base64-encoded Ed25519 signature over ``receipt``.

    Returns:
        Parsed receipt payload.

    Raises:
        InvalidSignature: If the signature does not match.
        ValueError: If the receipt is not valid JSON.
    """

    import base64

    public_key = _load_public_key()
    public_key.verify(base64.b64decode(signature_b64), receipt.encode("utf-8"))
    return json.loads(receipt)


def _derive_machine_key(machine_id: str) -> bytes:
    """Derive an authenticated encryption key bound to this machine.

    Combines the persistent machine_id with the local hardware node
    (MAC/hardware address) and a dedicated salt via PBKDF2-HMAC-SHA256.
    """
    salt = b"drunkenbot_ide_vault_v1"
    seed = f"{machine_id}:{uuid.getnode()}".encode("utf-8")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=50_000,
    )
    return base64.urlsafe_b64encode(kdf.derive(seed))


def _dpapi_protect(data: bytes) -> bytes:
    """Encrypt data bound to the current Windows user and machine credentials."""
    if sys.platform != "win32":
        return data
    import ctypes
    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    blob_in = DATA_BLOB(len(data), ctypes.cast(ctypes.create_string_buffer(data), ctypes.POINTER(ctypes.c_char)))
    blob_out = DATA_BLOB()
    if not ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(blob_in), "license_vault", None, None, None, 0, ctypes.byref(blob_out)
    ):
        raise RuntimeError("Windows DPAPI CryptProtectData failed.")
    result = ctypes.string_at(blob_out.pbData, blob_out.cbData)
    ctypes.windll.kernel32.LocalFree(blob_out.pbData)
    return result


def _dpapi_unprotect(data: bytes) -> bytes:
    """Decrypt data bound to the current Windows user and machine credentials."""
    if sys.platform != "win32":
        return data
    import ctypes
    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    blob_in = DATA_BLOB(len(data), ctypes.cast(ctypes.create_string_buffer(data), ctypes.POINTER(ctypes.c_char)))
    blob_out = DATA_BLOB()
    if not ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)
    ):
        raise RuntimeError("Windows DPAPI CryptUnprotectData failed.")
    result = ctypes.string_at(blob_out.pbData, blob_out.cbData)
    ctypes.windll.kernel32.LocalFree(blob_out.pbData)
    return result


def encrypt_license_metadata(metadata: dict, machine_id: Optional[str] = None) -> bytes:
    """Encrypt license metadata dictionary using machine-bound two-layer encryption."""
    if machine_id is None:
        machine_id = _get_or_create_machine_id()
    raw_bytes = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    fernet_key = _derive_machine_key(machine_id)
    f = Fernet(fernet_key)
    fernet_token = f.encrypt(raw_bytes)
    if sys.platform == "win32":
        try:
            dpapi_token = _dpapi_protect(fernet_token)
            return b"DBDPAPI:" + dpapi_token
        except Exception:
            pass
    return b"DBFERNET:" + fernet_token


def decrypt_license_metadata(blob: bytes, machine_id: Optional[str] = None) -> dict:
    """Decrypt machine-bound encrypted license blob and return dictionary.

    Raises:
        InvalidToken, RuntimeError, or ValueError if decryption fails.
    """
    if machine_id is None:
        machine_id = _get_or_create_machine_id()

    if blob.startswith(b"DBDPAPI:"):
        raw_cipher = blob[len(b"DBDPAPI:"):]
        fernet_token = _dpapi_unprotect(raw_cipher)
    elif blob.startswith(b"DBFERNET:"):
        fernet_token = blob[len(b"DBFERNET:"):]
    else:
        fernet_token = blob

    fernet_key = _derive_machine_key(machine_id)
    f = Fernet(fernet_key)
    raw_bytes = f.decrypt(fernet_token)
    return json.loads(raw_bytes.decode("utf-8"))


def save_encrypted_license(metadata: dict) -> None:
    """Persist license metadata encrypted on the user's machine."""
    LICENSE_DIR.mkdir(parents=True, exist_ok=True)
    blob = encrypt_license_metadata(metadata)
    LICENSE_VAULT_FILE.write_bytes(blob)
    if "license_key" in metadata and metadata["license_key"]:
        try:
            LICENSE_KEY_FILE.write_text(str(metadata["license_key"]).strip(), encoding="utf-8")
        except Exception:
            pass


def load_encrypted_license() -> Optional[dict]:
    """Load and decrypt the local license vault, or None if unreadable or missing."""
    if not LICENSE_VAULT_FILE.exists():
        return None
    try:
        blob = LICENSE_VAULT_FILE.read_bytes()
        return decrypt_license_metadata(blob)
    except Exception:
        return None


def clear_encrypted_license() -> None:
    """Delete the encrypted license vault."""
    LICENSE_VAULT_FILE.unlink(missing_ok=True)


def check_local_license(app_version: str) -> LicenseCheckResult:
    """Check if a valid, unexpired, machine-matched license is stored locally.

    Returns:
        LicenseCheckResult indicating if local encrypted license is valid.
    """
    metadata = load_encrypted_license()
    if not metadata:
        return LicenseCheckResult(
            valid=False,
            reason="No valid local encrypted license found on this machine.",
        )

    current_machine_id = _get_or_create_machine_id()
    vault_machine_id = metadata.get("machine_id")
    if vault_machine_id and vault_machine_id != current_machine_id:
        return LicenseCheckResult(
            valid=False,
            reason="Local license is bound to a different machine.",
        )

    if not metadata.get("valid", False):
        return LicenseCheckResult(
            valid=False,
            reason=metadata.get("reason", "Local license is marked invalid."),
        )

    version_ceiling = metadata.get("version_ceiling")
    if version_ceiling and not _is_version_within_ceiling(app_version, version_ceiling):
        return LicenseCheckResult(
            valid=False,
            reason=f"This license covers up to version {version_ceiling}.",
            version_ceiling=version_ceiling,
        )

    grace_period_until = metadata.get("grace_period_until")
    if grace_period_until:
        try:
            valid_until = datetime.fromisoformat(grace_period_until)
            if datetime.now(timezone.utc) > valid_until:
                return LicenseCheckResult(
                    valid=False,
                    reason="The local license validity period has expired.",
                    grace_period_until=grace_period_until,
                )
        except Exception:
            pass

    return LicenseCheckResult(
        valid=True,
        reason="Validated from local encrypted license.",
        used_offline_grace=False,
        version_ceiling=version_ceiling,
        grace_period_until=grace_period_until,
    )


def load_stored_license_key() -> Optional[str]:
    """Return the previously activated license key, if any.

    Returns:
        Stored license key, or ``None`` if never activated.
    """
    metadata = load_encrypted_license()
    if metadata and metadata.get("license_key"):
        return str(metadata["license_key"]).strip()
    if not LICENSE_KEY_FILE.exists():
        return None
    key = LICENSE_KEY_FILE.read_text(encoding="utf-8").strip()
    return key or None


def store_license_key(license_key: str) -> None:
    """Persist an activated license key for future launches.

    Args:
        license_key: License key entered during activation.
    """

    LICENSE_DIR.mkdir(parents=True, exist_ok=True)
    LICENSE_KEY_FILE.write_text(license_key.strip(), encoding="utf-8")


def _load_cached_receipt() -> Optional[tuple[str, str]]:
    """Load the last cached grace receipt and its signature, if present.

    Returns:
        ``(receipt, signature)`` tuple, or ``None`` if no cache exists or it
        is unreadable.
    """

    if not GRACE_CACHE_FILE.exists():
        return None
    try:
        cached = json.loads(GRACE_CACHE_FILE.read_text(encoding="utf-8"))
        return cached["receipt"], cached["signature"]
    except Exception:
        return None


def _store_cached_receipt(receipt: str, signature: str) -> None:
    """Cache a validated grace receipt and its signature to disk.

    Args:
        receipt: Canonical receipt JSON string.
        signature: Base64-encoded signature over ``receipt``.
    """

    LICENSE_DIR.mkdir(parents=True, exist_ok=True)
    GRACE_CACHE_FILE.write_text(json.dumps({"receipt": receipt, "signature": signature}), encoding="utf-8")


def _clear_cached_receipt() -> None:
    """Delete any cached grace receipt and encrypted vault."""

    GRACE_CACHE_FILE.unlink(missing_ok=True)
    clear_encrypted_license()


def _validate_online(license_key: str, app_version: str, server_url: str) -> Optional[dict]:
    """Call the cloud service's validation endpoint.

    Args:
        license_key: License key to validate.
        app_version: Running app's version.
        server_url: Base URL of the DrunkenBot cloud service.

    Returns:
        Parsed JSON response, or ``None`` if the server could not be
        reached at all (caller should fall back to the offline grace
        cache in that case). A reachable server that responds with
        ``valid: false`` is NOT a network failure -- that is returned
        normally so the caller treats it as an authoritative rejection.
    """

    body = json.dumps(
        {
            "license_key": license_key,
            "app_version": app_version,
            "telemetry": _collect_telemetry(),
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{server_url.rstrip('/')}/license/validate",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/plain, */*",
            "User-Agent": "Mozilla/5.0",
        },
        method="POST",
    )
    context = ssl.create_default_context(cafile=certifi.where())
    for attempt in range(1, _ONLINE_VALIDATION_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(
                    request,
                    timeout=_REQUEST_TIMEOUT_SECONDS,
                    context=context,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            print("HTTP Error:", e.code)
            print(e.read().decode())
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"License server attempt {attempt}/{_ONLINE_VALIDATION_ATTEMPTS} failed: {e}")
            if attempt < _ONLINE_VALIDATION_ATTEMPTS:
                time.sleep(min(2.0 * attempt, 8.0))
        except Exception:
            traceback.print_exc()
            return None
    return None


def check_license_at_launch(app_version: str, server_url: str) -> LicenseCheckResult:
    """Validate the license at app startup: local-first, online fallback.

    Order of operations:
      1. First check if locally user has a valid encrypted license.
         If valid, consider it valid immediately without remote network calls
         or activation dialogs.
      2. If local check fails (no file, expired, corrupted, or machine mismatch),
         fall back to online validation if a stored license key exists.
      3. Server unreachable -> fall back to a cached grace receipt, if one
         exists, is correctly signed, and has not expired.
      4. If all fail -> not licensed, return invalid result to prompt for activation.

    Args:
        app_version: Version of the running app (compared against the
            license's version ceiling).
        server_url: Base URL of the DrunkenBot cloud service.

    Returns:
        License check result.
    """

    # 1. Local-first check
    local_result = check_local_license(app_version)
    if local_result.valid:
        return local_result

    license_key = load_stored_license_key()
    if not license_key:
        return LicenseCheckResult(valid=False, reason="No license activated on this machine.")

    response = _validate_online(license_key, app_version, server_url)
    if response is not None:
        if response.get("valid"):
            receipt = response.get("receipt")
            signature = response.get("signature")
            version_ceiling = response.get("version_ceiling") or "999.0.0"
            grace_period_until = response.get("grace_period_until")

            vault_metadata = {
                "license_key": license_key,
                "machine_id": _get_or_create_machine_id(),
                "hardware_node": str(uuid.getnode()),
                "activated_at": datetime.now(timezone.utc).isoformat(),
                "last_verified_at": datetime.now(timezone.utc).isoformat(),
                "valid": True,
                "version_ceiling": version_ceiling,
                "grace_period_until": grace_period_until,
                "receipt": receipt,
                "signature": signature,
                "app_version": app_version,
            }
            save_encrypted_license(vault_metadata)

            if receipt and signature:
                _store_cached_receipt(receipt, signature)
            return LicenseCheckResult(
                valid=True,
                reason="Validated online.",
                version_ceiling=version_ceiling,
                grace_period_until=grace_period_until,
            )
        # Authoritative, live rejection -- do not let a stale cache override this.
        _clear_cached_receipt()
        return LicenseCheckResult(
            valid=False,
            reason=response.get("reason", "License is not valid."),
            version_ceiling=response.get("version_ceiling"),
            grace_period_until=response.get("grace_period_until"),
        )

    # Server unreachable: fall back to a cached grace receipt, if any.
    cached = _load_cached_receipt()
    if cached is None:
        return LicenseCheckResult(
            valid=False,
            reason="Could not reach the license server and no valid local license is available. "
            "Please connect to the internet once to validate your license.",
        )
    receipt_json, signature = cached
    try:
        payload = _verify_receipt(receipt_json, signature)
    except (InvalidSignature, ValueError, KeyError):
        return LicenseCheckResult(
            valid=False,
            reason="Cached license grace data is corrupt or invalid. "
            "Please connect to the internet once to re-validate your license.",
        )

    valid_until = datetime.fromisoformat(payload["valid_until"])
    if datetime.now(timezone.utc) > valid_until:
        return LicenseCheckResult(
            valid=False,
            reason="Your offline grace period has expired. Please connect to the internet to re-validate.",
        )
    if not _is_version_within_ceiling(app_version, payload["version_ceiling"]):
        return LicenseCheckResult(
            valid=False,
            reason=f"This license covers up to version {payload['version_ceiling']}.",
            version_ceiling=payload["version_ceiling"],
        )
    return LicenseCheckResult(
        valid=True,
        reason="Validated from cached offline grace receipt.",
        used_offline_grace=True,
        version_ceiling=payload["version_ceiling"],
    )


def activate_license(license_key: str, app_version: str, server_url: str) -> LicenseCheckResult:
    """Validate a license key with the server and persist encrypted metadata locally.

    Args:
        license_key: License key to activate.
        app_version: Current application version.
        server_url: Base URL of the DrunkenBot cloud service.

    Returns:
        LicenseCheckResult indicating activation outcome.
    """
    clean_key = license_key.strip()
    if not clean_key:
        return LicenseCheckResult(valid=False, reason="Please enter a license key.")

    response = _validate_online(clean_key, app_version, server_url)
    if response is not None:
        if response.get("valid"):
            version_ceiling = response.get("version_ceiling") or "999.0.0"
            grace_period_until = response.get("grace_period_until")
            receipt = response.get("receipt")
            signature = response.get("signature")

            metadata = {
                "license_key": clean_key,
                "machine_id": _get_or_create_machine_id(),
                "hardware_node": str(uuid.getnode()),
                "activated_at": datetime.now(timezone.utc).isoformat(),
                "last_verified_at": datetime.now(timezone.utc).isoformat(),
                "valid": True,
                "version_ceiling": version_ceiling,
                "grace_period_until": grace_period_until,
                "receipt": receipt,
                "signature": signature,
                "app_version": app_version,
            }
            save_encrypted_license(metadata)
            store_license_key(clean_key)
            if receipt and signature:
                _store_cached_receipt(receipt, signature)

            return LicenseCheckResult(
                valid=True,
                reason="License activated successfully.",
                version_ceiling=version_ceiling,
                grace_period_until=grace_period_until,
            )

        # Explicit rejection from server
        clear_encrypted_license()
        _clear_cached_receipt()
        return LicenseCheckResult(
            valid=False,
            reason=response.get("reason", "License key is invalid or not found."),
            version_ceiling=response.get("version_ceiling"),
            grace_period_until=response.get("grace_period_until"),
        )

    # Server unreachable during activation
    return LicenseCheckResult(
        valid=False,
        reason="Could not reach the license server to activate. "
        "Please check your internet connection and try again.",
    )



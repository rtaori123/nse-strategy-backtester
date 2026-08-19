"""Make HTTPS work behind a corporate TLS-inspecting proxy.

Some networks (e.g. corporate laptops) terminate TLS with a self-signed root CA
that is trusted by the Windows certificate store but NOT by certifi's bundle,
which yfinance's HTTP layer (curl_cffi) uses. That makes every data fetch fail
with ``CertificateVerifyError``.

On import this builds a combined CA bundle (certifi's public roots + the local
OS trust store) and points curl_cffi/requests at it via env vars — so
verification stays ON and simply trusts the proxy's root too. It is a no-op on
machines with no extra roots, and never disables verification.

Set ``BACKTESTER_NO_SSL_PATCH=1`` to skip entirely, or point ``CURL_CA_BUNDLE``
at your own bundle to override.
"""
from __future__ import annotations

import logging
import os
import ssl
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_BUNDLE = Path(__file__).resolve().parent.parent / "data" / "ca_bundle.pem"
_MAX_AGE_DAYS = 30


def _build_bundle(dest: Path) -> int:
    import certifi

    parts: list[str] = [Path(certifi.where()).read_text(encoding="utf-8")]
    added = 0
    if hasattr(ssl, "enum_certificates"):  # Windows only
        for store in ("ROOT", "CA"):
            try:
                for cert_bytes, enc, _trust in ssl.enum_certificates(store):
                    if enc == "x509_asn":
                        parts.append(ssl.DER_cert_to_PEM_cert(cert_bytes))
                        added += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not read cert store %s: %s", store, exc)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(parts), encoding="utf-8")
    return added


def ensure_ca_bundle() -> str | None:
    """Ensure a combined CA bundle exists and export the env vars. Returns path."""
    if os.environ.get("BACKTESTER_NO_SSL_PATCH"):
        return None
    # Respect an explicit user override.
    if os.environ.get("CURL_CA_BUNDLE") and Path(os.environ["CURL_CA_BUNDLE"]).exists():
        return os.environ["CURL_CA_BUNDLE"]

    fresh = (
        _BUNDLE.exists()
        and (time.time() - _BUNDLE.stat().st_mtime) / 86400.0 <= _MAX_AGE_DAYS
    )
    if not fresh:
        try:
            n = _build_bundle(_BUNDLE)
            logger.info("Built CA bundle with %d OS certs at %s", n, _BUNDLE)
        except Exception as exc:  # noqa: BLE001 - fall back to certifi default
            logger.warning("Could not build CA bundle: %s", exc)
            return None

    path = str(_BUNDLE)
    os.environ.setdefault("CURL_CA_BUNDLE", path)   # curl_cffi (yfinance)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", path)  # requests (NSE refresh)
    os.environ.setdefault("SSL_CERT_FILE", path)    # stdlib ssl
    return path


# Apply on import.
ensure_ca_bundle()

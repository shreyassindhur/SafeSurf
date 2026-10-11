"""
SafeSurf - live SSL and domain-age lookups
------------------------------------------
Replaces the two placeholder functions get_whois_info() and get_ssl_info()
in streamlit_app.py. Every lookup has a timeout and a safe fallback, so a
failed lookup never crashes the app (same idea as in the project report).
"""
import ssl
import socket
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import whois  # from the python-whois package (already imported in the app)

SSL_TIMEOUT = 4      # seconds
WHOIS_TIMEOUT = 6    # seconds

# Domains like example.co.uk or example.co.in have a 2-part ending.
TWO_PART_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk",
    "co.in", "org.in", "net.in", "gov.in", "ac.in", "edu.in",
    "com.au", "net.au", "org.au", "co.jp", "co.nz", "com.br", "co.za",
}


def registrable_domain(hostname):
    """'login.paypal.com' -> 'paypal.com' ; 'shop.example.co.uk' -> 'example.co.uk'"""
    parts = (hostname or "").lower().strip(".").split(".")
    if len(parts) <= 2:
        return ".".join(parts)
    if ".".join(parts[-2:]) in TWO_PART_SUFFIXES and len(parts) >= 3:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def get_ssl_info(domain):
    """
    Connect to port 443 and let Python verify the certificate.
    create_default_context() automatically checks that the certificate:
      - is signed by a trusted Certificate Authority
      - has not expired
      - matches the domain name
    Any failure raises an error, which we treat as 'not valid'.
    Returns: {'ssl_valid': 0/1, 'issuer': str, 'ok': True/False}
      'ok' = False means the lookup itself failed (network problem, timeout),
      which is different from 'the certificate is bad'.
    """
    if not domain:
        return {"ssl_valid": 0, "issuer": "", "ok": False}
    try:
        context = ssl.create_default_context()
        with socket.create_connection((domain, 443), timeout=SSL_TIMEOUT) as sock:
            with context.wrap_socket(sock, server_hostname=domain) as tls:
                cert = tls.getpeercert()
        issuer = dict(item[0] for item in cert.get("issuer", ()))
        return {
            "ssl_valid": 1,
            "issuer": issuer.get("organizationName", issuer.get("commonName", "")),
            "ok": True,
        }
    except ssl.SSLError:
        # Server answered, but the certificate is bad (expired, self-signed, wrong name)
        return {"ssl_valid": 0, "issuer": "", "ok": True}
    except ConnectionRefusedError:
        # Server is reachable but has no HTTPS at all (port 443 closed) - a real signal
        return {"ssl_valid": 0, "issuer": "", "ok": True}
    except (socket.timeout, socket.gaierror, ConnectionError, OSError):
        # Could not find or reach the server (DNS failure, blocked, timeout): result unknown
        return {"ssl_valid": 0, "issuer": "", "ok": False}


def _age_in_days(creation_date):
    """WHOIS may return a datetime, a list of datetimes, or nothing."""
    if isinstance(creation_date, list):
        creation_date = min(d for d in creation_date if d) if creation_date else None
    if not creation_date:
        return None
    if creation_date.tzinfo is None:
        creation_date = creation_date.replace(tzinfo=timezone.utc)
    days = (datetime.now(timezone.utc) - creation_date).days
    return max(days, 0)


def get_whois_info(domain):
    """
    Look up when the domain was registered and return its age in days.
    Runs in a worker thread so we can enforce our own timeout.
    Returns: {'domain_age_days': int, 'registrar': str, 'creation_date': str, 'ok': bool}
    """
    empty = {"domain_age_days": 0, "registrar": "", "creation_date": "", "ok": False}
    name = registrable_domain(domain)
    if not name or name.replace(".", "").isdigit():   # empty or a raw IP address
        return empty
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            record = pool.submit(whois.whois, name).result(timeout=WHOIS_TIMEOUT)
        age = _age_in_days(record.creation_date)
        if age is None:
            return empty
        created = record.creation_date
        if isinstance(created, list):
            created = min(created)
        return {
            "domain_age_days": age,
            "registrar": str(record.registrar or ""),
            "creation_date": str(created),
            "ok": True,
        }
    except (FutureTimeout, Exception):
        return empty

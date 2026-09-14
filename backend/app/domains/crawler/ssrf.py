import ipaddress
import os
import socket
from urllib.parse import urlparse


class SSRFViolation(Exception):
    pass


def validate_url_safe(url: str) -> bool:
    """
    Validates that a URL does not point to an internal or private IP address.
    Raises SSRFViolation if it does.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname

    if not hostname:
        raise SSRFViolation("Invalid hostname")

    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        raise SSRFViolation(f"Could not resolve hostname: {hostname}")

    allow_local = os.getenv("ALLOW_LOCAL_URLS", "false").lower() == "true"
    for info in addr_info:
        ip_str = info[4][0]
        ip = ipaddress.ip_address(ip_str)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_unspecified
        ):
            if allow_local:
                continue
            raise SSRFViolation(
                f"SSRF blocked: Hostname {hostname} resolves to private IP {ip_str}"
            )

    return True

import socket

import pytest

from app.domains.crawler.normalizer import normalize_url, resolve_url
from app.domains.crawler.ssrf import SSRFViolation, validate_url_safe


def test_normalize_url():
    assert (
        normalize_url("HTTPS://Example.COM:443/Path?q=1#frag")
        == "https://example.com/Path?q=1"
    )
    assert normalize_url("http://example.com:80") == "http://example.com/"
    assert normalize_url("http://example.com") == "http://example.com/"


def test_resolve_url():
    assert (
        resolve_url("https://example.com/foo/", "../bar") == "https://example.com/bar"
    )
    assert resolve_url("https://example.com/foo", "/bar") == "https://example.com/bar"


def test_ssrf_validation_safe(monkeypatch):
    # Stub DNS so the test is hermetic: a hostname resolving to a public IP is allowed
    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    assert validate_url_safe("https://example.com") is True


def test_ssrf_validation_rejects_hostname_resolving_to_private_ip(monkeypatch):
    # DNS rebinding style: a public-looking hostname that resolves to a private address
    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.delenv("ALLOW_LOCAL_URLS", raising=False)
    with pytest.raises(SSRFViolation):
        validate_url_safe("https://internal.example.com")


def test_ssrf_validation_unsafe():
    with pytest.raises(SSRFViolation):
        validate_url_safe("http://127.0.0.1:8080/admin")

    with pytest.raises(SSRFViolation):
        validate_url_safe("http://localhost/secret")

    with pytest.raises(SSRFViolation):
        validate_url_safe("http://169.254.169.254/latest/meta-data")

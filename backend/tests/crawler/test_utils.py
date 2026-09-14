import pytest
from app.domains.crawler.normalizer import normalize_url, resolve_url
from app.domains.crawler.ssrf import validate_url_safe, SSRFViolation

def test_normalize_url():
    assert normalize_url("HTTPS://Example.COM:443/Path?q=1#frag") == "https://example.com/Path?q=1"
    assert normalize_url("http://example.com:80") == "http://example.com/"
    assert normalize_url("http://example.com") == "http://example.com/"

def test_resolve_url():
    assert resolve_url("https://example.com/foo/", "../bar") == "https://example.com/bar"
    assert resolve_url("https://example.com/foo", "/bar") == "https://example.com/bar"

def test_ssrf_validation_safe():
    # Should resolve to public IPs
    assert validate_url_safe("https://example.com") is True

def test_ssrf_validation_unsafe():
    with pytest.raises(SSRFViolation):
        validate_url_safe("http://127.0.0.1:8080/admin")
        
    with pytest.raises(SSRFViolation):
        validate_url_safe("http://localhost/secret")
        
    with pytest.raises(SSRFViolation):
        validate_url_safe("http://169.254.169.254/latest/meta-data")

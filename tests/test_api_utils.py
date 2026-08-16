import pytest

from api_utils import InputValidator, RetryConfig, ValidationConfig, retry_with_backoff


def test_retry_with_backoff_retries_until_success(monkeypatch):
    delays = []
    attempts = {"count": 0}

    monkeypatch.setattr("api_utils.time.sleep", lambda delay: delays.append(delay))

    @retry_with_backoff(
        RetryConfig(
            max_retries=2,
            initial_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=False,
            retry_exceptions=(ValueError,),
        )
    )
    def flaky_operation():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ValueError("falha transitória")
        return "ok"

    assert flaky_operation() == "ok"
    assert attempts["count"] == 3
    assert delays == [1.0, 2.0]


def test_input_validator_validates_and_sanitizes_filenames(tmp_path):
    validator = InputValidator(ValidationConfig(max_file_size_mb=1, allowed_extensions={".jpg"}))
    image_path = tmp_path / "entrada.jpg"
    image_path.write_bytes(b"1234567890")

    is_valid, message = validator.validate_file(image_path)

    assert is_valid is True
    assert message == "OK"
    assert validator.sanitize_filename('relatorio<forense>:ocr?.jpg') == "relatorio_forense__ocr_.jpg"

def test_permanent_errors_are_not_retried(monkeypatch):
    """Regressão: API_RETRY_CONFIG repetia QUALQUER Exception 5x com backoff.

    Uma chave inválida gastava ~2 minutos em tentativas que nunca poderiam passar.
    """
    from api_utils import API_RETRY_CONFIG

    slept = []
    monkeypatch.setattr("api_utils.time.sleep", lambda delay: slept.append(delay))
    attempts = {"count": 0}

    @retry_with_backoff(API_RETRY_CONFIG)
    def chave_invalida():
        attempts["count"] += 1
        raise RuntimeError("Error code: 401 - Incorrect API key provided: sk-xxx")

    with pytest.raises(RuntimeError):
        chave_invalida()

    assert attempts["count"] == 1, "erro de autenticação não deve ser repetido"
    assert slept == []


def test_transient_errors_are_still_retried(monkeypatch):
    from api_utils import API_RETRY_CONFIG

    monkeypatch.setattr("api_utils.time.sleep", lambda delay: None)
    attempts = {"count": 0}

    @retry_with_backoff(API_RETRY_CONFIG)
    def instavel():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ConnectionError("connection reset by peer")
        return "ok"

    assert instavel() == "ok"
    assert attempts["count"] == 3


def test_is_transient_error_classifies_status_codes():
    from api_utils import is_transient_error

    class ErroComStatus(Exception):
        def __init__(self, status):
            super().__init__(f"http {status}")
            self.status_code = status

    assert is_transient_error(ErroComStatus(429)) is True
    assert is_transient_error(ErroComStatus(503)) is True
    assert is_transient_error(ErroComStatus(401)) is False
    assert is_transient_error(ErroComStatus(400)) is False

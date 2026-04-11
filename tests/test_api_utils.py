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
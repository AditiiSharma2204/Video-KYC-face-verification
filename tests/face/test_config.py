import pytest

from videokyc.face import FaceConfig


def test_default_threshold_per_model():
    assert FaceConfig(model_name="ArcFace").effective_threshold == 0.68
    assert FaceConfig(model_name="Facenet512").effective_threshold == 0.30


def test_explicit_threshold_wins():
    assert FaceConfig(model_name="ArcFace", threshold=0.5).effective_threshold == 0.5


def test_unknown_model_without_threshold_is_an_error():
    with pytest.raises(ValueError, match="No default threshold"):
        FaceConfig(model_name="Mystery").effective_threshold  # noqa: B018


def test_from_env(monkeypatch):
    monkeypatch.setenv("FACEMATCH_MODEL", "Facenet512")
    monkeypatch.setenv("FACEMATCH_THRESHOLD", "0.25")
    monkeypatch.setenv("FACEMATCH_LIVENESS", "1")
    cfg = FaceConfig.from_env()
    assert (cfg.model_name, cfg.effective_threshold, cfg.liveness) == ("Facenet512", 0.25, True)

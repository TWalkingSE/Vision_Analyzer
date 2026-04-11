from analysis_pipeline import get_model_short_name as get_pipeline_model_short_name
from batch_image_analyzer import (
    GPU_MODEL_PROFILES,
    ReportGenerator,
    get_recommended_gpu_profile,
)


def test_gpu_profiles_include_new_tiers():
    assert ["4gb", "6gb", "8gb", "16gb", "24gb", "32gb"] == list(GPU_MODEL_PROFILES.keys())


def test_recommended_gpu_profile_thresholds_cover_all_tiers():
    assert get_recommended_gpu_profile(3.4) == ""
    assert get_recommended_gpu_profile(3.5) == "4gb"
    assert get_recommended_gpu_profile(5.5) == "6gb"
    assert get_recommended_gpu_profile(7.5) == "8gb"
    assert get_recommended_gpu_profile(12.0) == "16gb"
    assert get_recommended_gpu_profile(20.0) == "24gb"
    assert get_recommended_gpu_profile(28.0) == "32gb"


def test_new_gpu_models_have_stable_short_names():
    assert get_pipeline_model_short_name("qwen3.5:2b") == "qwen35-2b"
    assert get_pipeline_model_short_name("qwen3-vl:2b") == "qwen3vl-2b"
    assert get_pipeline_model_short_name("qwen3.5:4b") == "qwen35-4b"
    assert get_pipeline_model_short_name("qwen3-vl:4b") == "qwen3vl-4b"
    assert ReportGenerator.get_model_short_name("qwen3.5:2b") == "qwen35-2b"
    assert ReportGenerator.get_model_short_name("qwen3-vl:4b") == "qwen3vl-4b"
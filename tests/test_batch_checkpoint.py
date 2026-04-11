import json

from batch_checkpoint import (
    BatchCheckpointManager,
    build_batch_job_config,
    build_batch_signature,
    get_default_checkpoint_path,
)


def create_checkpoint_manager(tmp_path, export_formats=None):
    output_dir = tmp_path / "relatorios"
    output_dir.mkdir(exist_ok=True)

    job_config = build_batch_job_config(
        selected_models=[("gpt-5.4-mini", "openai")],
        analysis_mode="geral",
        ocr_engine="glm-ocr",
        export_formats=export_formats or ["md"],
        yolo_model="yolo11s",
    )
    return BatchCheckpointManager(
        checkpoint_path=get_default_checkpoint_path(output_dir),
        job_signature=build_batch_signature(job_config),
        job_config=job_config,
    )


def test_checkpoint_resume_skips_completed_images(tmp_path):
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")

    report_a = tmp_path / "relatorios" / "a_gpt54-mini.md"
    report_a.parent.mkdir(exist_ok=True)
    report_a.write_text("ok", encoding="utf-8")

    manager = create_checkpoint_manager(tmp_path)
    manager.prepare_run([image_a, image_b], resume=False)
    manager.record_result(
        image_path=image_a,
        success_count=1,
        failed_count=0,
        reports=[report_a],
        errors=[],
    )

    resume_state = manager.prepare_run([image_a, image_b], resume=True)

    assert resume_state.resumed is True
    assert resume_state.pending_images == [image_b]
    assert len(resume_state.skipped_entries) == 1
    assert resume_state.skipped_entries[0]["image_name"] == "a.jpg"


def test_checkpoint_resets_when_batch_signature_changes(tmp_path):
    image_a = tmp_path / "a.jpg"
    image_a.write_bytes(b"a")

    report_a = tmp_path / "relatorios" / "a_gpt54-mini.md"
    report_a.parent.mkdir(exist_ok=True)
    report_a.write_text("ok", encoding="utf-8")

    manager = create_checkpoint_manager(tmp_path)
    manager.prepare_run([image_a], resume=False)
    manager.record_result(
        image_path=image_a,
        success_count=1,
        failed_count=0,
        reports=[report_a],
        errors=[],
    )

    resumed_manager = create_checkpoint_manager(tmp_path, export_formats=["md", "json"])
    resume_state = resumed_manager.prepare_run([image_a], resume=True)
    checkpoint_path = get_default_checkpoint_path(tmp_path / "relatorios")
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))

    assert resume_state.resumed is False
    assert resume_state.pending_images == [image_a]
    assert resume_state.skipped_entries == []
    assert "configuração do lote mudou" in resume_state.reset_reason
    assert payload["entries"] == {}


def test_checkpoint_invalidates_completed_entry_when_report_is_missing(tmp_path):
    image_a = tmp_path / "a.jpg"
    image_a.write_bytes(b"a")

    missing_report = tmp_path / "relatorios" / "missing.md"

    manager = create_checkpoint_manager(tmp_path)
    manager.prepare_run([image_a], resume=False)
    manager.record_result(
        image_path=image_a,
        success_count=1,
        failed_count=0,
        reports=[missing_report],
        errors=[],
    )

    resume_state = manager.prepare_run([image_a], resume=True)
    checkpoint_path = get_default_checkpoint_path(tmp_path / "relatorios")
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))

    assert resume_state.resumed is True
    assert resume_state.pending_images == [image_a]
    assert resume_state.skipped_entries == []
    assert payload["entries"] == {}
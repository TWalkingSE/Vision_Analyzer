from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


CHECKPOINT_FILENAME = ".vision_batch_checkpoint.json"
CHECKPOINT_VERSION = 1


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_selected_models(selected_models: Optional[Iterable[Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []

    for entry in selected_models or []:
        if isinstance(entry, dict):
            model_name = entry.get("name") or entry.get("model") or ""
            model_type = entry.get("type") or ""
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            model_name, model_type = entry[0], entry[1]
        else:
            raise ValueError("selected_models entries must be tuples or dicts")

        normalized.append({
            "name": str(model_name),
            "type": str(model_type),
        })

    return normalized


def build_batch_job_config(
    *,
    selected_models: Optional[Iterable[Any]],
    analysis_mode: str,
    ocr_engine: str,
    export_formats: Optional[Iterable[str]] = None,
    yolo_model: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "analysis_mode": analysis_mode,
        "ocr_engine": ocr_engine,
        "export_formats": sorted({str(fmt) for fmt in (export_formats or ["md"])}),
        "selected_models": _normalize_selected_models(selected_models),
        "yolo_model": str(yolo_model or ""),
    }


def build_batch_signature(job_config: dict[str, Any]) -> str:
    payload = {
        "version": CHECKPOINT_VERSION,
        **job_config,
    }
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def get_default_checkpoint_path(output_dir: Path) -> Path:
    return Path(output_dir) / CHECKPOINT_FILENAME


@dataclass(frozen=True)
class ResumePreparation:
    pending_images: list[Path]
    skipped_entries: list[dict[str, Any]]
    resumed: bool
    reset_reason: Optional[str] = None


class BatchCheckpointManager:
    def __init__(self, checkpoint_path: Path, job_signature: str, job_config: dict[str, Any]):
        self.checkpoint_path = Path(checkpoint_path)
        self.job_signature = job_signature
        self.job_config = job_config
        self._state = self._build_empty_state()
        self._has_existing_checkpoint = False
        self._load_error: Optional[str] = None
        self._load()

    def _build_empty_state(self) -> dict[str, Any]:
        now = _timestamp()
        return {
            "version": CHECKPOINT_VERSION,
            "job_signature": self.job_signature,
            "job_config": self.job_config,
            "created_at": now,
            "updated_at": now,
            "entries": {},
        }

    def _load(self) -> None:
        if not self.checkpoint_path.exists():
            return

        try:
            payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._load_error = str(exc)
            return

        if not isinstance(payload, dict):
            self._load_error = "checkpoint payload must be a JSON object"
            return

        entries = payload.get("entries")
        if not isinstance(entries, dict):
            self._load_error = "checkpoint entries must be a JSON object"
            return

        payload.setdefault("version", CHECKPOINT_VERSION)
        payload.setdefault("job_signature", "")
        payload.setdefault("job_config", {})
        payload.setdefault("created_at", _timestamp())
        payload.setdefault("updated_at", payload["created_at"])
        self._state = payload
        self._has_existing_checkpoint = True

    def _save(self) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._state["updated_at"] = _timestamp()
        self.checkpoint_path.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._has_existing_checkpoint = True
        self._load_error = None

    def _reset(self) -> None:
        self._state = self._build_empty_state()
        self._save()

    def _normalize_path(self, image_path: Path) -> str:
        return os.path.normcase(str(Path(image_path).resolve()))

    def _is_completed_entry(self, entry: Optional[dict[str, Any]]) -> bool:
        if not entry or entry.get("status") != "completed":
            return False

        reports = entry.get("reports") or []
        return bool(reports) and all(Path(report).exists() for report in reports)

    def prepare_run(self, images: Iterable[Path], *, resume: bool) -> ResumePreparation:
        images = [Path(image) for image in images]

        if not resume:
            self._reset()
            return ResumePreparation(pending_images=images, skipped_entries=[], resumed=False)

        if self._load_error:
            reason = f"checkpoint inválido: {self._load_error}"
            self._reset()
            return ResumePreparation(pending_images=images, skipped_entries=[], resumed=False, reset_reason=reason)

        if not self._has_existing_checkpoint:
            self._reset()
            return ResumePreparation(pending_images=images, skipped_entries=[], resumed=False)

        if self._state.get("version") != CHECKPOINT_VERSION:
            self._reset()
            return ResumePreparation(
                pending_images=images,
                skipped_entries=[],
                resumed=False,
                reset_reason="versão do checkpoint incompatível; checkpoint reiniciado",
            )

        if self._state.get("job_signature") != self.job_signature:
            self._reset()
            return ResumePreparation(
                pending_images=images,
                skipped_entries=[],
                resumed=False,
                reset_reason="configuração do lote mudou; checkpoint reiniciado",
            )

        skipped_entries: list[dict[str, Any]] = []
        pending_images: list[Path] = []
        stale_paths: list[str] = []

        for image_path in images:
            normalized = self._normalize_path(image_path)
            entry = self._state["entries"].get(normalized)

            if self._is_completed_entry(entry):
                skipped_entries.append(entry)
            else:
                pending_images.append(image_path)
                if entry and entry.get("status") == "completed":
                    stale_paths.append(normalized)

        for stale_path in stale_paths:
            self._state["entries"].pop(stale_path, None)

        self._save()
        return ResumePreparation(
            pending_images=pending_images,
            skipped_entries=skipped_entries,
            resumed=True,
        )

    def record_result(
        self,
        *,
        image_path: Path,
        success_count: int,
        failed_count: int,
        reports: Iterable[Path | str],
        errors: Optional[Iterable[str]] = None,
    ) -> None:
        report_paths = [str(Path(report)) for report in reports]
        if success_count > 0 and failed_count == 0:
            status = "completed"
        elif success_count > 0:
            status = "partial"
        else:
            status = "failed"

        self._state["entries"][self._normalize_path(image_path)] = {
            "image_path": str(Path(image_path).resolve()),
            "image_name": Path(image_path).name,
            "status": status,
            "success_count": success_count,
            "failed_count": failed_count,
            "reports": report_paths,
            "errors": list(errors or []),
            "updated_at": _timestamp(),
        }
        self._save()
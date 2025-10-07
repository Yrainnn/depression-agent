import sys
import types
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def _stub_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_jinja = types.ModuleType("jinja2")

    class _FakeTemplate:
        def render(self, **_: object) -> str:
            return "<html></html>"

    class _FakeEnv:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def from_string(self, _template: str) -> _FakeTemplate:
            return _FakeTemplate()

    fake_jinja.Environment = lambda *args, **kwargs: _FakeEnv(*args, **kwargs)  # type: ignore[attr-defined]
    fake_jinja.select_autoescape = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "jinja2", fake_jinja)

    fake_weasyprint = types.ModuleType("weasyprint")

    class _FakeHTML:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def write_pdf(self, path: str) -> None:
            Path(path).write_bytes(b"PDF")

    fake_weasyprint.HTML = lambda *args, **kwargs: _FakeHTML(*args, **kwargs)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "weasyprint", fake_weasyprint)

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    config_stub = types.ModuleType("packages.common.config")

    class _Settings:
        OSS_ENDPOINT = ""
        OSS_BUCKET = ""
        OSS_PREFIX = ""
        OSS_ACCESS_KEY_ID = ""
        OSS_ACCESS_KEY_SECRET = ""
        OSS_BASE_URL = ""

    config_stub.settings = _Settings()

    packages_stub = types.ModuleType("packages")
    common_stub = types.ModuleType("packages.common")
    packages_stub.common = common_stub
    common_stub.config = config_stub

    monkeypatch.setitem(sys.modules, "packages", packages_stub)
    monkeypatch.setitem(sys.modules, "packages.common", common_stub)
    monkeypatch.setitem(sys.modules, "packages.common.config", config_stub)


def test_build_pdf_uploads_to_oss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    uploads = []

    class FakeOSS:
        enabled = True

        def store_artifact(self, sid: str, category: str, path: Path, metadata=None):
            uploads.append({
                "sid": sid,
                "category": category,
                "path": Path(path),
                "metadata": metadata,
            })
            return f"https://oss.example/{category}/{Path(path).name}"

    import services.report.build as report_build

    class FakeHTML:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def write_pdf(self, path: str) -> None:
            Path(path).write_bytes(b"PDF")

    monkeypatch.setattr(report_build, "oss_client", FakeOSS())
    monkeypatch.setattr(report_build, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(report_build, "HTML", lambda *a, **kw: FakeHTML())

    from services.report.build import build_pdf

    result = build_pdf(
        "sid-123",
        {
            "items": [
                {"item_id": "H01", "score": 1, "question": "情绪"},
            ],
            "total_score": {"得分序列": "1", "pre_correction_total": 1},
        },
    )

    assert result["report_url"].startswith("https://oss.example/reports/")
    assert uploads and uploads[0]["category"] == "reports"
    assert uploads[0]["path"].exists()
    assert uploads[0]["metadata"]["format"] == "pdf"


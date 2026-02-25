"""Tests for JSON-to-Raven conversion utility."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def _load_json_to_raven_module():
    """Load Raven json_to_raven.py as a module."""
    module_path = (
        Path(__file__).resolve().parents[1]
        / "lemurcalls"
        / "whisperformer"
        / "Raven"
        / "json_to_raven.py"
    )
    spec = importlib.util.spec_from_file_location("json_to_raven_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestJsonToRaven:
    def test_without_score_column(self, tmp_path):
        """Export should not include Score when score key is missing."""
        module = _load_json_to_raven_module()
        input_json = tmp_path / "preds.jsonr"
        output_txt = tmp_path / "preds.txt"

        payload = {
            "onset": [0.1, 1.2],
            "offset": [0.5, 1.5],
            "cluster": ["m", "h"],
        }
        input_json.write_text(json.dumps(payload), encoding="utf-8")

        module.json_to_raven_selection_table(str(input_json), str(output_txt))

        df = pd.read_csv(output_txt, sep="\t")
        assert "Score" not in df.columns
        assert list(df["Cluster"]) == ["m", "h"]

    def test_with_score_column(self, tmp_path):
        """Export should include Score when score key is present and valid."""
        module = _load_json_to_raven_module()
        input_json = tmp_path / "preds.jsonr"
        output_txt = tmp_path / "preds.txt"

        payload = {
            "onset": [0.1, 1.2],
            "offset": [0.5, 1.5],
            "cluster": ["m", "h"],
            "score": [0.81, 0.92],
        }
        input_json.write_text(json.dumps(payload), encoding="utf-8")

        module.json_to_raven_selection_table(str(input_json), str(output_txt))

        df = pd.read_csv(output_txt, sep="\t")
        assert "Score" in df.columns
        assert df["Score"].tolist() == pytest.approx([0.81, 0.92])

    def test_score_length_mismatch_is_skipped_with_warning(self, tmp_path, capsys):
        """Mismatched score length should skip Score column and print warning."""
        module = _load_json_to_raven_module()
        input_json = tmp_path / "preds.jsonr"
        output_txt = tmp_path / "preds.txt"

        payload = {
            "onset": [0.1, 1.2],
            "offset": [0.5, 1.5],
            "cluster": ["m", "h"],
            "score": [0.81],  # mismatch on purpose
        }
        input_json.write_text(json.dumps(payload), encoding="utf-8")

        module.json_to_raven_selection_table(str(input_json), str(output_txt))
        captured = capsys.readouterr()

        df = pd.read_csv(output_txt, sep="\t")
        assert "Score" not in df.columns
        assert "Score column skipped" in captured.out

    def test_missing_required_key_raises_value_error(self, tmp_path):
        """Missing required keys should raise a ValueError."""
        module = _load_json_to_raven_module()
        input_json = tmp_path / "preds.jsonr"
        output_txt = tmp_path / "preds.txt"

        payload = {
            "onset": [0.1],
            "offset": [0.5],
            # cluster intentionally missing
        }
        input_json.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="missing keys"):
            module.json_to_raven_selection_table(str(input_json), str(output_txt))

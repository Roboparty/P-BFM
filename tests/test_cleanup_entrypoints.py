"""The standalone backward exporter keeps its CLI and checkpoint defaults."""
from unittest.mock import Mock

import tyro

from pbfm.inference.export import backward_encoder
from scripts.export_backward_encoder_onnx import main


def test_backward_export_cli_uses_shared_function(monkeypatch, tmp_path):
    (tmp_path / "checkpoint").mkdir()
    load = Mock(return_value=object())
    export = Mock(return_value=tmp_path / "backward.onnx")
    monkeypatch.setattr(backward_encoder, "load_model_from_checkpoint_dir", load)
    monkeypatch.setattr(backward_encoder, "export_backward_encoder_from_model", export)
    assert main is backward_encoder.export_backward_encoder_from_checkpoint
    result = tyro.cli(main, args=["--model-folder", str(tmp_path), "--device", "cpu", "--no-verify"])
    assert result == export.return_value
    load.assert_called_once_with(tmp_path / "checkpoint", device="cpu")
    assert export.call_args.args == (load.return_value, tmp_path / "exported" / "backward_encoder.onnx")
    assert export.call_args.kwargs == {"batch_size": 1, "opset": 13, "verify": False, "atol": 1e-4, "rtol": 1e-4}

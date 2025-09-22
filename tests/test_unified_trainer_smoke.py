"""Smoke tests for the unified trainer module."""

from pathlib import Path
import importlib.machinery
import types
import sys


# Ensure the project root is on the module search path so ``src`` can be imported.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Provide lightweight stubs for optional dependencies imported during module import.
peft_stub = types.ModuleType("peft")
peft_stub.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)


class _DummyLoraConfig:
    def __init__(self, *args, **kwargs):  # pragma: no cover - simple stub
        self.args = args
        self.kwargs = kwargs


def _return_model(model, *args, **kwargs):  # pragma: no cover - simple stub
    return model


peft_stub.LoraConfig = _DummyLoraConfig
peft_stub.get_peft_model = _return_model
peft_stub.prepare_model_for_kbit_training = _return_model
peft_stub.PeftModel = type(
    "_DummyPeftModel",
    (),
    {"from_pretrained": classmethod(lambda cls, *a, **k: None)},
)
sys.modules.setdefault("peft", peft_stub)


def test_unified_trainer_can_be_imported():
    """Ensure the UnifiedChessTrainer class can be imported and instantiated."""

    from src.training.unified_trainer import UnifiedChessTrainer

    trainer = UnifiedChessTrainer()

    # The trainer should expose the key collaborators configured during __init__.
    assert trainer.model_validator is not None
    assert isinstance(trainer.eval_datasets, dict)

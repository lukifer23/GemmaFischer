import tempfile
from pathlib import Path
from unittest.mock import patch

from data.prepare_dataset import convert_and_save


@patch('datasets.load_dataset')
def test_sample_streaming(mock_load_dataset):
    """Ensure sample mode streams only n items."""

    class DummyIterable:
        def __iter__(self):
            for _ in range(150):
                yield {"task": "t", "input": "i", "expected_output": "o"}

    mock_load_dataset.return_value = DummyIterable()

    with tempfile.TemporaryDirectory() as temp_dir:
        convert_and_save(temp_dir, full=False)
        output_file = Path(temp_dir) / "chess_conversations.json"
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

    assert len(lines) == 100
    mock_load_dataset.assert_called_once_with(
        "Thytu/ChessInstruct", split="train", streaming=True
    )

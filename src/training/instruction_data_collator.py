from dataclasses import dataclass
from typing import List, Dict

import torch
from transformers import PreTrainedTokenizerBase


def _append_eos(tokenizer: PreTrainedTokenizerBase, ids: List[int]) -> List[int]:
    """Append EOS token to the list if the tokenizer defines one."""
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        return ids + [eos_id]
    return ids


@dataclass
class InstructionDataCollator:
    """Collate function for instruction-style datasets.

    Each example must contain ``prompt`` and ``response`` fields. The prompt tokens
    are masked out in the ``labels`` tensor by assigning ``-100`` so that loss is
    only computed on the response portion.
    """

    tokenizer: PreTrainedTokenizerBase
    max_length: int = 2048

    def __call__(self, examples: List[Dict[str, str]]):
        batch_input_ids: List[List[int]] = []
        batch_labels: List[List[int]] = []

        for ex in examples:
            prompt = ex.get("prompt", "")
            response = ex.get("response", "")

            prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
            response_ids = self.tokenizer(response, add_special_tokens=False).input_ids
            response_ids = _append_eos(self.tokenizer, response_ids)

            input_ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + response_ids

            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]
                labels = labels[: self.max_length]

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)

        max_len = max(len(ids) for ids in batch_input_ids)
        pad_id = self.tokenizer.pad_token_id or 0

        padded_inputs = []
        padded_labels = []
        attention_masks = []

        for ids, lbls in zip(batch_input_ids, batch_labels):
            pad_len = max_len - len(ids)
            padded_inputs.append(ids + [pad_id] * pad_len)
            attention_masks.append([1] * len(ids) + [0] * pad_len)
            padded_labels.append(lbls + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

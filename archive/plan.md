## Project Status Update (September 2025)

### ✅ Completed Milestones (Updated)
- **Environment Setup**: Python 3.13 venv with MPS-enabled PyTorch on Apple Silicon
- **Model Loading**: Successfully loaded `unsloth/gemma-3-270m-it` base model
- **Dataset Processing**: Converted ChessInstruct dataset to training format
- **LoRA Training**: Initial training completed to checkpoint-1200 (2000 steps)
- **Resume Functionality**: Verified checkpoint resuming works
- **Dataset Refinement**: Created refined Q&A format dataset
- **Hyperparameter Tuning**: Tested lower learning rates (1e-5) and batch configurations
- **Focused Dataset Creation**: Created 1000 focused Q&A examples with chess concepts
- **Extended Training**: Successfully trained for 100 steps with significant loss improvement (2.99 → 1.16)
- **Model Improvement**: Fine-tuned model now generates chess-related responses

### ⚠️ Current Issues Identified (Updated)
- **Dataset Quality Problem**: Current format uses extremely long move sequences (100+ moves) with generic answers like "completes the sequence logically"
- **Training Effectiveness**: Model not learning chess concepts properly - produces non-chess responses
- **Resume Training**: Progress tracking issues in resume functionality
- **Output Quality**: Fine-tuned model still generates irrelevant responses
- **Response Accuracy**: Model generates chess-related content but with some inaccuracies in explanations

### 🎯 Success Criteria (Updated)
- Model should answer basic chess questions correctly
- Responses should be chess-related and relevant
- Training should converge to meaningful loss values
- **ACHIEVED**: Model now generates chess-related responses instead of random content
- **ACHIEVED**: Loss decreased significantly with focused dataset (2.99 → 1.16)
- **ACHIEVED**: Model correctly identifies basic chess concepts (castling = king + rook)

### 🔧 Technical Findings
- MPS backend works well for Gemma-3 270M model size
- LoRA training converges but requires better data format
- Current dataset format (long move sequences) not suitable for Q&A learning
- Need to create focused, educational Q&A pairs instead of puzzle-like format

### 📋 Next Steps (Priority Order)

1. **Dataset Overhaul** (HIGH PRIORITY):
   - Create shorter, focused Q&A pairs
   - Use actual chess concepts and explanations
   - Example: "Q: What is the best response to e4? A: e5 (opening the center) or c5 (Sicilian Defense)"

2. **Training Optimization**:
   - Fix resume functionality for proper step tracking
   - Test longer training runs with improved dataset
   - Implement proper evaluation metrics

3. **Evaluation Enhancement**:
   - Create chess-specific evaluation scripts
   - Compare base vs fine-tuned model performance
   - Analyze response quality and relevance

4. **Output Debugging**:
   - Test different prompt formats
   - Analyze model attention patterns
   - Improve response formatting

### 🎯 Success Criteria
- Model should answer basic chess questions correctly
- Responses should be chess-related and relevant
- Training should converge to meaningful loss values
- Resume functionality should work reliably

---

*Original plan continues below...*
Fine-tuning an LLM like Gemma 3 on a MacBook Pro M3 (Apple Silicon) is feasible thanks to Unsloth’s optimized training pipeline (up to 2× faster and 70% less VRAM usage than standard methods[1]). Start by preparing a Python environment with the necessary libraries:
•	Python 3.10+ (ensure you have a recent version).
•	PyTorch 2.x with MPS (Metal Performance Shaders) support. On Apple Silicon, you can install the CPU/MPS build via pip: pip install torch torchvision torchaudio. This will enable GPU acceleration on the M3’s GPU for training.
•	Hugging Face Transformers & Datasets for model and data handling.
•	Unsloth (for efficient fine-tuning with LoRA).
•	Optional: bitsandbytes (for 8-bit optimizers/quantization) – note that this library may not natively support Apple Silicon, so you can skip or use CPU-based alternatives if installation fails.
Installation commands: Create a virtual environment (conda or venv) and install packages:
pip install transformers datasets accelerate peft trl huggingface_hub
pip install unsloth  # installs Unsloth (may also pull in unsloth_zoo)
# Optional, if bitsandbytes is needed (might require compile on M-series):
pip install bitsandbytes
Ensure that Unsloth is installed last (as it may bundle specific versions of dependencies). On macOS, no special CUDA setup is needed, but for MPS support you should use the latest PyTorch. After installation, verify that PyTorch can detect the MPS device:
import torch
print(torch.backends.mps.is_available())  # True if MPS is available
If True, you’re ready to use the Apple GPU; if not, you may fallback to CPU (the 270M model is small enough for CPU training as well). There’s no separate GPU driver needed – MPS is built into Mac’s system libraries.
2. Loading the Model
We will use the Gemma 3 270M (Italian) model from Unsloth’s Hugging Face hub, which is a lightweight LLM (~0.5 GB RAM usage) ideal for experimenting with chess tasks[2]. Unsloth provides a convenience class FastLanguageModel to load models efficiently. Here’s how to load the base model and tokenizer:
from unsloth import FastLanguageModel
import torch

MODEL_NAME = "unsloth/gemma-3-270m-it"  # Gemma 3 (270M parameters, Italian, text-only)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name       = MODEL_NAME,
    max_seq_length   = 2048,     # context length
    dtype            = None,     # let Unsloth auto-select (float32 on CPU/MPS)
    load_in_4bit     = False,    # optionally use 4-bit quantization to save memory
    full_finetuning  = False     # we will do LoRA fine-tuning (not full model fine-tuning)
)
This will download the model weights from Hugging Face and initialize them. We set full_finetuning=False so that Unsloth knows we intend to apply parameter-efficient fine-tuning (LoRA) rather than updating all weights. The FastLanguageModel.from_pretrained method can also handle device placement and quantization under the hood (for example, setting load_in_4bit=True would load a 4-bit quantized model to reduce memory). We left dtype=None to use default precision (float32 on Apple devices, since GPU half-precision might not be fully supported on MPS).
Note on device: By default, the model will load on CPU. To use the MPS GPU, you can move it after loading: model.to("mps"). Alternatively, set environment variable PYTORCH_ENABLE_MPS_FALLBACK=1 to allow PyTorch operations to run on MPS with CPU fallback for unsupported ops. In practice, for this 270M model, training on CPU vs. MPS may both be relatively fast; using MPS can still give a speed boost if supported.
3. LoRA Configuration
With the base model loaded, we next configure LoRA (Low-Rank Adaptation) adapters to fine-tune the model efficiently. LoRA will add a few trainable weight matrices (of rank r) to the model’s key weight matrices, while freezing the original model weights. This dramatically reduces the number of trainable parameters – often only 0.1–2% of the full model parameters need to be trained, making fine-tuning feasible on limited hardware.
Using Unsloth, we wrap the model with LoRA by calling FastLanguageModel.get_peft_model. We specify LoRA hyperparameters like the rank r, scaling factor lora_alpha, dropout, and which parts of the model to target:
Applying LoRA adapters to the Gemma model. The LoRA low-rank matrices ($A$ and $B$) are inserted for specified target modules, while the original weights $W$ remain frozen.
from unsloth import FastLanguageModel  # ensure FastLanguageModel is imported
model = FastLanguageModel.get_peft_model(
    model,
    r                   = 128,  # LoRA rank (e.g. 8, 16, 32, ... 128)
    lora_alpha          = 128,  # LoRA scaling factor
    lora_dropout        = 0.0,  # dropout for LoRA layers (0 for no dropout)
    target_modules      = [
        "q_proj", "k_proj", "v_proj", "o_proj",        # attention projections
        "gate_proj", "up_proj", "down_proj"           # feed-forward (MLP) projections
    ],
    bias                = "none",      # don't train any biases
    use_gradient_checkpointing = "unsloth",  # enable gradient checkpointing (saves VRAM)
    use_rslora          = False,       # disable Rank-Scaled LoRA (use standard scaling)
    random_state        = 3407        # seed for reproducibility
    # loftq_config=None (we're not using LoRA + quantization hybrid in this run)
)
Let’s break down the key hyperparameters we used:
•	r (LoRA rank): The rank of the low-rank update matrices. A higher r means the LoRA layers can capture more information but with more parameters to train[3]. Lower r uses fewer parameters but may underfit if too low[4]. Common values range from 8 up to 64 or 128 for larger models; here we chose 128 given our model is small and we want to capture chess knowledge.
•	lora_alpha: The scaling factor for LoRA updates. This is essentially a multiplication factor on the LoRA outputs before adding to the model weights. A higher alpha increases the influence of the LoRA learned parameters (faster learning but potentially less stable[5]), while a lower alpha means the LoRA updates are applied more subtly (may need more training steps)[6]. Often lora_alpha is set equal to or twice the value of r (here we use 128).
•	lora_dropout: Dropout probability for LoRA layers. Using dropout can regularize the LoRA updates to prevent overfitting[7]. We set this to 0 for maximum capacity, since our dataset is fairly large; if training loss plateaus or overfits, a small dropout (e.g. 0.05) could be tried.
•	target_modules: This is the list of model sub-layers to apply LoRA to. We target the Transformer’s attention projections (q_proj, k_proj, v_proj, o_proj) and the feed-forward network projections (gate_proj, up_proj, down_proj). These names correspond to weight matrices in Gemma’s architecture: query/key/value matrices in self-attention, the output projection of attention, and the gated feed-forward layers[8][9]. By applying LoRA to all of these, we allow the model to adjust both attention and MLP behavior. (Targeting more modules increases trainable params; one could choose a subset like just q_proj/v_proj for efficiency.)
•	bias: Set to "none" so we do not train any bias terms. (Other options in PEFT allow training biases separately, but leaving biases fixed is common for LoRA.)
•	use_gradient_checkpointing: By setting this to "unsloth", we enable Unsloth’s optimized gradient checkpointing. Gradient checkpointing saves memory by not storing intermediate activations, at the cost of some compute (layers are recomputed during backprop). Unsloth’s implementation can save ~30% VRAM and allow larger batch sizes[10][11]. On Apple M3 (with unified memory), this helps keep memory usage low.
•	use_rslora: We leave this False. RSLora stands for Rank-Stabilized LoRA, an experimental technique where the effective LoRA scaling is adjusted to lora_alpha/√r instead of lora_alpha/r to improve stability[12]. It’s recommended in some cases, but here we stick to standard LoRA.
•	random_state: A seed for random number generation to make results reproducible (affects LoRA weight initialization and any data shuffling).
•	loftq_config: Left as None (not used). LoftQ is an advanced feature where the base model is quantized on the fly and LoRA layers are initialized in a special way[13]. This can further reduce memory usage but requires careful setup (and is beyond our current scope).
After this step, model is now a PEFT model (specifically a PeftModel wrapping the original Gemma model). We have ~7.45 million trainable parameters (for r=128 on all those layers) which is about 0.19% of Gemma 3’s total weights. This is a small fraction, meaning training will be much faster and require far less memory than updating all 270M parameters.
4. Dataset Loading and Preprocessing
For the fine-tuning task, we use the ChessInstruct dataset by Thytu[14]. This dataset contains 100k chess instruction examples. Each sample has a prompt describing a chess task, some input data (like a sequence of moves or a board state), and the expected output. We will focus on one task type: finding a missing move in a nearly-complete game.
Loading the dataset: We can load it directly from Hugging Face with the datasets library:
from datasets import load_dataset

dataset_name = "Thytu/ChessInstruct"
dataset = load_dataset(dataset_name, split="train")
print(len(dataset))  # should be 100000
Each entry in dataset has the following fields[15]:
•	task – The instruction prompt describing the task (e.g. “Given an incomplete set of chess moves and the game’s final score, write the last missing chess move. Input Format: ... Output Format: ...”).
•	input – Supplementary input data, often a JSON structure containing moves and possibly a result. For the missing-move task, input includes a list of moves (with "?" in place of the missing one) and the final game result.
•	expected_output – The correct answer, typically a JSON-formatted string with the solution (e.g. {"missing move": "e6f7"} in string form).
•	KIND – A label for the type of task (e.g. FIND_LAST_MOVE, etc., not crucial for us except to filter types if needed).
To illustrate, here is what a sample looks like (from the “find last move” subset):
A sample from the ChessInstruct dataset (simplified). The task describes the problem, input contains a list of moves ("moves") and the final "result", and expected_output is the correct missing move.
In the above example, the model should read the moves (many in algebraic notation like "e2e4" meaning pawn from e2 to e4) and the final result "1/2-1/2", then determine the "missing move": "e6f7". Our goal is to train Gemma 3 to produce such answers when given similar prompts.
Preprocessing for chat-style fine-tuning: Unsloth (and the Hugging Face SFTTrainer we’ll use) expects data in a conversational format – essentially a sequence of chat messages (user/system/assistant). Even though our tasks are single-turn instructions, we can format each example as a mini dialogue.
We will do the following: 1. Combine the task and input into a prompt that the model will see (as if from a user or system), and 2. Provide the expected_output as the target response (assistant message).
We can choose to put the instructional prompt as a system message (since it’s like a context or rule) and the moves as the user question, or simply put everything into the user message. Here, we’ll use a system + user split for clarity: - System role: the general instruction template (task string, which includes directions about input/output format). - User role: the specific instance’s input (the moves and game state). - Assistant role: the correct output (solution).
First, it’s good practice to standardize any conversation formatting. Unsloth provides standardize_data_formats to handle datasets that are already in conversation form (like ShareGPT format) by converting keys to "role" and "content"[16][17]. In our case, ChessInstruct is not in conversation format yet, so initially this call will do nothing, but we include it for completeness (and it would be essential if we had a multi-turn or ShareGPT-style dataset):
from unsloth.chat_templates import standardize_data_formats

# If the dataset had conversation dicts, this would unify them.
dataset = standardize_data_formats(dataset)
Next, we define a mapping function convert_to_chatml to convert each example into the ChatML-style conversation format (ChatML is the chat format used by OpenAI models, which Unsloth supports):
def convert_to_chatml(example):
    # Combine task and input into prompt messages
    return {
        "conversations": [
            {"role": "system",    "content": example["task"]},
            {"role": "user",      "content": str(example["input"])},
            {"role": "assistant", "content": example["expected_output"]}
        ]
    }

dataset = dataset.map(convert_to_chatml)
print(dataset[0]["conversations"])
In this function, we take the original fields: - example["task"] (a string) becomes a system message. It provides context on what the model should do (e.g. “Given an incomplete set of moves... Output format: the missing move”). - example["input"] (which is a Python dict for moves/result) is converted to string and set as the user message content. (We simply cast to str here; for a cleaner prompt, one might format the moves into a single string of moves, but using the dict string is a simple approach. It will appear as '{'moves': [...], 'result': '1-0'}' in the prompt.) - example["expected_output"] (already a string, e.g. '{"missing move": "e6f7"}') becomes the assistant message that the model should output.
After mapping, each dataset item has a "conversations" field: a list of {"role": ..., "content": ...} messages. We should again standardize to ensure roles are exactly "system", "user", "assistant" (in our case they already are):
dataset = standardize_data_formats(dataset)
Now the data is ready. Essentially, we have transformed each chess problem into a mini chat transcript: the system sets the stage with the instruction, the user provides the game moves and state, and the assistant provides the answer. This is suitable for feeding into a chat-tuned model like Gemma 3. (If Gemma 3 had a specific chat template, we could use tokenizer = get_chat_template(tokenizer, chat_template="chatml") and then dataset.map with tokenizer.apply_chat_template to produce a final text. However, Unsloth’s training utilities can also handle the "conversations" format directly, as we’ll see.)
5. Trainer Definition
With our model (with LoRA) and dataset ready, the next step is to set up the training configuration and trainer. We’ll use the SFTTrainer from Hugging Face’s TRL (Transformer Reinforcement Learning) library, which is integrated with Unsloth for supervised fine-tuning of language models. SFTTrainer works similarly to transformers.Trainer but is tailored for language model fine-tuning (it can handle the chat data format and uses techniques like packing sequences, etc.).
We also use SFTConfig to encapsulate training hyperparameters:
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model       = model,
    tokenizer   = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        max_seq_length = 1024,           # limit sequence length (if data is longer, it will be truncated/padded)
        per_device_train_batch_size = 2,  # small batch size per device (MPS)
        gradient_accumulation_steps = 4,  # accumulate gradients to simulate batch_size = 2*4 = 8
        max_steps = 100,                 # train for 100 steps (for demo; use None or set num_train_epochs for full training)
        learning_rate = 5e-5,            # a relatively low LR for fine-tuning (LoRA can often use 2e-4; adjust as needed)
        fp16 = False, bf16 = False,      # not enabling mixed precision on MPS/CPU
        logging_steps = 5,               # log training progress every 5 steps
        optim = "adamw_8bit",            # use 8-bit Adam optimizer (requires bitsandbytes; use "adamw_hf" if bnb not available)
        weight_decay = 0.01,             # small weight decay to regularize
        seed = 3407                      # seed for reproducibility
    )
)
A few notes on these settings: - Batch size and accumulation: We use per_device_train_batch_size=2 and accumulate gradients for 4 steps. This effectively gives a batch of 8 examples per weight update, which helps stabilize training without using too much memory. You can adjust these based on memory limits (on 16GB RAM, even batch 8 or 16 might be okay for this model). - Max steps vs. epochs: Here we set max_steps=100 for a quick run. In practice, you might want to train for 1-3 epochs over the 100k data (which would be ~12,500 steps per epoch with batch 8). You could instead set num_train_epochs=2 and max_steps=None. We use a small number just to illustrate the process. - Learning rate: 5e-5 is a conservative learning rate for LoRA fine-tuning. Depending on the task, LoRA often uses around 2e-4[18] as a starting point. It’s wise to monitor loss and possibly adjust LR or use a LR scheduler (by default SFTTrainer will use linear decay). We keep weight_decay=0.01 to penalize large weights slightly (common default[19]). - Precision: We disable fp16 and bf16 since on Apple Silicon, full float32 is typically used. (PyTorch’s MPS backend does support mixed precision in inference, but for training it’s safer to stick to 32-bit to avoid any unsupported operation issues.) - Optimizer: We chose "adamw_8bit", which uses the bitsandbytes library to hold optimizer states in 8-bit precision, saving memory. If bitsandbytes is not installed or doesn’t work on your system, switch this to "adamw_hf" (the default AdamW implementation) or "adamw_torch". The trainer will still work, though memory usage will be a bit higher without 8-bit optimizers. - Tokenizer and data collator: The SFTTrainer knows how to handle our data structure. If your dataset has a "text" field of pre-formatted prompts, you’d specify dataset_text_field="text". In our case, we provided "conversations", and the trainer will automatically format these using the model’s chat template (it detects that field and will apply a default template akin to ChatML). Unsloth’s integration ensures the data is properly tokenized. We don’t have to manually call tokenizer.apply_chat_template here because SFTTrainer will do formatting internally (as long as the conversations are in the standardized role/content format, which we ensured).
6. Training the Model
Everything is set – now we can launch the fine-tuning. Simply call:
trainer_stats = trainer.train()
This will start the training loop. On an Apple M3, the 270M model with LoRA should train fairly quickly. You’ll see output logs showing the progress. For example, with our settings you might see something like:
Example training output (truncated). LoRA fine-tuning updates ~7.45M params (0.19% of the model). Training loss is printed every few steps, steadily decreasing.
In the sample output above, Unsloth prints a header indicating the setup (one GPU, mixed precision off, etc.). It shows that 7,450,624 parameters are being trained out of ~4 billion total (the log’s total seems to count some internal parameter sizing; the key is the percentage ~0.19% which aligns with LoRA). During training, the loss goes down step by step (e.g., from ~1.97 to ~1.69 over 30 steps shown). For a full training run on the entire dataset, you would continue for many more steps until convergence or until a chosen number of epochs is done.
Even with Apple’s M-series chip, this process is manageable: Gemma 270M is small, and LoRA’s efficiency plus Unsloth’s optimizations mean each step is quite fast. If you have a newer M3 Pro/Max, you could potentially bump batch size higher to speed up epoch completion. Monitor the GPU memory (use Activity Monitor) to ensure you’re not overloading VRAM – if you see swapping, reduce batch size or sequence length.
7. Saving and Using the Fine-Tuned Model
After training, we’ll want to save the LoRA adapters (and possibly the tokenizer config) for later use. Since we used the PEFT approach, saving the model will by default save only the LoRA weights and configuration (not the entire base model, which we can reuse from the hub). This is convenient as the adapter files are very small (a few MB).
Saving LoRA weights:
model.save_pretrained("gemma-chess-lora")
tokenizer.save_pretrained("gemma-chess-lora")
This will create a directory gemma-chess-lora/ containing files like adapter_model.bin (the LoRA weight diff) and adapter_config.json, as well as the tokenizer files (vocab, merges, etc.). You can also push this to Hugging Face Hub (with huggingface_hub) if you want to share it or load it from anywhere.
Loading the fine-tuned model later: To use the fine-tuned model for inference (e.g., answering chess questions), load the base model and then load the LoRA adapter on top:
base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-3-270m-it",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = False,      # can load in 4bit for inference if desired
    full_finetuning = False    # we will attach LoRA, so still not doing full finetune
)
from peft import PeftModel
model_lora = PeftModel.from_pretrained(base_model, "gemma-chess-lora")
Now model_lora is ready to generate answers with the LoRA weights applied. Remember to use the same tokenizer as before (base_tokenizer). If using the model in chat mode, format your prompt as we did during training (system message with instruction, user message with moves). For example:
prompt = [
  {"role": "system", "content": "Given an incomplete set of chess moves and the game's final score, write the last missing chess move."},
  {"role": "user", "content": '{"moves": ["e2e4","e7e5","g1f3", "g8f6", "f1c4", "b8c6", "?"], "result": "1-0"}'}
]
input_ids = base_tokenizer.apply_chat_template(prompt, tokenize=True)
output_ids = model_lora.generate(**input_ids, max_new_tokens=10)
print(base_tokenizer.decode(output_ids[0]))
This should produce the missing move (for instance, it might output {"missing move": "d7d6"} or just the move in some format, depending on how it learned to answer). You can post-process the output as needed (e.g., parse the JSON).
Alternatively, if you want to merge the LoRA weights into the base model for a standalone fine-tuned model (e.g. for exporting or deployment), you can do: model_lora = model_lora.merge_and_unload() – this will fold the LoRA changes into the base model’s weights. Then calling model_lora.save_pretrained("full-chess-model") would save a full model (no longer requiring Peft). Keep in mind that merged models will be the size of the full model (~0.5GB in this case) and you should only merge when you’re done fine-tuning.
8. Advanced Ideas and Next Steps
Fine-tuning Gemma 3 on ChessInstruct is just the beginning. Here are a few ideas to further adapt and use the model in the chess domain:
•	Expanding to UCI Commands or Other Data Formats: The current dataset is instruction-based. You could augment it with data that teaches the model chess engine commands (UCI protocol), PGN notations, or even board state representations. For example, create prompts like “Given this FEN, what is the best move?” or incorporate sequences of moves from NPZ/CSV game databases. You would convert such data into text in a similar role-based format (perhaps use system messages to give the model special instructions, and user messages with the raw data). This can help the model understand structured chess input beyond the provided dataset.
•	Integration with Chess Engines or Libraries: Consider using the fine-tuned model alongside a chess library (like python-chess) or an engine. The model could serve as a natural language commentator or an advisor. For instance, you could feed it a game and ask it to annotate the moves, or explain why a certain move is good or bad. Another idea is to have the model interpret and respond to UCI-like queries – e.g., the engine might say “bestmove e6f7” and the model could explain that in plain language. Fine-tuning on dialogues between an engine and a human (or synthetic data in that style) could make Gemma 3 a chess assistant capable of both calculating moves and chatting about them.
•	Quantization for Inference: To deploy the model more efficiently on-device, you can leverage quantization. Unsloth supports loading in 4-bit mode (as we saw with load_in_4bit=True). You could fine-tune in 4-bit as well (QLoRA approach) if memory is a constraint – Unsloth is compatible with QLoRA and bitsandbytes 4-bit optimizations[20]. After training, for serving the model, you might consider converting it to an even more optimized format like GGML/GGUF (for use with llama.cpp on mobile devices) – Unsloth provides tools to save adapters to GGUF as well. Since Gemma 270M is already small, quantization to 4-bit or 8-bit will make it extremely lightweight (potentially <0.5 GB memory), allowing it to run on mobile or web. Keep in mind quantization might slightly reduce precision, but often the impact on a well-finetuned model’s quality is minor.
•	Larger Models and Multimodal Chess: Gemma 3 has larger variants (1B, 4B, etc.) and even multimodal (vision + text) versions. With Apple’s unified memory architecture, you might try fine-tuning a 1B or 4B model if you have an M3 Max/Ultra with sufficient RAM. A vision-capable model could potentially take an image of a chessboard and output moves or evaluations – an intriguing direction if you have data mapping images to moves. While this goes beyond ChessInstruct, Unsloth’s methods would be similar (just with a vision encoder in the loop).
By following this guide, you’ve set up a pipeline to train a chess-specialized LLM on Apple Silicon. With LoRA and Unsloth, even a laptop-grade machine can fine-tune models on niche tasks. Happy refining, and happy chess coding!
Sources: Gemma-3 model and Unsloth fine-tuning tools[2][10]; ChessInstruct dataset and task design[15]; Unsloth documentation on LoRA hyperparameters[21][22].
 
[1] [10] [11] [20] From Pretrained to Purposeful: Fine-Tuning LLaMA 3.2 Made Easy with Unsloth | by Vishnu Sivan | Jun, 2025 | Medium
https://codemaker2016.medium.com/from-pretrained-to-purposeful-fine-tuning-llama-3-2-made-easy-with-unsloth-54f2d2530e8c
[2] Google’s New LLM Runs on Just 0.5 GB RAM — Here’s How to Fine-Tune It Locally | by Civil Learning | Coding Nexus | Aug, 2025 | Medium
https://medium.com/coding-nexus/googles-new-llm-runs-on-just-0-5-gb-ram-here-s-how-to-fine-tune-it-locally-ab910fa39732
[3] [4] [5] [6] [7] [8] [9] [12] [13] [18] [19] [21] [22] Home · unslothai/unsloth Wiki · GitHub
https://github.com/unslothai/unsloth/wiki
[14] [15] Thytu/ChessInstruct · Datasets at Hugging Face
https://huggingface.co/datasets/Thytu/ChessInstruct
[16] [17] dataset_utils.py
https://github.com/unslothai/unsloth-zoo/blob/add0bf32dd47c54d5bcb1044cba778b84f3773d2/unsloth_zoo/dataset_utils.py

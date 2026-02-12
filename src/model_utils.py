"""
Model loading with QLoRA. Uses float16 for T4 GPU compatibility.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


def load_tokenizer(model_name):
    """Load tokenizer with left-padding (needed for batch generation)."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_model_with_lora(
    model_name,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=None,
    quantization="4bit",
):
    """Load a quantized model and attach LoRA adapters to it."""
    if target_modules is None:
        # these cover all the linear layers in Qwen-style transformers
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    bnb_config = None
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if quantization in ("4bit", "8bit"):
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def create_reference_model(model):
    """
    Snapshot the LoRA weights so we can compute KL penalty later.
    We don't load a second model -- that wouldn't fit in GPU memory.
    """
    ref_params = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            ref_params[name] = param.data.clone().detach()
    print(f"Reference policy: {len(ref_params)} LoRA tensors frozen")
    return ref_params

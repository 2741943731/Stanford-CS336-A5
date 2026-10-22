import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_flash_attn_2_available


def pick_attn_implementation(device: str) -> str:
    """flash_attention_2 when it is really installed, otherwise SDPA.

    flash-attn is an optional compiled extension: `uv sync` only installs it on
    Linux/x86-64 with the matching torch build. Asking for it unconditionally
    makes from_pretrained raise ImportError before a single weight is loaded.
    """
    if str(device).startswith("cpu"):
        return "eager"
    if is_flash_attn_2_available():
        return "flash_attention_2"
    return "sdpa"


def get_model_and_tokenizer(
    model_id_or_dir: str,
    device: str,
    attn_implementation: str | None = None,
):
    implementation = attn_implementation or pick_attn_implementation(device)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_dir,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation=implementation,
        )
    except (ImportError, ValueError) as error:
        # flash-attn can be installed but unusable (torch / CUDA version mismatch).
        if implementation == "sdpa":
            raise
        print(
            f"[checkpoint] {implementation} unavailable ({error}); falling back to sdpa"
        )
        implementation = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_dir,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation=implementation,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    print(
        f"[checkpoint] loaded {model_id_or_dir} on {device} "
        f"with attn_implementation={implementation}"
    )
    return model, tokenizer

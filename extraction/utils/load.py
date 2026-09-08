import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_NAME = "Qwen/Qwen3.5-9B"

#adapted for mac
def load_qwen():
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Loading model on:", device)

    dtype = (
        torch.float16
        if device in ["mps", "cuda"]
        else torch.float32
    )

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        local_files_only=True,
    )

    model = model.to(device)
    model.eval()

    return model, processor
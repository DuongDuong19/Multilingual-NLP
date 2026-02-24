"""
generate.py  –  Load checkpoint mới nhất và chạy inference.
"""

import os
import torch
from model import MTT

CKPT_DIR = "checkpoints"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    print("[INFO] Khởi tạo model…")
    model = MTT().to(DEVICE)
    model.eval()

    path = None
    if os.path.isdir(CKPT_DIR):
        pts = sorted(
            [f for f in os.listdir(CKPT_DIR) if f.endswith(".pt")],
            key=lambda f: int(f.split("step")[1].split(".")[0]) if "step" in f else -1,
        )
        if pts:
            path = os.path.join(CKPT_DIR, pts[-1])

    if path:
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"[INFO] Loaded: {path}  (epoch={ckpt.get('epoch','?')}, step={ckpt.get('step','?')})")
    else:
        print("[WARN] Không tìm thấy checkpoint – dùng weights random.")

    return model


def run(
    texts: list[str],
    max_new_tokens: int       = 64,
    repetition_penalty: float = 1.5,
):
    model = load_model()
    print("\n" + "═" * 60)
    with torch.no_grad():
        outputs = model.generate(
            texts,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    for src, out in zip(texts, outputs):
        print(f"  SRC : {src}")
        print(f"  OUT : {out if out.strip() else '(empty – model cần train thêm)'}")
        print()
    print("═" * 60)


if __name__ == "__main__":
    test_inputs = [
        "The president of France visited Vietnam last week.",
        "Apple Inc. reported record revenue of $120 billion in Q4.",
        "Three people were injured in a car accident near Berlin.",
        "The new policy affects 50 percent of the population.",
    ]
    run(test_inputs)
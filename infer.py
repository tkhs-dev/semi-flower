# infer.py
# Load sample.pt and run inference on input.png (28x28 grayscale)

import argparse
import os

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Import the model definition from the project
from pytorchexample.task import Net


id_mapping = {
 0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j',
    46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't',
    56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z'
}

def centerlize_image(img: Image.Image, canvas_size: int = 28) -> Image.Image:
    """
    手書き文字画像を中央配置して返す
    img : リサイズ後のPIL Image（L）
    canvas_size : モデル入力サイズ
    """
    img = img.convert("L")
    img_np = np.array(img)

    # 文字領域を検出（白背景想定）
    mask = img_np < 250
    coords = np.column_stack(np.where(mask))

    if coords.size == 0:
        # 文字が無い場合はそのまま中央に
        return Image.new("L", (canvas_size, canvas_size), 255)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    char_img = img_np[y_min:y_max+1, x_min:x_max+1]

    h, w = char_img.shape

    # キャンバス作成
    canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255

    y_offset = (canvas_size - h) // 2
    x_offset = (canvas_size - w) // 2

    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = char_img

    return Image.fromarray(canvas)


def main():
    parser = argparse.ArgumentParser(description="Run inference with given model on input image")
    parser.add_argument("--model", type=str, default="final-model.pt", help="Path to model state dict file")
    parser.add_argument("--image", type=str, default="input.png", help="Path to input image (28x28) to classify")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to show")
    args = parser.parse_args()

    model_path = args.model
    image_path = args.image

    if not os.path.isfile(model_path):
        print(f"Model file not found: {model_path}")
        return
    if not os.path.isfile(image_path):
        print(f"Image file not found: {image_path}")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize model and load state dict
    model = Net()
    state_dict = torch.load(model_path, map_location=device)
    # Try to extract a plain state_dict from common wrapper formats
    def _extract_state_dict(obj):
        # If it's already a dict of tensors, return it
        if isinstance(obj, dict):
            # Heuristic: check if values look like tensors
            if all(hasattr(v, "dim") for v in obj.values()):
                return obj
            for key in ("model_state_dict", "state_dict", "model"):  # common wrappers
                if key in obj and isinstance(obj[key], dict):
                    return obj[key]
            # Look for nested dict values that are state_dict-like
            for v in obj.values():
                if isinstance(v, dict) and all(hasattr(x, "dim") for x in v.values()):
                    return v
        return None

    extracted = _extract_state_dict(state_dict)
    if extracted is None:
        raise RuntimeError("Could not interpret the loaded model file as a PyTorch state_dict")

    try:
        model.load_state_dict(extracted)
    except Exception:
        # Retry with non-strict in case of key mismatch
        model.load_state_dict(extracted, strict=False)

    model.to(device)
    model.eval()

    # Load and preprocess image
    img = Image.open(image_path).convert("L")  # grayscale
    img = img.resize((28, 28))
    img = centerlize_image(img, 28)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    tensor = transform(img).unsqueeze(0).to(device)  # shape: [1, 1, 28, 28]

    topk = max(1, args.topk)
    with torch.no_grad():
        outputs = model(tensor)  # [1, C]
        probs = torch.softmax(outputs, dim=1).squeeze(0)  # [C]
        num_classes = probs.shape[0]
        k = min(topk, num_classes)
        values, indices = torch.topk(probs, k=k)

    print(f"Top {k} predictions (class_id, confidence):")
    for rank, (idx, val) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        print(f"{rank}. {id_mapping[idx]}({idx}) (confidence: {val:.4f})")


if __name__ == "__main__":
    main()


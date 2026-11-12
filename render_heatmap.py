import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps
from spatial_aux import SpatialHead


def main():
    parser = argparse.ArgumentParser(description="Render a wrist-token heatmap on a matching 224px wrist image")
    parser.add_argument("prefix", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path, default=Path(__file__).parent / "weights/spatial_head.pt")
    args = parser.parse_args()
    prefix = torch.load(args.prefix, map_location="cpu", weights_only=True)
    if isinstance(prefix, torch.Tensor) and prefix.ndim == 2:
        prefix = prefix.unsqueeze(0)
    if not isinstance(prefix, torch.Tensor) or prefix.ndim != 3 or prefix.shape[0] != 1 or prefix.shape[1] < 768 or prefix.shape[2] != 2048:
        parser.error("Expected a single prefix tensor shaped [T, 2048] or [1, T, 2048], T >= 768")
    with Image.open(args.image) as source:
        wrist = source.convert("RGB")
    if wrist.size != (224, 224):
        parser.error("Expected the matching padded 224 x 224 wrist image")
    head = SpatialHead().eval()
    head.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    with torch.inference_mode():
        prob = head(prefix).softmax(-1).reshape(16, 16)
    if not torch.isfinite(prob).all():
        parser.error("Non-finite heatmap")
    pixels = (prob / prob.max() * 255).round().byte().contiguous()
    mask = Image.frombytes("L", (16, 16), bytes(pixels.flatten().tolist())).resize(wrist.size, Image.Resampling.BILINEAR)
    colors = ImageOps.colorize(mask, "darkred", "yellow")
    Image.composite(colors, wrist, mask.point(lambda x: round(x * 0.7))).save(args.output, format="PNG")


if __name__ == "__main__":
    main()

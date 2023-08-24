import argparse
from pathlib import Path

from diffusers.utils import load_image
import torch

from custom_pipeline import CustomPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script with two path arguments.")
    parser.add_argument("path_img1", help="content img")
    parser.add_argument("path_img2", help="style img")
    parser.add_argument("path_out", help="output dir")
    args = parser.parse_args()
    if not Path(args.path_img1).is_file():
        print(f"{args.path_img1} does not exist")

    if not Path(args.path_img2).is_file():
        print(f"{args.path_img2} does not exist")

    outdir = Path(args.path_out)
    outdir.mkdir(parents=True, exist_ok=True)

    pipe = CustomPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-unclip-small",
        torch_dtype=torch.float16,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)

    image_init1 = load_image(args.path_img1)
    image_init2 = load_image(args.path_img2)

    pipe_result = pipe(image_init2, image_latents=image_init1)
    for idx, i in enumerate(pipe_result.images):
        i.save(str(outdir / f"mix_result_{idx}.png"))

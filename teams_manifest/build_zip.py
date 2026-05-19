"""
Build the Teams app package (lab-chatbot.zip) for deployment.

Steps:
  1. Generate color.png  — 192x192, blue background #0078D4, white "LAB" centred
  2. Generate outline.png — 32x32, white background, blue 20x20 square centred
  3. Zip manifest.json + color.png + outline.png -> teams_manifest/lab-chatbot.zip
"""

import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent


def make_color_icon(path: Path) -> None:
    img = Image.new("RGB", (192, 192), color=(0, 120, 212))  # #0078D4
    draw = ImageDraw.Draw(img)

    text = "LAB"
    try:
        font = ImageFont.truetype("arial.ttf", size=60)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (192 - text_w) // 2 - bbox[0]
    y = (192 - text_h) // 2 - bbox[1]
    draw.text((x, y), text, fill=(255, 255, 255), font=font)

    img.save(path, "PNG")


def make_outline_icon(path: Path) -> None:
    img = Image.new("RGB", (32, 32), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # 20x20 blue square centred in 32x32
    offset = (32 - 20) // 2
    draw.rectangle(
        [offset, offset, offset + 19, offset + 19],
        fill=(0, 120, 212),
    )

    img.save(path, "PNG")


def build_zip() -> None:
    color_path = HERE / "color.png"
    outline_path = HERE / "outline.png"
    manifest_path = HERE / "manifest.json"
    zip_path = HERE / "lab-chatbot.zip"

    make_color_icon(color_path)
    make_outline_icon(outline_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(manifest_path, "manifest.json")
        zf.write(color_path, "color.png")
        zf.write(outline_path, "outline.png")

    print(f"Zip généré : {zip_path.relative_to(HERE.parent)}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    build_zip()

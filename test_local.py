"""
Local test script for Qwen-Image-Edit handler.
Run this before deploying to verify everything works.
"""

import base64
import json
from pathlib import Path


def encode_image(image_path: str) -> str:
    """Encode an image file to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def decode_and_save(base64_str: str, output_path: str):
    """Decode base64 and save as image."""
    image_bytes = base64.b64decode(base64_str)
    with open(output_path, "wb") as f:
        f.write(image_bytes)
    print(f"Saved output to: {output_path}")


def test_handler():
    """Test the handler locally."""
    from handler import handler

    # Create a test input (you need to provide a test image)
    test_image_path = "test_input.png"

    if not Path(test_image_path).exists():
        print(f"Please create a test image at: {test_image_path}")
        print("Example: Save any PNG image as 'test_input.png'")
        return

    # Encode the test image
    image_base64 = encode_image(test_image_path)

    # Create test job
    test_job = {
        "input": {
            "image": image_base64,
            "prompt": "Change the background to a sunset scene",
            "num_inference_steps": 30,  # Lower for faster testing
            "true_cfg_scale": 4.0,
            "seed": 42,
        }
    }

    print("Running inference...")
    result = handler(test_job)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    # Save the output
    decode_and_save(result["image"], "test_output.png")
    print("Test completed successfully!")


if __name__ == "__main__":
    test_handler()

"""
RunPod Serverless Handler for Qwen-Image-Edit
"""

import runpod
import torch
import base64
import io
from PIL import Image
from diffusers import QwenImageEditPipeline

# Global pipeline instance
pipeline = None


def load_model():
    """Load the Qwen-Image-Edit model and all components."""
    global pipeline
    if pipeline is None:
        print("=" * 50)
        print("Downloading and loading Qwen-Image-Edit model...")
        print("This may take a while on first run (~58GB download)")
        print("=" * 50)

        # Download and load the full pipeline (includes all components:
        # - Text encoder
        # - VAE
        # - UNet/Transformer
        # - Scheduler
        # - Tokenizer
        pipeline = QwenImageEditPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        pipeline.to("cuda")

        # Warm up the pipeline with a dummy inference (optional)
        print("Model loaded and ready!")
        print("=" * 50)
    return pipeline


def decode_base64_image(image_data: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    if image_data.startswith("data:"):
        image_data = image_data.split(",", 1)[1]
    image_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Log image details for debugging
    print(f"Decoded image: size={img.size}, mode={img.mode}")

    return img


def encode_image_to_base64(image: Image.Image, format: str = "PNG", quality: int = 95) -> str:
    """Encode PIL Image to base64 string."""
    buffer = io.BytesIO()
    if format.upper() == "JPEG":
        image.save(buffer, format=format, quality=quality)
    else:
        image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def resize_image(image: Image.Image, width: int = None, height: int = None) -> Image.Image:
    """Resize image if dimensions specified."""
    if width and height:
        return image.resize((width, height), Image.Resampling.LANCZOS)
    elif width:
        ratio = width / image.width
        return image.resize((width, int(image.height * ratio)), Image.Resampling.LANCZOS)
    elif height:
        ratio = height / image.height
        return image.resize((int(image.width * ratio), height), Image.Resampling.LANCZOS)
    return image


def handler(job):
    """
    RunPod serverless handler for image editing using QwenImageEditPipeline.

    API Reference: https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage

    All Available Parameters:
    {
        "input": {
            # Required
            "image": "<base64 encoded image>",          # Single image input
            "images": ["<base64_1>", "<base64_2>"],     # OR multiple images (1-3)
            "prompt": "Edit instruction",

            # Generation Parameters (QwenImageEditPipeline supported)
            "negative_prompt": " ",           # What to avoid (default: " ")
            "num_inference_steps": 50,        # Denoising steps (1-150, default: 50)
            "true_cfg_scale": 4.0,            # CFG guidance (1.0-20.0, default: 4.0)
            "seed": null,                     # Random seed for reproducibility
            "num_images": 1,                  # Number of output images (1-4)

            # Image Size
            "width": null,                    # Output width (default: same as input)
            "height": null,                   # Output height (default: same as input)
            "max_size": 1024,                 # Max dimension to prevent OOM (default: 1024)

            # Output Options
            "output_format": "PNG",           # PNG, JPEG, WEBP
            "output_quality": 95,             # JPEG/WEBP quality (1-100)
            "return_input": false             # Also return the input image(s)
        }
    }

    Note: 'strength' and 'guidance_scale' are NOT supported by QwenImageEditPipeline.

    Multi-Image Examples:
    - Combine person from image1 with background from image2
    - "The person from image 1 is standing in the scene from image 2"
    """
    job_input = job.get("input", {})

    # Validate required fields
    if "image" not in job_input and "images" not in job_input:
        return {"error": "Missing required field: 'image' or 'images'"}
    if "prompt" not in job_input:
        return {"error": "Missing required field: prompt"}

    try:
        # Load model
        pipe = load_model()

        # Decode input image(s) - support both single and multiple
        input_images = []
        if "images" in job_input:
            # Multiple images provided as list
            for img_data in job_input["images"][:3]:  # Cap at 3 images
                input_images.append(decode_base64_image(img_data))
        elif "image" in job_input:
            # Single image (can be string or list)
            if isinstance(job_input["image"], list):
                for img_data in job_input["image"][:3]:
                    input_images.append(decode_base64_image(img_data))
            else:
                input_images.append(decode_base64_image(job_input["image"]))

        if not input_images:
            return {"error": "No valid images provided"}

        # Use first image for size reference
        original_size = input_images[0].size

        # === Generation Parameters ===
        # See: https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage
        prompt = job_input["prompt"]
        negative_prompt = job_input.get("negative_prompt", " ")  # Space is default
        num_inference_steps = job_input.get("num_inference_steps", 50)
        true_cfg_scale = job_input.get("true_cfg_scale", 4.0)  # CFG guidance, >1 enables negative_prompt
        seed = job_input.get("seed", None)
        num_images = min(job_input.get("num_images", 1), 4)  # Cap at 4

        # === Image Size Parameters ===
        width = job_input.get("width", None)
        height = job_input.get("height", None)
        max_size = job_input.get("max_size", 1024)

        # === Output Parameters ===
        output_format = job_input.get("output_format", "PNG").upper()
        output_quality = job_input.get("output_quality", 95)
        return_input = job_input.get("return_input", False)

        # Validate parameters
        num_inference_steps = max(1, min(150, int(num_inference_steps)))
        true_cfg_scale = max(1.0, min(20.0, float(true_cfg_scale)))
        output_quality = max(1, min(100, int(output_quality)))

        if output_format not in ["PNG", "JPEG", "WEBP"]:
            output_format = "PNG"

        # Resize all input images if needed to prevent OOM
        # Also ensure dimensions are divisible by 14 (Qwen patch size)
        PATCH_SIZE = 14
        processed_images = []
        for img in input_images:
            w, h = img.size

            # First, scale down if too large
            if max_size and (w > max_size or h > max_size):
                ratio = max_size / max(w, h)
                w = int(w * ratio)
                h = int(h * ratio)

            # Ensure dimensions are divisible by patch size
            w = (w // PATCH_SIZE) * PATCH_SIZE
            h = (h // PATCH_SIZE) * PATCH_SIZE

            # Minimum size
            w = max(w, PATCH_SIZE * 4)  # At least 56 pixels
            h = max(h, PATCH_SIZE * 4)

            if (w, h) != img.size:
                print(f"Resizing image from {img.size} to ({w}, {h})")
                img = img.resize((w, h), Image.Resampling.LANCZOS)

            processed_images.append(img)

        # For single image, pass directly; for multiple, pass as list
        # Important: Qwen pipeline expects single PIL.Image for single edit
        if len(processed_images) == 1:
            input_for_pipeline = processed_images[0]
            print(f"Single image mode: size={input_for_pipeline.size}")
        else:
            input_for_pipeline = processed_images
            print(f"Multi-image mode: {len(processed_images)} images")

        # Set up generator for reproducibility
        generator = None
        if seed is not None:
            seed = int(seed)
            generator = torch.Generator(device="cuda").manual_seed(seed)

        # Build pipeline kwargs
        # See: https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage
        # Supported params: image, prompt, negative_prompt, true_cfg_scale,
        #                   num_inference_steps, num_images_per_prompt, generator
        pipe_kwargs = {
            "image": input_for_pipeline,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "true_cfg_scale": true_cfg_scale,
        }

        # Only add num_images_per_prompt if generating multiple outputs
        if num_images > 1:
            pipe_kwargs["num_images_per_prompt"] = num_images

        # Add generator for reproducibility
        if generator:
            pipe_kwargs["generator"] = generator

        print(f"Pipeline kwargs: image_count={1 if not isinstance(input_for_pipeline, list) else len(input_for_pipeline)}, "
              f"prompt='{prompt[:50]}...', steps={num_inference_steps}, true_cfg={true_cfg_scale}")

        # Run inference
        output = pipe(**pipe_kwargs)

        # Process output images
        output_images = []
        for img in output.images:
            # Resize output if requested
            if width or height:
                img = resize_image(img, width, height)
            output_images.append(encode_image_to_base64(img, format=output_format, quality=output_quality))

        # Build response
        response = {
            "images": output_images,
            "image": output_images[0],  # First image for convenience
            "count": len(output_images),
            "format": output_format,
            "seed": seed,
            "parameters": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "true_cfg_scale": true_cfg_scale,
                "num_images": num_images,
                "input_count": len(processed_images),
                "original_size": list(original_size),
                "processed_size": list(processed_images[0].size),
            }
        }

        # Optionally return input image(s)
        if return_input:
            if len(processed_images) == 1:
                response["input_image"] = encode_image_to_base64(
                    processed_images[0], format=output_format, quality=output_quality
                )
            else:
                response["input_images"] = [
                    encode_image_to_base64(img, format=output_format, quality=output_quality)
                    for img in processed_images
                ]

        return response

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# Model loads on first request (not at startup)
# This allows worker to initialize quickly and pass health checks
# First request will be slower while model downloads

# Start the serverless handler
runpod.serverless.start({"handler": handler})

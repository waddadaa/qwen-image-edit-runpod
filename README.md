# Qwen-Image-Edit RunPod Serverless

Deploy Qwen-Image-Edit as a serverless API on RunPod for AI-powered image editing.

## Features

- Natural language image editing
- **Multi-image input** (1-3 images for composition, style transfer, face swap)
- Multiple output formats (PNG, JPEG, WEBP)
- Batch generation (up to 4 output variants)
- Automatic image resizing
- Full parameter control

## Quick Start

### 1. Build & Push Docker Image

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/qwen-image-edit-runpod.git
cd qwen-image-edit-runpod

# Build the Docker image (this will take a while - downloads ~58GB model)
docker build -t your-dockerhub-username/qwen-image-edit:latest .

# Push to Docker Hub
docker push your-dockerhub-username/qwen-image-edit:latest
```

### 2. Deploy on RunPod

1. Go to [RunPod Serverless Console](https://runpod.io/console/serverless)
2. Click **New Endpoint**
3. Configure:
   - **Image**: `your-dockerhub-username/qwen-image-edit:latest`
   - **GPU**: A6000 or A100 (48GB+ VRAM required)
   - **Container Disk**: 100GB minimum
4. Click **Deploy**

### 3. Use the API

```python
import runpod
import base64

runpod.api_key = "YOUR_RUNPOD_API_KEY"
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

# Load image
with open("input.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Edit image
result = endpoint.run_sync({
    "image": image_b64,
    "prompt": "Change the background to a sunset beach"
})

# Save result
with open("output.png", "wb") as f:
    f.write(base64.b64decode(result["image"]))
```

## API Reference

### Endpoint

```
POST https://api.runpod.ai/v2/{endpoint_id}/runsync
```

### Request Body

```json
{
  "input": {
    "image": "<base64_encoded_image>",
    "prompt": "Your edit instruction"
  }
}
```

### Parameters

#### Required

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | string or array | Base64 encoded image(s) - single string or array of strings |
| `images` | array | Alternative: array of base64 encoded images (1-3 optimal) |
| `prompt` | string | Natural language edit instruction |

> **Note:** Use either `image` or `images`, not both. For multi-image tasks, use `images` array.

#### Generation Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `negative_prompt` | string | `" "` | - | What to avoid in output |
| `num_inference_steps` | int | `50` | 1-150 | Denoising steps (higher = better quality, slower) |
| `true_cfg_scale` | float | `4.0` | 1.0-20.0 | Prompt adherence strength |
| `guidance_scale` | float | `7.5` | 1.0-20.0 | Classifier-free guidance scale |
| `strength` | float | `0.8` | 0.0-1.0 | How much to transform the image |
| `seed` | int | random | - | Random seed for reproducibility |
| `num_images` | int | `1` | 1-4 | Number of image variants to generate |

#### Image Size Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `width` | int | input size | Output image width |
| `height` | int | input size | Output image height |
| `max_size` | int | `1024` | Max dimension (prevents OOM errors) |

#### Output Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_format` | string | `"PNG"` | Output format: PNG, JPEG, WEBP |
| `output_quality` | int | `95` | JPEG/WEBP quality (1-100) |
| `return_input` | bool | `false` | Include processed input in response |

### Response

```json
{
  "image": "<base64_encoded_image>",
  "images": ["<base64_image_1>", "<base64_image_2>"],
  "count": 2,
  "format": "PNG",
  "seed": 12345,
  "parameters": {
    "prompt": "Change the background to sunset",
    "negative_prompt": " ",
    "num_inference_steps": 50,
    "true_cfg_scale": 4.0,
    "guidance_scale": 7.5,
    "strength": 0.8,
    "num_images": 2,
    "original_size": [1920, 1080],
    "processed_size": [1024, 576]
  }
}
```

### Error Response

```json
{
  "error": "Error message",
  "traceback": "Full stack trace"
}
```

## Usage Examples

### Basic Edit

```python
result = endpoint.run_sync({
    "image": image_b64,
    "prompt": "Make the sky blue with fluffy clouds"
})
```

### High Quality Edit

```python
result = endpoint.run_sync({
    "image": image_b64,
    "prompt": "Transform to oil painting style, Ultra HD, 4K",
    "num_inference_steps": 100,
    "true_cfg_scale": 6.0
})
```

### Generate Multiple Variants

```python
result = endpoint.run_sync({
    "image": image_b64,
    "prompt": "Add dramatic lighting",
    "num_images": 4,
    "seed": 42
})

# Save all variants
for i, img_b64 in enumerate(result["images"]):
    with open(f"variant_{i}.png", "wb") as f:
        f.write(base64.b64decode(img_b64))
```

### Resize Output

```python
result = endpoint.run_sync({
    "image": image_b64,
    "prompt": "Remove the person from the scene",
    "width": 1920,
    "height": 1080,
    "output_format": "JPEG",
    "output_quality": 90
})
```

### Subtle Edit (Low Strength)

```python
result = endpoint.run_sync({
    "image": image_b64,
    "prompt": "Slightly enhance the colors",
    "strength": 0.3
})
```

### Multi-Image Composition

```python
# Load multiple images
with open("person.png", "rb") as f:
    person_b64 = base64.b64encode(f.read()).decode()
with open("background.png", "rb") as f:
    bg_b64 = base64.b64encode(f.read()).decode()

# Combine person with new background
result = endpoint.run_sync({
    "images": [person_b64, bg_b64],
    "prompt": "Place the person from image 1 into the beach scene from image 2"
})
```

### Multi-Image Style Transfer

```python
result = endpoint.run_sync({
    "images": [content_b64, style_b64],
    "prompt": "Apply the artistic style from image 2 to image 1"
})
```

### Multi-Image Face Swap / Identity

```python
result = endpoint.run_sync({
    "images": [face_b64, scene_b64],
    "prompt": "Put the person from image 1 into the scene in image 2, maintaining their identity"
})
```

### cURL Example

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "'"$(base64 -w0 input.png)"'",
      "prompt": "Change hair color to blue",
      "num_inference_steps": 50
    }
  }'
```

## Prompt Tips

### Single Image Prompts

| Edit Type | Example Prompt |
|-----------|----------------|
| Color change | `"Change the car color to metallic red"` |
| Background | `"Replace background with mountain landscape"` |
| Style transfer | `"Convert to watercolor painting style"` |
| Add elements | `"Add sunglasses and a hat to the person"` |
| Remove objects | `"Remove the tree from the left side"` |
| Lighting | `"Add golden hour sunset lighting"` |
| Weather | `"Make it a rainy day with wet streets"` |
| Text | `"Change the sign to read 'Welcome'"` |

### Multi-Image Prompts (1-3 images optimal)

| Use Case | Example Prompt |
|----------|----------------|
| Person + Background | `"Place the person from image 1 into the scene from image 2"` |
| Face swap | `"Put the face from image 1 onto the person in image 2"` |
| Style transfer | `"Apply the artistic style of image 2 to image 1"` |
| Object insertion | `"Add the product from image 1 onto the table in image 2"` |
| Outfit transfer | `"Dress the person in image 1 with clothes from image 2"` |
| Two people | `"The person from image 1 and person from image 2 standing together"` |
| Three-way blend | `"Combine the person from image 1, background from image 2, and lighting from image 3"` |

### Quality Boosters

Append these to your prompts for better results:

- English: `", Ultra HD, 4K, cinematic composition."`
- Detailed: `", highly detailed, professional photography"`

### Parameter Tuning

| Goal | Recommended Settings |
|------|---------------------|
| Fast preview | `num_inference_steps: 25` |
| High quality | `num_inference_steps: 75-100` |
| Strong edit | `strength: 0.9, true_cfg_scale: 6.0` |
| Subtle edit | `strength: 0.3-0.5` |
| Consistent results | Set `seed` to fixed value |

## Project Structure

```
qwen-image-edit-runpod/
├── Dockerfile          # RunPod-optimized container (includes model)
├── handler.py          # Serverless request handler
├── requirements.txt    # Python dependencies
├── LICENSE             # Apache 2.0 License
└── README.md           # This file
```

## Hardware Requirements

- **GPU**: 48GB+ VRAM (A6000, A100)
- **Container Disk**: 100GB minimum
- **Model Size**: ~58GB (baked into Docker image)

## Troubleshooting

### Out of Memory (OOM)

Reduce `max_size` parameter or use smaller input images:

```python
result = endpoint.run_sync({
    "image": image_b64,
    "prompt": "Your edit",
    "max_size": 768
})
```

### Blurry Results

Increase inference steps and CFG scale:

```python
result = endpoint.run_sync({
    "image": image_b64,
    "prompt": "Your edit",
    "num_inference_steps": 100,
    "true_cfg_scale": 6.0
})
```

## License

Apache 2.0 - Same as the [Qwen-Image-Edit model](https://huggingface.co/Qwen/Qwen-Image-Edit). See [LICENSE](LICENSE) for details.

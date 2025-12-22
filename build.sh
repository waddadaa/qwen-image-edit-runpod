#!/bin/bash
# Build script for Qwen-Image-Edit RunPod deployment

# Set your Docker Hub username or registry
REGISTRY="${DOCKER_REGISTRY:-your-dockerhub-username}"
IMAGE_NAME="qwen-image-edit"
TAG="${1:-latest}"

echo "Building Docker image: ${REGISTRY}/${IMAGE_NAME}:${TAG}"

# Build the image
docker build -t ${REGISTRY}/${IMAGE_NAME}:${TAG} .

# Optional: Push to registry
if [ "$2" == "--push" ]; then
    echo "Pushing to registry..."
    docker push ${REGISTRY}/${IMAGE_NAME}:${TAG}
    echo "Image pushed: ${REGISTRY}/${IMAGE_NAME}:${TAG}"
fi

echo "Done!"
echo ""
echo "To push the image, run:"
echo "  docker push ${REGISTRY}/${IMAGE_NAME}:${TAG}"
echo ""
echo "Or re-run this script with --push flag:"
echo "  ./build.sh ${TAG} --push"

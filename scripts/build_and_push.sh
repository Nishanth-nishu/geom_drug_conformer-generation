#!/bin/bash
# ============================================================
# build_and_push.sh — Build Docker image and push to nishanthr23/nextmol
#
# Run this from a machine that has Docker installed (e.g., your laptop
# or any node with docker daemon access).
#
# Usage:
#   bash scripts/build_and_push.sh           # build + push latest
#   bash scripts/build_and_push.sh v1.1      # build + push specific tag
#
# Requirements:
#   docker login nishanthr23 (done once)
#   docker buildx (for multi-platform, optional)
# ============================================================

set -euo pipefail

DOCKER_USER="nishanthr23"
IMAGE_NAME="nextmol"
TAG="${1:-latest}"
FULL_TAG="$DOCKER_USER/$IMAGE_NAME:$TAG"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================================"
echo "  Building: $FULL_TAG"
echo "  Context:  $PROJECT_ROOT"
echo "============================================================"

# Login check
if ! docker info &>/dev/null; then
    echo "ERROR: Docker daemon not running. Start Docker first."
    exit 1
fi

# Build
docker build \
    --platform linux/amd64 \
    --tag "$FULL_TAG" \
    --tag "$DOCKER_USER/$IMAGE_NAME:latest" \
    --file "$PROJECT_ROOT/Dockerfile" \
    "$PROJECT_ROOT"

echo ""
echo "Build complete. Pushing to Docker Hub..."
docker push "$FULL_TAG"

if [ "$TAG" != "latest" ]; then
    docker push "$DOCKER_USER/$IMAGE_NAME:latest"
fi

echo ""
echo "============================================================"
echo "  PUSHED: $FULL_TAG"
echo ""
echo "  Pull on any node with:"
echo "    docker pull $FULL_TAG"
echo ""
echo "  Run Exp F (2 GPUs):"
echo "    docker run --gpus all --rm \\"
echo "      -v /path/to/data:/workspace/data \\"
echo "      -v /path/to/experiments:/workspace/experiments \\"
echo "      $FULL_TAG \\"
echo "      torchrun --nproc_per_node=2 autoresearch/mol_train_ddp.py --exp F"
echo "============================================================"

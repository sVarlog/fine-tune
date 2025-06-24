FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DISABLE_TRANSFORMERS_SDPA=1

# Python + system tools
RUN apt update && apt install -y python3-pip git

# Optional: Create symlink
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install latest PyTorch
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install other tools
RUN pip install transformers datasets peft accelerate bitsandbytes trl einops

WORKDIR /workspace
COPY . .

CMD ["python", "scripts/train.py"]
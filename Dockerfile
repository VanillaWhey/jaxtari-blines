# jaxtari-blines (branch cluster-runs: ORCA/ORBiT, OCCAM opc) for an NVIDIA DGX, on NGC's PyTorch image.
#
# The environment is uv.lock's (JAX with its own CUDA 12 / cuDNN from pip, JAXtari from git), in /opt/venv,
# separate from the image's PyTorch.
#
# Build on the DGX from a fresh clone (LF line endings, so the logged BLINES_COMMIT isn't "-dirty"):
#   git clone -b cluster-runs https://github.com/VanillaWhey/jaxtari-blines.git && cd jaxtari-blines
#   docker build -t blines .
#
# JAXtari's sprites are not in the image: installing them means confirming ROM ownership, and they must not be
# redistributed. Install them once into a folder on the DGX:
#   mkdir -p ~/jaxtari-data ~/blines-runs
#   docker run --rm -it -v ~/jaxtari-data:/opt/data/jaxtari blines python -m jaxtari.install_sprites
#
# One run per GPU: JAX reserves memory on every GPU it sees, so give each container one (device=0..7):
#   docker run --rm --gpus '"device=0"' --ipc=host \
#       -v ~/jaxtari-data:/opt/data/jaxtari -v ~/blines-runs:/runs -e WANDB_API_KEY \
#       blines python main.py +alg=pqn_rgb_paper ENV_ID=pong OCCAM_MASK=opc \
#       ENTITY=<wandb entity> PROJECT=<wandb project> SAVE_PATH=/runs/models hydra.run.dir=/runs/hydra/pong_opc_1
# W&B runs go to /runs/wandb (WANDB_DIR); without WANDB_API_KEY add WANDB_MODE=offline (wandb sync later).
FROM nvcr.io/nvidia/pytorch:25.03-py3

# uv 0.13.7 wrote uv.lock (lock format revision 5)
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    # The image puts PyTorch's libraries first on LD_LIBRARY_PATH; JAX must load the CUDA/cuDNN from its own
    # pip wheels, so keep only the CUDA forward-compatibility and driver folders of the image's value.
    LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    # jaxtari looks for its sprites in $XDG_DATA_HOME/jaxtari (mounted, see above)
    XDG_DATA_HOME=/opt/data \
    JAXTARI_SKIP_SPRITE_UPDATE=1 \
    WANDB_DIR=/runs \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/jaxtari-blines

# dependencies first, so code changes don't reinstall them
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen \
    && python -c "import jax, flax, optax, hydra; print('jax', jax.__version__)" \
    && mkdir -p /opt/data/jaxtari /runs

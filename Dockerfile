# jaxtari-blines (branch cluster-runs: ORCA/ORBiT, OCCAM opc) for an NVIDIA GPU machine such as a DGX.
#
# JAX brings its own CUDA 12 / cuDNN / NCCL from pip (uv.lock), so the image only needs the host's NVIDIA
# driver, which `docker run --gpus` mounts in. The base is NVIDIA's minimal CUDA image, without its CUDA
# forward-compatibility driver (see below); a driver too old for JAX's CUDA libraries shows up as an error
# of JAX at start-up.
#
# Build from a fresh clone (LF line endings, so the logged BLINES_COMMIT isn't "-dirty"):
#   git clone -b cluster-runs https://github.com/VanillaWhey/jaxtari-blines.git && cd jaxtari-blines
#   docker build -t blines .
#
# JAXtari's sprites are not in the image: installing them means confirming ROM ownership, and they must not be
# redistributed. Install them once into a folder on the machine:
#   mkdir -p ~/jaxtari-data ~/blines-runs
#   docker run --rm -it -v ~/jaxtari-data:/opt/data/jaxtari blines python -m jaxtari.install_sprites
#
# One run per GPU: JAX reserves memory on every GPU it sees, so give each container one (device=0, 1, ...):
#   docker run --rm --gpus '"device=0"' --ipc=host \
#       -v ~/jaxtari-data:/opt/data/jaxtari -v ~/blines-runs:/runs -e WANDB_API_KEY \
#       blines python main.py +alg=pqn_rgb_paper ENV_ID=pong OCCAM_MASK=opc \
#       ENTITY=<wandb entity> PROJECT=<wandb project> SAVE_PATH=/runs/models hydra.run.dir=/runs/hydra/pong_opc_1
# W&B runs go to /runs/wandb (WANDB_DIR); without WANDB_API_KEY add WANDB_MODE=offline (wandb sync later).
FROM nvidia/cuda:12.9.2-base-ubuntu24.04

# git: uv fetches JAXtari from its git repository (uv.lock); runs also log the blines commit.
# cuda-compat: the base image's forward-compatibility libcuda (driver 575) comes before the host's driver
# library and makes cuInit fail on GeForce GPUs and on newer drivers (e.g. 580 on a DGX); JAX needs the
# host's driver library, so remove it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && apt-get purge -y cuda-compat-12-9 \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig

# the base image's driver check (NVIDIA_REQUIRE_CUDA: cuda>=12.9) is for its own CUDA runtime, which isn't
# used; JAX's CUDA libraries also run on older drivers (e.g. 572), so leave the check to JAX
ENV NVIDIA_DISABLE_REQUIRE=1

# uv 0.13.7 wrote uv.lock (lock format revision 5)
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /usr/local/bin/

# Python 3.12 (.python-version) is installed by uv into /opt/python; the environment goes into /opt/venv.
# Works with BuildKit and with the legacy builder (no RUN --mount); uv keeps no cache in the image.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    PATH=/opt/venv/bin:$PATH

# jaxtari looks for its sprites in $XDG_DATA_HOME/jaxtari (mounted, see above)
ENV XDG_DATA_HOME=/opt/data \
    JAXTARI_SKIP_SPRITE_UPDATE=1 \
    WANDB_DIR=/runs \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/jaxtari-blines

# dependencies first, so code changes don't reinstall them
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen \
    && python -c "import jax, flax, optax, hydra; print('jax', jax.__version__)" \
    && mkdir -p /opt/data/jaxtari /runs

CMD ["bash"]

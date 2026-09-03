# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04@sha256:813c3b834aaafe2df2891f9a840c412a4df3c073dd9e0ac75aab810ed82a5b48
ENV DEBIAN_FRONTEND=noninteractive CUBLAS_WORKSPACE_CONFIG=:4096:8

# The CUDA base carries no Python and no compiler. python3-dev and build-essential are
# needed at RUN time: Triton compiles cuda_utils.c on the first jit launch and that
# compile needs the CPython headers, and kernel.py is Triton.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv build-essential ca-certificates git \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN pip install --no-cache-dir "torch==2.13.0" "triton==3.7.1" "einops" "pytest==9.0.3"
# The graded anchor is a vendored flash-linear-attention 0.5.2 kernel, installed here.
RUN pip install --no-cache-dir "flash-linear-attention==0.5.2"

# task.toml runs the solver as `agent`; Harbor's installer chowns its venv to that
# account before it will run, so the user must exist.
RUN useradd -m -s /bin/bash agent && mkdir -p /workspace
RUN chown -R agent:agent /opt/venv

# The agent-visible surface only. tests/ and solution/ stay off-image.
COPY instruction.md kernel.py reference.py bench.py /workspace/
RUN chmod +x /workspace/bench.py && chown -R agent:agent /workspace
WORKDIR /workspace

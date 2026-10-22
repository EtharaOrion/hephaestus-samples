# Digest-pinned, not tag-pinned: a moved tag is a different image than the one
# graded. Same digest sibling bundle 96df9def pins for this identical tag.
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04@sha256:813c3b834aaafe2df2891f9a840c412a4df3c073dd9e0ac75aab810ed82a5b48
ENV DEBIAN_FRONTEND=noninteractive CUBLAS_WORKSPACE_CONFIG=:4096:8

# One shell token, no spaces. A space here splits into three argv entries in
# every unquoted RUN below and failed the build at the guard with
# "[: too many arguments".
ENV AGENT_VENV=/opt/agent-runtime-venv

# The base image carries neither a Python nor git. python3-dev and a compiler are
# needed at RUN time and not only at build time, because Triton compiles
# cuda_utils.c on the first jit launch and that compile needs the CPython headers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv build-essential git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# triton 3.7.1 is a hard floor, not a preference. flash-linear-attention refuses
# to run the gated delta rule on Hopper for triton >= 3.4.0 and < 3.7.1, raising
# "produces incorrect results", so on any earlier triton the graded denominator
# does not exist. Measured on this host: triton 3.4.0 raises, 3.7.1 runs.
RUN pip install --no-cache-dir "torch==2.13.0" "triton==3.7.1" "einops" "pytest==9.0.3"

# The reference library is present for the VERIFIER to time against. The agent is
# free to read it; calling it from the submission is rejected post hoc by G1.
RUN git clone https://github.com/fla-org/flash-linear-attention /opt/fla \
 && git -C /opt/fla checkout b4cfd53d646abab93541e82b8068c130de42ac3c \
 && pip install --no-cache-dir -e /opt/fla \
 && git -C /opt/fla rev-parse HEAD > /opt/fla/COMMIT \
 && rm -rf /opt/fla/.git /opt/fla/.github

RUN useradd -m -s /bin/bash agent && mkdir -p /workspace
# instruction.md tells the solver it "may install what you need". Without this the
# venv on PATH is root-owned while task.toml runs the agent as `agent`, so every
# pip install dies with EACCES. Measured 2026-08-12: an iteration lost its chosen
# remediation path to "Permission denied: /opt/venv/.../ninja".
RUN chown -R agent:agent /opt/venv

RUN apt-get update -qq && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p "$AGENT_VENV" && chown agent:agent "$AGENT_VENV"
USER agent
RUN set -eux; \
    curl -LsSf https://astral.sh/uv/install.sh | sh; \
    . "$HOME/.local/bin/env"; \
    uv python install 3.12; \
    uv venv "$AGENT_VENV" --python 3.12 --clear; \
    . "$AGENT_VENV/bin/activate"; \
    uv pip install openhands-sdk openhands-tools fastapi
# The guard harbor runs, asserted at build time so a broken install fails the
# BUILD rather than the eighth iteration of a fifty-iteration run.
RUN [ -f "$AGENT_VENV/bin/python" ] \
    && "$AGENT_VENV/bin/python" -c "import openhands.sdk"
USER root

# Paths resolve against the compose build context, which is the BUNDLE ROOT
# (docker-compose.yaml sets context: ..). instruction.md sits at that root;
# kernel.py, reference.py and bench.py sit under environment/. Copying all four
# as bare names failed the build with "/reference.py": not found.
COPY instruction.md environment/kernel.py environment/reference.py environment/bench.py /workspace/
RUN chmod +x /workspace/bench.py && chown -R agent:agent /workspace
WORKDIR /workspace

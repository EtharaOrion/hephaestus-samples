# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04@sha256:813c3b834aaafe2df2891f9a840c412a4df3c073dd9e0ac75aab810ed82a5b48
ENV DEBIAN_FRONTEND=noninteractive CUBLAS_WORKSPACE_CONFIG=:4096:8

# The base image carries neither a Python nor a compiler. python3-dev and
# build-essential are needed at RUN time and not only at build time, because
# Triton compiles cuda_utils.c on the first jit launch and that compile needs the
# CPython headers -- the candidate kernel.py is Triton, so this fires on the very
# first graded call. git is NOT installed: unlike the gated-delta bundles, this
# task clones no library into the image (see the reference note below).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv build-essential ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# torch 2.13.0 is a hard FLOOR, not a preference: the graded denominator is a
# torch._grouped_mm (CUTLASS grouped-GEMM) composition, and _grouped_mm does not
# exist on earlier torch. Measured on this host: torch 2.13.0 exposes
# torch._grouped_mm, so the reference MoE pipeline runs; drop below it and the
# denominator this whole bundle is timed against cannot be built, which makes the
# task ungradable rather than merely slower. triton 3.7.1 is the assembler for
# the candidate's own jit kernels (kernel.py is Triton) and is the version this
# host validated the oracle and starter on. pytest runs the recorded (non-gating)
# deterministic half. No einops, no flash-linear-attention: reference.py imports
# torch and nothing else, so neither is on any graded path.
RUN pip install --no-cache-dir "torch==2.13.0" "triton==3.7.1" "pytest==9.0.3"

# UNLIKE the gated-delta bundles, NO reference library is cloned. The anchor is a
# torch-native composition (torch._grouped_mm), not a vendored Triton kernel from
# flash-linear-attention, so there is nothing to install for the verifier to time
# against beyond torch itself. The forbidden-symbol scan (G1) still covers the
# grouped-GEMM / MoE-serving stacks (torch._grouped_mm, vllm, sglang,
# grouped_gemm, transformer_engine, flashinfer) by package name; none of them are
# present in the image, so a submission cannot import the production path even if
# it tried, and the candidate's own mandatory `def fused_moe` never
# false-positives against them.

# task.toml declares [agent] user = "agent", and Harbor's openhands-sdk installer
# chowns its venv to that account before it will run. The user has to exist or
# setup dies with "chown: invalid user: agent:agent" before the model is ever
# called, which reads as a hard task rather than a broken image.
RUN useradd -m -s /bin/bash agent && mkdir -p /workspace
# The venv on PATH is root-owned while task.toml runs the agent as `agent`. This
# task runs no-network at the environment stage (nothing to fetch), so the agent
# will not pip install at runtime -- but the chown is kept for parity with the
# fleet image and so any agent-side venv write (bytecode caches, editable
# metadata) does not die with EACCES.
RUN chown -R agent:agent /opt/venv

# THE AGENT-VISIBLE SURFACE. Harbor stages the environment; it does not copy
# bundle-root files into the agent workspace. Without this the solver opens an
# empty directory and burns tokens searching the filesystem for a reference.py
# that was never staged.
#
# tests/ and solution/ are deliberately NOT here. The private boundary is that
# these four files are the whole agent-visible surface.
COPY instruction.md kernel.py reference.py bench.py /workspace/
RUN chmod +x /workspace/bench.py && chown -R agent:agent /workspace
WORKDIR /workspace

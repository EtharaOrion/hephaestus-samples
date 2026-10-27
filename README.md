<p align="center"><img src="images/hero.svg" alt="Project" width="720"></p>

# Project: samples

<p align="center"><b>The samples lane: 3 of 5 Harbor task bundles.</b></p>

<p align="center"><a href="#summary"><img src="https://img.shields.io/badge/bundles-3_of_5-845EF7" alt="bundles: 3 of 5"></a> <a href="#evidence-status"><img src="https://img.shields.io/badge/sealed-0-845EF7" alt="sealed: 0"></a> <a href="#difficulty-tiers"><img src="https://img.shields.io/badge/difficulty-measured-845EF7" alt="difficulty: measured"></a></p>

<p align="center"><sub><a href="#abstract">Abstract</a> | <a href="#summary">Summary</a> | <a href="#difficulty-tiers">Difficulty tiers</a> | <a href="#evidence-status">Evidence status</a> | <a href="#coverage">Coverage</a> | <a href="#dataset-structure">Dataset structure</a> | <a href="#verification-and-quality-assurance">Verification and quality assurance</a> | <a href="#reproduction">Reproduction</a> | <a href="#cite">Cite</a> | <a href="#license">License</a> | <a href="#contact">Contact</a></sub></p>

## Abstract

The curated lane of this corpus: at most 5 Harbor task bundles, every anchor seated first and every other bundle in seal order. Bundles past that ceiling live in `delivery/`.

Each bundle is one Harbor task in a directory named by its content-derived uuid5, so the name doubles as an integrity check. This page is rendered from the resident bundles on every reconcile; a value a bundle does not carry reads `absent`, and the page never fills one in.

## Summary

| Property | Value |
| --- | --- |
| Bundles | 3 of 5 |
| Sealed | 0 of 3 |
| Registrations | unregistered (3) |
| Audit verdicts | none (3) |
| Harbor schema | 1.4 (3) |

## Difficulty tiers

Difficulty is measured by the signed pilot alone and never declared in a bundle, so this page states no tier a bundle claims.

## Evidence status

The verdict is the audit outcome recorded at the sealed digest, and `none` means no verdict has landed for that digest yet. Only an authoring run removes a bundle from this lane, so removal waits for the next one.

| uuid | Verdict | Sealed digest | Stream |
| --- | --- | --- | --- |
| 6d5ef72e-a408-59ca-83d5-907e56675db1 | none | absent | none |
| 96df9def-c9a9-58c0-a88c-f1ed47f97fed | none | absent | none |
| bc81a86b-7118-5bce-a3c1-fcb6f2459870 | none | absent | none |

## Coverage

| Keyword | Bundles |
| --- | --- |
| bounded-continuous | 2 |
| gpu-kernel | 2 |
| linear-attention | 2 |
| optimization | 2 |
| triton | 2 |

## Dataset structure

| Bundle | Task | Keywords | Registration | Sealed at | Tier |
| --- | --- | --- | --- | --- | --- |
| [`6d5ef72e-a408-59ca-83d5-907e56675db1`](6d5ef72e-a408-59ca-83d5-907e56675db1/) | `hephaestus/moe_routed_fwdbwd` | absent | unregistered | unsealed | - |
| [`96df9def-c9a9-58c0-a88c-f1ed47f97fed`](96df9def-c9a9-58c0-a88c-f1ed47f97fed/) | `hephaestus/chunked_dplr_delta` | bounded-continuous, gpu-kernel, linear-attention, optimization, triton | unregistered | unsealed | - |
| [`bc81a86b-7118-5bce-a3c1-fcb6f2459870`](bc81a86b-7118-5bce-a3c1-fcb6f2459870/) | `hephaestus/kda_chunk_gdr_fwdbwd` | bounded-continuous, gpu-kernel, linear-attention, optimization, triton | unregistered | unsealed | - |

```mermaid
%%{init: {"theme":"base","themeVariables":{"primaryColor":"#131a33","primaryTextColor":"#e8ecff","primaryBorderColor":"#845EF7","lineColor":"#22d3ee","fontFamily":"monospace"}}}%%
flowchart LR
    bundle["&lt;uuid&gt;/"]
    bundle --> n0["task.toml<br/>runner manifest"]
    bundle --> n1["instruction.md<br/>objective the agent reads"]
    bundle --> n2["environment/<br/>agent sandbox image"]
    bundle --> n3["tests/<br/>verifier, hidden from the agent"]
    bundle --> n4["solution/<br/>private oracle"]
    bundle --> n5["README.md"]
    bundle --> n6["inspector.html"]
    bundle --> n7["plots/"]
    bundle --> n8["trajectories/"]
```

```mermaid
%%{init: {"theme":"base","themeVariables":{"primaryColor":"#131a33","primaryTextColor":"#e8ecff","primaryBorderColor":"#845EF7","lineColor":"#22d3ee","fontFamily":"monospace"}}}%%
flowchart LR
    staging["staging/"]
    staging --> samples["samples/<br/>3 of 5"]
    samples --> delivery["delivery/<br/>0 of 10000"]
    delivery --> stock["stock/<br/>0 of 10000"]
```

## Verification and quality assurance

Each bundle's signed provenance carrier holds its screening roots, screening instant, and absolute expiry. Placement moves that carrier into the private derivation set, which this page never reads, so reconcile lists no root it cannot verify here.

## Reproduction

Each bundle directory is a complete Harbor task: `task.toml` is the runner manifest, `instruction.md` the objective, `environment/` the sandbox image, and `tests/` the verifier. Point a Harbor runner at one bundle directory to reproduce it; `solution/` stays private.

## Cite

```bibtex
@misc{hephaestus2026,
  title        = {hephaestus: An Adversarial RL Environment for Autonomous GPU Kernel Optimization via an Iterative Agent Loop},
  author       = {Ethara.AI Research},
  year         = {2026},
  note         = {Measured, never claimed. Forge the kernel, test the metal, remember what it cost.}
}
```

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 Ethara.AI. Vendored upstream repositories and any task bundles retain their own original licences.

## Contact

Security and benchmark-integrity disclosure: security@ethara.ai (see [SECURITY.md](SECURITY.md)). Conduct: conduct@ethara.ai (see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)).

---

## Copyright

Copyright (c) 2026 Green Rider Technology LLP (Ethara AI). All rights reserved.

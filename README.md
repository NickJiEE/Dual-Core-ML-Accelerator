# Dual-Core ML Accelerator for Attention Mechanism
**ECE 260B — VLSI Integrated Circuits and Systems Design, UCSD (Winter 2026)**

> ⚠️ **NDA Notice**: This project was developed under UCSD's ECE 260B course using the TSMC 65nm process node. Source files, RTL, scripts, and process-specific data are under NDA and cannot be shared publicly. This repository serves as design documentation only.

---

## Overview

A dual-core hardware accelerator for the scaled dot-product attention mechanism used in transformer-based machine learning models. The design computes **Q×K** across 8-element, 8-bit integer vectors using a **systolic array architecture**, synthesized and place-and-routed in **TSMC 65nm** targeting **1 GHz**.

**Team:** Nick Ji, Iris Chou, Karan Humpal, Nachiket Laul, Peter Chang, Zhao Cai  
**My Contributions:** Step 1 (sole contributor) · Step 3 & 4 (GLS debugging and modifications)

---

## Design Progression

The project was structured as a 6-step progressive build:

| Step | Description |
|------|-------------|
| 1 | Single-core RTL, Synthesis & PnR |
| 2 | Output Normalization |
| 3 | Hierarchical Synthesis of Single Core (with SRAM hard blocks) |
| 4 | Hierarchical Synthesis of Dual Core (with CDC bridge) |
| 5 | Optimization (pipelining, async handshake) |
| 6 | Sparsity-Aware Attention (row-level sparsity) |

---

## Architecture

### Compute Model
Each core computes a row of the Q×K dot product:
```
[Q0K0, Q0K1, ..., Q0K7]  @ cycle 0
[Q1K0, Q1K1, ..., Q1K7]  @ cycle 1
...
```
Results are stored in a partial sum memory (PMEM) and then normalized row-by-row.

### Single Core Architecture

> 📷 **Figure placeholder**: `figures/core_architecture.png`  
> *(Slide 6 — the mac_array / mac_col / kmem / ofifo / pmem block diagram)*

The core is organized as an 8×8 2D systolic array. K vectors are preloaded into each `mac_col`'s local registers. Q vectors then stream west-to-east through the array each cycle, and each MAC cell accumulates a partial dot product. The SFU at the bottom of each column drains completed partial sums into `ofifo`, which then writes to `pmem`.

### Systolic Dataflow

> 📷 **Figure placeholder**: `figures/systolic_computation_wave.png`  
> *(Slides 7–10 — the wave propagation showing K1\*Q1 → K1\*Q2/K2\*Q1 → ... → K8\*Q8)*

Each cycle, a new Q vector enters from the left while the previous Q vectors continue propagating rightward. This creates a diagonal wave of active multiply-accumulate operations across the array, achieving near-100% utilization once the pipeline is full.

### Dual Core Architecture

> 📷 **Figure placeholder**: `figures/dual_core_cdc.png`  
> *(Slide 15 — the two-core diagram showing CLK_core0 / CLK_core1 and sum_q[1–8] / sum_q[9–16] exchange)*

```
         ┌────────────┐       ┌────────────┐
 Q ─────►│   core0    │       │   core1    │◄───── Q
K[0:7]──►│  (clk0)    │       │  (clk1)    │◄──K[8:15]
         └─────┬──────┘       └──────┬─────┘
               │  sum_q[1–8]         │
               │─────────────────────►
               │        sum_q[9–16]  │
               ◄─────────────────────│
               │   async_sum_bridge  │
               └──────────┬──────────┘
                          │  4-phase req/ack handshake
                          ▼
                   merged normalization
```

The two cores operate on **independent clock domains**. A CDC bridge (`async_sum_bridge.v`) synchronizes only control signals (req/ack) through two flip-flops, avoiding metastability on the full data bus.

---

## Step 1 — Single Core RTL, Synthesis & PnR

### Key Design Decisions
- **Floorplan:** Square (AR=1), 65% utilization, 10 µm margins — balances wirelength for the regular systolic array while preserving routing headroom for wide buses (`mem_in[127:0]`)
- **Routing:** Timing- and SI-driven routing simultaneously, capped at M6. SI awareness mitigates crosstalk on wide buses; M6 cap reflects intent to integrate as a hard block later
- **Post-route:** DRV → setup → hold fix sequence; DCAP and filler cells inserted in two separate passes for explicit control over decap density

### Results
| Metric | Value |
|--------|-------|
| Technology | TSMC 65nm |
| Target Frequency | 1 GHz |
| WNS | −0.12 ns |
| Total Power (PnR) | 715.28 mW |
| Leakage Power (PnR) | 13.35 mW |
| Total Power (VCD/GLS) | 23.1 mW |
| Area | 520,417 µm² |

> **Note:** The large gap between PnR power and VCD power is expected — PnR reports worst-case switching activity across the full design, while VCD power is annotated from actual simulation activity during stable operation only (excluding the initial setup stage).

---

## Step 2 — Output Normalization

> 📷 **Figure placeholder**: `figures/normalization_flow.png`  
> *(Slide 14 — the pmem fetch → sfp → writeback loop diagram)*

### Algorithm
After MAC operations complete, the OFIFO drains partial sums into PMEM. A normalizer then performs a sequential read-modify-write over each PMEM row:

```
for each row i in PMEM:
    row_sum = Σ |PMEM[i][j]|  for j in 0..7
    PMEM[i][j] = |PMEM[i][j]| / row_sum   (fixed-point, frac_bits=12)
```

A `norm_en` signal gates the writeback, keeping the existing OFIFO datapath intact.

---

## Step 3 — Hierarchical Synthesis of Single Core

### SRAM Hard Blocks
Two SRAM variants were synthesized independently before core integration:
- `sram_w16_64` — Q/K memories
- `sram_w16_160` — PMEM

During synthesis, SRAMs are treated as black boxes (DC sees only interfaces). GLS used behavioral SRAM models with SDF back-annotation at worst-case corner.

`sfp_row` replaces the combinational normalizer from Step 2 with a **pipelined acc/div flow**, and the instruction word expands to 20 bits to expose `acc`, `div`, and `sfp_wr` controls.

### Results
| Metric | Value |
|--------|-------|
| WNS | −9.428 ns |
| Total Power (PnR) | 269.90 mW |
| Total Power (VCD/GLS) | 55.45 mW |
| Area | 1,306,400 µm² |

> The large WNS and area reflect the pre-optimization baseline and the overhead of synthesized SRAMs as STD cells — addressed in Step 5.

---

## Step 4 — Hierarchical Synthesis of Dual Core

### Clock Domain Crossing
The `async_sum_bridge.v` implements a **4-phase req/ack handshake** to exchange each core's partial sums across clock domains. Only control signals cross domain boundaries through synchronizer flip-flops — the full data bus is never synchronized directly.

### Results
| Metric | Value |
|--------|-------|
| WNS | −6.776 ns |
| Total Power (PnR) | 39.60 mW |
| Total Power (VCD/GLS) | 20.32 mW |
| Area | 3,911,600 µm² (~3× single core) |

---

## Step 5 — Optimization

### Techniques Applied
- **Pipeline splitting:** `sfp_row.v` divides the sum-then-divide critical path into two stages (`acc`, `div`), directly targeting the WNS bottleneck
- **Async handshake hardening:** CDC bridge synchronizes only req/ack through two flip-flops rather than the full data bus, eliminating the metastability risk on wide data paths
- **Quad-core considered** but deemed unfeasible within time constraints

### Results
| Metric | Before (Step 4) | After (Step 5) | Improvement |
|--------|----------------|----------------|-------------|
| WNS | −6.776 ns | −0.543 ns | +6.2 ns |
| Total Power (PnR) | 39.60 mW | 22.44 mW | ~43% reduction |
| Total Power (VCD) | 20.32 mW | 10.38 mW | ~49% reduction |
| Area | 3,911,600 µm² | 1,293,600 µm² | ~67% reduction |

---

## Step 6 — Sparsity-Aware Attention (Row-Level)

### B2: Row-Level Sparsity
`sfp_row.v` was modified to compute `Si = Σ |Qi×Kj|` after each row accumulates. If `Si < threshold`, the normalization stage is skipped/gated for that row, reducing switching activity.

- **Threshold:** 250 (absolute sum), targeting ~30% sparsity
- `sum_comb` computes each row's absolute sum combinationally
- `skip_q` gates the normalization writeback per row

### Verification
```
Row 0: skip_hw=1  skip_exp=1  →  SKIPPED (below threshold)
Row 1: skip_hw=0  skip_exp=0  →  HW: 21 17 13 8 4 30 26 6  |  Expected: 21 17 13 8 4 30 26 6  ✓
Row 2: skip_hw=0  skip_exp=0  →  HW: 16 16 16 16 16 16 16 16  |  Expected: match  ✓
...
RESULT: ALL PASSED  (48/48 checks, 0 mismatches)
```

### Power Impact
| Configuration | Total Power |
|---------------|-------------|
| Single core (no sparsity) | 83.2 mW |
| Single core (B2 sparsity) | 64.9 mW |
| Dual core (B2 sparsity) | 82.2 mW |

---

## Full Performance Summary

### Post-PnR Reports
| Step | Description | Leakage (mW) | Total Power (mW) | Area (µm²) | WNS (ns) |
|------|-------------|-------------|-----------------|------------|---------|
| 1 | Single core | 13.35 | 715.28 | 520,417 | −0.12 |
| 3 | Hierarchical single core | 4.90 | 269.90 | 1,306,400 | −9.428 |
| 4 | Dual core | 7.48 | 39.60 | 3,911,600 | −6.776 |
| 5 | Optimized dual core | 2.03 | 22.44 | 1,293,600 | −0.543 |
| 6 | Sparsity-aware dual core | 2.10 | 82.20 | 227,719 | −0.856 |

### Gate-Level Simulation VCD Power
| Step | Leakage (mW) | Total Power (mW) |
|------|-------------|-----------------|
| 1 | 13.56 | 23.1 |
| 3 | 4.96 | 55.45 |
| 4 | 7.47 | 20.32 |
| 5 | 1.55 | 10.38 |

---

## Toolchain

| Tool | Purpose |
|------|---------|
| Synopsys Design Compiler | RTL synthesis, SDC constraint application |
| Cadence Innovus | Floorplan, placement, CTS, routing, PnR |
| Cadence Xcelium | Behavioral and gate-level simulation |
| Synopsys PrimeTime (VCD) | Power analysis with activity annotation |
| TSMC 65nm PDK | Standard cell library, SRAM macros |

---

## Repository Structure

Source files are under NDA and not included. This repo contains documentation only.

```
README.md
figures/
  core_architecture.png          ← mac_array / mac_col / kmem / ofifo / pmem block diagram
  systolic_computation_wave.png  ← K*Q wave propagation across cycles
  normalization_flow.png         ← pmem fetch → sfp → writeback loop
  dual_core_cdc.png              ← two-core clock domains + partial sum exchange
  gls_waveform_step1.png         ← gate-level simulation waveform (Step 1)
  layout_step3.png               ← single core PnR layout (Step 3)
  layout_step4.png               ← dual core PnR layout (Step 4)
  layout_step5.png               ← optimized dual core PnR layout (Step 5)
```

---

## Acknowledgements

Course: ECE 260B, UC San Diego — VLSI Integrated Circuits and Systems Design (Winter 2026)  
Instructor and course staff provided the baseline RTL template and TSMC 65nm PDK access.

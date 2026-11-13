# INT8 Systolic-Array MLP Accelerator on Artix-7 — Project Plan

Nov 13, 2026 · @Master

## Overview

Build an 8×8 INT8 systolic-array matrix-multiply engine in SystemVerilog on the Artix-7 XC7A200T. Use it to run a small quantised MLP (64→32→10, handwritten digits) entirely on-chip, bit-exact against a NumPy integer reference model. Then push the same array through an open-source ASIC flow (sky130) to report silicon area and Fmax. Target: a finished, documented repo by mid-January 2027, ready for Graphcore's technical assessment and spring interviews.

**In scope:** post-training INT8 quantisation in Python, the PE array, tiling controller, on-chip weight/activation buffers, requantisation and ReLU, a UART result link, verification, FPGA implementation, an ASIC trial flow.

**Out of scope:** training on the FPGA, DDR3, convolution layers, an AXI/DMA host interface. These are named as future work in the write-up so the scope is stated, not discovered in interview.

## Fit with the Graphcore Silicon Summer Intern 2027

The project touches three of the four Silicon teams (logical design, verification, physical design), and DFT gets a stretch item. It also adds the AI-workload context the current CV lacks. Requirements are taken from the [Silicon Summer Intern 2027 listing](https://builtinlondon.uk/job/silicon-summer-intern-2027/11766168).

| What Graphcore lists | What this project shows | Evidence you will have |
| --- | --- | --- |
| Logical design: RTL design, SoC integration | Parameterised PE and array RTL, tiling controller, buffers, requantisation unit | RTL in repo; block diagram; resource table |
| Verification methods | Bit-exact differential testing against an integer NumPy model, assertions, constrained-random tiles, mutation testing | Mismatch count, assertion count, mutation results |
| Physical implementation, timing, EDA tools | Vivado timing closure; Yosys + OpenROAD (OpenLane 2) on sky130 for the array | WNS/Fmax on FPGA; ASIC area, Fmax, cell count |
| DFT (stretch) | Scan-chain insertion on the PE array in the ASIC flow | Scan-inserted area overhead |
| Interest in AI compute | Weight-stationary dataflow, data reuse, INT8 quantisation, accumulator sizing | Accuracy float vs INT8; utilisation per layer |
| C, C++ or Python | Python model and tooling; C++ bit-exact model in Verilator (stretch) | Scripts and C++ testbench in repo |

It also reuses your current strengths: you already know differential verification and STA-driven re-architecture from the feed handler, so the interview story becomes "same method, AI workload".

## Architecture

A weight-stationary 8×8 array: each PE holds one INT8 weight, activations flow left to right, partial sums flow top to bottom. Each weight tile is reused across a batch of images, which is where the throughput comes from.

&#91;embedded content: accelerator block diagram · 7 blocks\]

Only the array does arithmetic at scale; everything around it is BRAM and control, so array utilisation is the number that matters.

**Workload**

- Dataset: scikit-learn handwritten digits, 1,797 images of 8×8 pixels, 80/20 train/test split (about 360 test images).
- Model: MLP 64 → 32 (ReLU) → 10, trained in NumPy. 2,368 MACs and 2,368 INT8 weights per inference, plus 42 biases.
- Tiling: layer 1 (32×64) = 32 tiles of 8×8; layer 2 (10×32, padded to 16×32) = 8 tiles.

**Quantisation (integer-only inference)**

- Symmetric INT8 weights, per output channel; INT8 activations; INT32 biases.
- Requantisation after each layer = integer multiply by a fixed-point scale, arithmetic right shift, round, then clamp. ReLU folds into the clamp (lower bound 0).
- The NumPy integer model uses exactly the same arithmetic, so hardware and model must agree bit for bit.

**Datapath decisions you can defend in interview**

- Accumulator width from first principles: |INT8 × INT8| ≤ 128 × 128 = 16,384, and K = 64 terms gives ≤ 2^20. So a 21-bit signed partial sum is enough; use 24 bits to leave room for the bias, instead of a default 32.
- Two builds of the same RTL: multipliers mapped to DSP48E1 slices versus forced into LUTs. Compare LUTs, DSPs and Fmax. This is the area-versus-resource trade-off in miniature.
- Batch size B sets utilisation: loading an 8×8 tile costs about 8 cycles plus about 15 cycles of fill and drain, then B images stream through. Measure utilisation for B = 1, 8, 64.
- Buffers in BRAM: weights (about 2.4 KB), the whole test set (360 × 64 B ≈ 23 KB), ping-pong activation buffers between layers.
- Host link: reuse your UART from the feed handler. The PC sends "run", the board returns predictions, a correct-count and cycle counters. The whole test set runs on-chip, so UART speed never limits the measured throughput.

## Hardware and software

Everything runs on the board you already own and free tools; nothing needs buying.

| Layer | Tool or part | Used for |
| --- | --- | --- |
| FPGA board | ALINX AX7A200B (Artix-7 XC7A200T-2): 740 DSP48E1, 365 BRAM36, about 134k LUTs | Implementation, on-board runs |
| Host link | On-board USB-UART, PC with Python + pyserial | Start runs, read back predictions and counters |
| RTL | SystemVerilog | PE, array, controller, buffers, requantisation |
| FPGA flow | Vivado 2025.2 (synthesis, implementation, XDC, STA, xsim) | Timing closure, resource reports, simulation |
| Fast simulation and lint | Verilator | Lint every commit; C++ testbench for large random runs (stretch) |
| Model and quantisation | Python 3, NumPy, scikit-learn | Dataset, training, INT8 quantisation, integer golden model, test vectors |
| ASIC trial flow | Yosys, OpenROAD via OpenLane 2, SkyWater sky130 PDK | Synthesis, place and route of the PE array; area and Fmax |
| Automation | Bash, Tcl, Make, git, GitHub Actions (stretch) | One-command regression, scripted builds, CI |

**Tech stack in one line:** SystemVerilog · Vivado · Verilator · Python/NumPy/scikit-learn · Yosys/OpenROAD/OpenLane 2 on sky130 · Tcl/Bash · git.

## Target key parameters

The headline result: an MLP running on-chip at ≥ 150 MHz, bit-exact on every test image, losing ≤ 1 point of accuracy to INT8. Targets below are estimates from the architecture; the CV quotes only measured numbers, never these.

| Parameter | Target | How it is measured |
| --- | --- | --- |
| Array size | 8×8 = 64 INT8 MACs per cycle | RTL parameter |
| Clock (Fmax) | ≥ 150 MHz, stretch 200 MHz; WNS ≥ 0, 0 failing endpoints | Vivado routed timing report |
| Peak throughput | 9.6 GMAC/s at 150 MHz | 64 × clock |
| Single-image latency (B = 1) | ≤ 1,000 cycles (≤ 6.7 µs at 150 MHz) | On-chip cycle counter |
| Batched throughput (B = 64) | ≥ 2 million inferences/s | Cycle counter over the whole test set |
| Array utilisation (B = 64) | ≥ 60% of peak | Ideal 37 cycles per image ÷ measured cycles |
| Accuracy | INT8 within 1 point of float | NumPy float vs integer model on the test split |
| Bit-exactness | 0 mismatches on all ≈ 360 test images, on board | Board predictions vs integer model |
| Random-tile verification | 0 mismatches over ≥ 100,000 random 8×8 tiles, including −128/127 corners | Simulation vs NumPy |
| FPGA resources (DSP build) | 64 DSP48E1 (8.6% of 740); LUTs and BRAM reported | Vivado utilisation report |
| DSP vs LUT build | Both builds reported side by side | Two implementation runs |
| ASIC trial (sky130) | Area, Fmax and cell count reported; no target set | OpenLane 2 reports |
| Verification quality | Unit testbenches for PE, array, controller, requantiser; ≥ 3 mutants caught | Regression log, mutation script |

Add a Measured column to this table as results come in; the write-up compares target and measured side by side.

## Verification plan

The same method as the feed handler, bottom up: every level is checked against the NumPy integer model before the next level is built.

1. **PE unit test:** exhaustive INT8 × INT8 products (65,536 pairs) and accumulator overflow corners; assertions on the valid/handshake signals.
2. **Array test:** constrained-random 8×8 tiles with ≥ 100,000 runs, plus directed corners (all −128, all 127, identity, zero rows).
3. **Requantiser test:** every INT32 range boundary, rounding ties, clamp at 0 and 127; compared value by value with the NumPy function.
4. **Controller test:** tiling order, accumulation across K tiles, layer-to-layer buffer swap; assertions that no tile is skipped or used twice.
5. **Full-model differential test:** all test images through RTL vs the integer model, every output value compared, not just the predicted class.
6. **On board:** the same test set from BRAM; the board's correct-count and a checksum of all outputs must equal the model's.
7. **Mutation testing:** break the RTL on purpose (drop a carry bit, swap a tile index, off-by-one shift) and confirm the regression fails.

One command runs levels 1–5 and exits non-zero on any failure, as in your existing regression script.

## Timeline

&#91;embedded content: project timeline · 7 workstreams, 2 milestones\]

Week 1 stays Python-only because ML CW I is due 20 Nov and CW II on 26 Nov; RTL starts on 30 Nov. If time runs short, the ASIC trial shrinks to Yosys synthesis only, while verification and board results are kept whole.

## Deliverables, CV entry and interview prep

**Deliverables**

- Public GitHub repo: RTL, testbenches, Python model, build scripts, one-command regression.
- README with a block diagram and a headline-results table; a `docs/results.md` where every number points to the report it came from.
- Vivado timing and utilisation reports for both builds; OpenLane 2 area and timing reports.
- A short board video or photo of a run reporting the test-set accuracy.

**CV entry template** (fill the X's with measured numbers only)

- INT8 systolic-array MLP accelerator on Artix-7 (individual project). Designed an 8×8 weight-stationary array in SystemVerilog running a quantised 64–32–10 MLP on-chip at X MHz: X inferences/s at X% array utilisation, X% accuracy vs X% in float.
- Verified bit-exact against an integer NumPy model: 0 mismatches over X random tiles and all X test images, on silicon. Sized the accumulator at 24 bits from a bound of 2^20 instead of a default 32.
- Took the PE array through Yosys/OpenROAD on sky130: X µm² at X MHz; compared DSP48 against LUT multipliers (X vs X LUTs, X vs X MHz).

**Interview points to prepare**

- Weight-stationary vs output-stationary vs row-stationary: what each reuses, and why you chose yours.
- Why batch size changes utilisation, and what fill/drain costs at B = 1.
- How requantisation works with a fixed-point multiplier and shift, and where rounding error enters.
- Why INT8 inference loses so little accuracy on this model, and when it would lose more.
- What changes from FPGA to ASIC: DSP blocks vs standard cells, timing closure, area as the cost.

**Risks and fallbacks**

| Risk | Fallback |
| --- | --- |
| Coursework deadlines (ML CW I on 20 Nov, CW II on 26 Nov) eat the first two weeks | Week 1 is Python-only and light; RTL starts after 26 Nov |
| 150 MHz not met | Pipeline the PE output register; report the achieved clock honestly |
| OpenLane 2 setup takes too long | Run Yosys synthesis only with the sky130 library; report area and estimated timing |
| Layer-2 padding (10 → 16 rows) lowers utilisation | Report it per layer; it becomes an interview talking point |

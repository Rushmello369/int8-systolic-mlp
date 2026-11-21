# Results

Every number here points to the report or log it came from. No estimates.

| Metric | Value | Source |
| --- | --- | --- |
| Float accuracy (test split, 360 images) | 96.67 % (348 / 360) | `python model/train.py` |
| INT8 accuracy (test split, 360 images) | 96.11 % (346 / 360) | `python model/golden.py` |
| INT8 loss vs float | −0.56 points (target ≤ 1) | `python model/golden.py` |
| RTL vs integer model, all 360 test images | 0 mismatches (every hidden value, score and prediction) | `scripts\sim.ps1 mlp_core` |
| Clock, routed design | 150 MHz met: WNS +0.580 ns, WHS +0.013 ns | [timing_summary.rpt](../fpga/reports/timing_summary.rpt) |
| Board vs integer model, all 360 test images | 0 mismatches at five batch sizes; cycle counts equal to simulation | [board_run.txt](../fpga/reports/board_run.txt) |
| Whole test set on the board, batch 360 | 15,263 cycles = 101.8 µs at 150 MHz | [board_run.txt](../fpga/reports/board_run.txt) |
| DSP48E1 slices | 72 of 740 (64 PEs + 8 requantisers) | [utilization.rpt](../fpga/reports/utilization.rpt) |

## Float baseline details

Run: `python model/train.py` (seed 0, 150 epochs, learning rate 0.1, batch 32, no L2).

| Quantity | Value | Used for |
| --- | --- | --- |
| Validation accuracy (288 images) | 97.2 % | choosing hyperparameters |
| Training accuracy | 100.0 % | overfitting check |
| max \|W1\| | 1.434 | INT8 weight scale, layer 1 |
| max \|W2\| | 2.237 | INT8 weight scale, layer 2 |
| max hidden activation | 8.660 | INT8 activation scale between layers |

## INT8 model details

Run: `python model/quantize.py`, then `python model/golden.py`. Calibrated on the training split only.

| Quantity | Value | Note |
| --- | --- | --- |
| Images where INT8 and float predict differently | 2 / 360 | |
| Layer 1 weight scale (per neuron) | 0.00272 .. 0.01129 | one scale per row of W1 |
| Hidden activation scale | 0.06819 | 8.660 / 127 |
| Layer 2 weight scale (whole matrix) | 0.01762 | one scale so integer scores compare directly |
| Requantisation multiplier m1 | 164 .. 678, shift 16 | fits 16 bits |
| max \|b1_int\|, max \|b2_int\| | 946, 224 | |
| Largest layer 1 accumulator (test set) | 15,685 (15 bits signed) | worst-case bound 2^20 (21 bits) |
| Largest layer 2 accumulator (test set) | 19,694 (16 bits signed) | |
| Accumulator width chosen | 24 bits | `test_accumulators_fit_24_bits` |
| Hidden INT8 values seen | 0 .. 121 | |
| Python tests | 11 passed | `python -m pytest model -v` |

## RTL simulation

Simulator: Vivado xsim 2025.2. Each testbench prints one PASS or FAIL line;
run it with `powershell -ExecutionPolicy Bypass -File scripts\sim.ps1 <name>`.
All runs use seed 1.

| Testbench | What it checks | Checks passed |
| --- | --- | --- |
| `pe` | every INT8 × INT8 product, accumulator wrap | 85,799 |
| `pe_array` (8×8) | 100,000 random tiles plus corners | 6,200,720 |
| `skew_buffer` (8 lanes, 8 bits) | 100,000 random cycles plus impulse and extremes | 800,172 |
| `systolic_core` (8×8) | 20,000 random tiles; latency = Rows + Cols − 1 = 15 cycles | 1,240,704 |
| `requant` | acc sweeps over ±2^17 for six multipliers, extremes, 1,000,000 random | 2,572,882 |
| `argmax` | ties, one-winner cases, 400,000 random | 398,780 |
| `mlp_core` | all 360 images at five batch sizes against `model/gen_vectors.py` | 26,810 |
| `fpga_top` | the same through the MMCM, UART and host protocol | 725 |

Non-square shapes (3×5, 5×3) of `pe_array` and `systolic_core` also pass.

### Cycles per run of the whole test set (360 images)

From `scripts\sim.ps1 mlp_core`. Utilisation = 2,368 useful MACs per image × 360
images ÷ (64 MACs per cycle × cycles).

| Batch size | Cycles | Array utilisation |
| --- | --- | --- |
| 360 | 15,263 | 87.3 % |
| 64 | 19,573 | 68.1 % |
| 8 | 53,191 | 25.0 % |
| 7 | 59,225 | 22.5 % |
| 1 | 324,721 | 4.1 % |

Each tile costs 8 load cycles and 13 drain cycles whatever the batch size, so
small batches leave the array idle most of the time.

### Mutation testing

Each bug was planted by hand, the testbench run, and the file restored.

| Planted bug | Caught by |
| --- | --- |
| requant drops the rounding half-step | `requant`: 97,358 mismatches |
| argmax gives ties to the higher digit | `argmax`: 87,332 mismatches |
| drain one cycle shorter than Rows + Cols − 3 | `mlp_core`: assertion `NoLoadInFlight_A` |
| bias not added on the first input block | `mlp_core`: 2,166 mismatches |
| wrong input block in the layer 2 tile index | `mlp_core`: 1,060 mismatches |

## FPGA implementation

Vivado 2025.2, part xc7a200tfbg484-2, routed design and bitstream.
Build: `powershell -ExecutionPolicy Bypass -File scripts\build.ps1`.

| Quantity | Value | Source |
| --- | --- | --- |
| System clock | 150 MHz (MMCM from the 200 MHz board clock) | [clocks.rpt](../fpga/reports/clocks.rpt) |
| Worst setup slack (WNS) | +0.580 ns, 0 failing endpoints | [timing_summary.rpt](../fpga/reports/timing_summary.rpt) |
| Worst hold slack (WHS) | +0.013 ns, 0 failing endpoints | [timing_summary.rpt](../fpga/reports/timing_summary.rpt) |
| Slice LUTs | 1,586 (1.19 %) | [utilization.rpt](../fpga/reports/utilization.rpt) |
| Slice registers | 2,226 (0.83 %) | [utilization.rpt](../fpga/reports/utilization.rpt) |
| Block RAM tiles | 19.5 (5.34 %) | [utilization.rpt](../fpga/reports/utilization.rpt) |
| DSP48E1 | 72 (9.73 %) | [utilization.rpt](../fpga/reports/utilization.rpt) |

The first build missed 150 MHz (WNS −1.831 ns, 72 failing endpoints). Three fixes closed it:
the argmax tree got one register per compare level, the PE multiply-add moved into DSP48
slices (`use_dsp`), and a register was added between the input memories and the array.

## On the board

ALINX AX7A200B, bitstream from `scripts\build.ps1`, programmed over JTAG from the
Vivado Hardware Manager. Run: `.venv\Scripts\python.exe host\run_board.py --port COM4 --batch <n>`.
Full output: [board_run.txt](../fpga/reports/board_run.txt).

| Batch size | Predictions vs model | Correct | Checksum | Cycles (board = simulation) | Time at 150 MHz |
| --- | --- | --- | --- | --- | --- |
| 360 | 360 / 360 match | 346 / 360 | 0x001b5b5f | 15,263 | 101.8 µs |
| 64 | 360 / 360 match | 346 / 360 | 0x001b5b5f | 19,573 | 130.5 µs |
| 8 | 360 / 360 match | 346 / 360 | 0x001b5b5f | 53,191 | 354.6 µs |
| 7 | 360 / 360 match | 346 / 360 | 0x001b5b5f | 59,225 | 394.8 µs |
| 1 | 360 / 360 match | 346 / 360 | 0x001b5b5f | 324,721 | 2,164.8 µs |

The board's cycle counts equal the simulated ones exactly: the design has no
data-dependent timing, so any difference would point to a fault, not noise.

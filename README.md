# INT8 Systolic-Array MLP Accelerator

An 8×8 weight-stationary INT8 systolic array in SystemVerilog, running a quantised
64→32→10 MLP (scikit-learn handwritten digits) entirely on an Artix-7 XC7A200T,
bit-exact against a NumPy integer reference model. The PE array is also taken
through Yosys/OpenROAD (OpenLane 2) on sky130 for area and Fmax.

> Status: work in progress. Numbers below are filled in only from measured reports.

## Headline results

| Metric | Target | Measured |
| --- | --- | --- |
| Clock (Fmax, routed) | ≥ 150 MHz | – |
| Batched throughput (B = 64) | ≥ 2 M inferences/s | – |
| Array utilisation (B = 64) | ≥ 60 % | – |
| Accuracy, float vs INT8 | ≤ 1 pt loss | – |
| Bit-exact mismatches, test set on board | 0 | – |
| Random-tile mismatches (≥ 100k tiles) | 0 | – |

Every number links to its source report in [docs/results.md](docs/results.md).

## Repository layout

```
model/      Python: dataset, float training, INT8 quantisation, integer golden model, test vectors
rtl/        SystemVerilog
  pkg/        shared parameters and types
  core/       PE, systolic array, requantiser
  ctrl/       tiling controller
  mem/        weight / activation buffers (BRAM)
  io/         UART
  top/        FPGA top level
tb/         testbenches, one folder per unit + full-model differential test
fpga/       Vivado: XDC constraints, Tcl build scripts, saved reports
asic/       OpenLane 2 config and reports (sky130)
host/       PC-side pyserial script: start a run, read predictions and counters
scripts/    regression and mutation-testing scripts
docs/       plan, architecture notes, results
```

## Quick start

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash; use .venv/bin/activate on Linux
pip install -r requirements.txt
make model                         # train, quantise, write test vectors
```

## Toolchain

SystemVerilog · Vivado 2025.2 · Verilator · Python/NumPy/scikit-learn ·
Yosys/OpenROAD/OpenLane 2 on sky130 · Tcl/Bash/Make · git

## Out of scope

Training on the FPGA, DDR3, convolution layers, AXI/DMA host interface (future work).

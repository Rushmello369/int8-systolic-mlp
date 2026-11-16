# Roadmap

Each phase has an exit test. Do not start the next phase until the exit test passes:
every hardware level is checked against the NumPy integer model before the next is built.

| # | Phase | Dates (2026–27) | Main files | Exit test |
| --- | --- | --- | --- | --- |
| 0 | Repo and environment | 13 Nov | `README.md`, `.gitignore`, `requirements.txt` | Repo on GitHub, `.venv` works |
| 1 | Float MLP in NumPy | 14–17 Nov | `model/dataset.py`, `model/train.py` | Test accuracy ≥ 95 %, weights saved |
| 2 | INT8 quantisation + integer golden model | 18–29 Nov (light, coursework) | `model/quantize.py`, `model/golden.py`, `model/test_golden.py` | INT8 within 1 pt of float; golden model uses only integer ops |
| 3 | Test vectors + PE RTL | 30 Nov – 6 Dec | `model/gen_vectors.py`, `rtl/pkg/accel_pkg.sv`, `rtl/core/pe.sv`, `tb/pe/` | 65,536 products exact; Verilator lint clean |
| 4 | 8×8 systolic array | 7–13 Dec | `rtl/core/pe_array.sv`, `tb/pe_array/` | 0 mismatches over ≥ 100k random tiles + corners |
| 5 | Requantiser, buffers, tiling controller | 14–23 Dec | `rtl/core/requant.sv`, `rtl/mem/`, `rtl/ctrl/` | Requant boundaries exact; controller assertions hold |
| 6 | Full-model differential simulation | 24–28 Dec | `tb/top/`, `scripts/regress.sh` | All test images bit-exact in sim; `make regress` green |
| 7 | FPGA implementation | 29 Dec – 5 Jan | `rtl/io/uart_*.sv`, `rtl/top/fpga_top.sv`, `fpga/` | WNS ≥ 0 at ≥ 150 MHz; DSP vs LUT builds reported |
| 8 | On-board run + mutation testing | 6–9 Jan | `host/run_board.py`, `scripts/mutate.py` | Board checksum = model; ≥ 3 mutants caught |
| 9 | ASIC trial (sky130) | 10–13 Jan | `asic/openlane/config.json` | Area, Fmax, cell count reported (fallback: Yosys only) |
| 10 | Write-up | 14–17 Jan | `README.md`, `docs/results.md`, `docs/architecture.md` | Every number links to a report |

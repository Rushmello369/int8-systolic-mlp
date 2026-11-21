"""Run the 360 test images on the board and check the results against the model.

What it does:
  1. Sends the run command over the USB-UART: 'R', then the batch size as two
     bytes, high byte first.
  2. Reads the reply: one predicted digit per test image (360 bytes), then the
     number of correct predictions (2 bytes), the checksum of all scores
     (4 bytes) and the cycle count of the run (4 bytes). Numbers are big-endian.
  3. Compares the digits, the correct count and the checksum with the files
     written by model/gen_vectors.py. They must match exactly.

The protocol is described in rtl/io/host_if.sv.

Usage, from the project root, with the board programmed:
    .venv\\Scripts\\python.exe host\\run_board.py --port COM5
    .venv\\Scripts\\python.exe host\\run_board.py --port COM5 --batch 64

The port name is in Windows Device Manager, under "Ports (COM & LPT)"
(the board's CP2102 USB-UART). Exit code 0 means everything matched.
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "model" / "data"

BAUD_RATE = 115200  # Must match BaudRate in rtl/top/fpga_top.sv
CLOCK_HZ = 150_000_000  # System clock of fpga_top, for turning cycles into time
MAX_BATCH = 512  # MaxBatch in rtl/pkg/accel_pkg.sv
RUN_COMMAND = b"R"
SUMMARY_BYTES = 10  # 2 correct count + 4 checksum + 4 cycle count

# Cycles measured in simulation (scripts\sim.ps1 mlp_core, see docs/results.md).
# The board runs the same RTL, so its cycle count must be identical.
SIMULATED_CYCLES = {360: 15263, 64: 19573, 8: 53191, 7: 59225, 1: 324721}


def read_hex_file(file_name):
    """Read a $readmemh file: one hexadecimal number per line."""
    numbers = []
    path = DATA_DIR / file_name
    for line in path.read_text().splitlines():
        line = line.strip()
        if line != "":
            numbers.append(int(line, 16))
    return numbers


def build_command(batch_size):
    """The three bytes that start a run: 'R', batch size high byte, low byte."""
    return RUN_COMMAND + batch_size.to_bytes(2, "big")


def parse_reply(reply, num_images):
    """Split the reply bytes into digits, correct count, checksum and cycles."""
    digits = list(reply[:num_images])
    summary = reply[num_images:num_images + SUMMARY_BYTES]
    correct_count = int.from_bytes(summary[0:2], "big")
    checksum = int.from_bytes(summary[2:6], "big")
    cycles = int.from_bytes(summary[6:10], "big")
    return digits, correct_count, checksum, cycles


def read_exactly(port, num_bytes, timeout_s):
    """Read num_bytes from the serial port, or fewer if timeout_s runs out."""
    received = bytearray()
    deadline = time.monotonic() + timeout_s
    while len(received) < num_bytes and time.monotonic() < deadline:
        chunk = port.read(num_bytes - len(received))
        received.extend(chunk)
    return bytes(received)


def check_results(digits, correct_count, checksum, expected_digits, expected_summary):
    """Compare the board's answers with the model. Returns the number of problems."""
    problems = 0

    # Every predicted digit, image by image
    wrong_images = []
    for image in range(len(expected_digits)):
        if digits[image] != expected_digits[image]:
            wrong_images.append(image)
    if len(wrong_images) == 0:
        print(f"Predictions: all {len(expected_digits)} match the model")
    else:
        problems += len(wrong_images)
        print(f"Predictions: {len(wrong_images)} differ from the model")
        for image in wrong_images[:10]:
            print(f"  image {image}: board {digits[image]}, model {expected_digits[image]}")

    expected_correct = expected_summary[0]
    expected_checksum = expected_summary[1]

    if correct_count == expected_correct:
        print(f"Correct count: {correct_count} / {len(expected_digits)} (matches the model)")
    else:
        problems += 1
        print(f"Correct count: board {correct_count}, model {expected_correct}  MISMATCH")

    if checksum == expected_checksum:
        print(f"Checksum: 0x{checksum:08x} (matches the model)")
    else:
        problems += 1
        print(f"Checksum: board 0x{checksum:08x}, model 0x{expected_checksum:08x}  MISMATCH")

    return problems


def main():
    parser = argparse.ArgumentParser(description="Run the test set on the board.")
    parser.add_argument("--port", required=True, help="serial port, e.g. COM5")
    parser.add_argument("--batch", type=int, default=360,
                        help=f"images per batch, 1 to {MAX_BATCH} (default 360)")
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="seconds to wait for the whole reply (default 5)")
    args = parser.parse_args()

    if args.batch < 1 or args.batch > MAX_BATCH:
        parser.error(f"--batch must be between 1 and {MAX_BATCH}")

    try:
        import serial  # pyserial, listed in requirements.txt
    except ImportError:
        print("pyserial is not installed: run  .venv\\Scripts\\pip install pyserial")
        return 2

    expected_digits = read_hex_file("expected_pred.mem")
    expected_summary = read_hex_file("expected_summary.mem")
    num_images = len(expected_digits)
    reply_length = num_images + SUMMARY_BYTES

    with serial.Serial(args.port, BAUD_RATE, timeout=0.1) as port:
        # Throw away anything left over from an earlier, interrupted run.
        port.reset_input_buffer()
        port.write(build_command(args.batch))
        reply = read_exactly(port, reply_length, args.timeout)

    if len(reply) < reply_length:
        print(f"Timeout: got {len(reply)} of {reply_length} bytes.")
        print("Check the port name, that the board is programmed, and that the")
        print("LED for 'MMCM locked' is lit. Press the reset key and try again.")
        return 1

    digits, correct_count, checksum, cycles = parse_reply(reply, num_images)
    problems = check_results(digits, correct_count, checksum, expected_digits,
                             expected_summary)

    run_time_us = cycles / CLOCK_HZ * 1e6
    print(f"Batch size {args.batch}: {cycles} cycles = {run_time_us:.1f} us at 150 MHz")
    if args.batch in SIMULATED_CYCLES:
        simulated = SIMULATED_CYCLES[args.batch]
        if cycles == simulated:
            print(f"Cycle count matches simulation ({simulated})")
        else:
            problems += 1
            print(f"Cycle count: board {cycles}, simulation {simulated}  MISMATCH")

    if problems == 0:
        print("BOARD PASS")
        return 0
    print(f"BOARD FAIL: {problems} problem(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""Tests for run_board.py that need no board: command bytes and reply parsing.

Run from the project root:
    .venv\\Scripts\\python.exe -m pytest host -v
Needs the .mem files from model/gen_vectors.py.
"""

import run_board


def make_reply(digits, correct_count, checksum, cycles):
    """Build the bytes the board would send, in the order host_if.sv sends them."""
    reply = bytes(digits)
    reply += correct_count.to_bytes(2, "big")
    reply += checksum.to_bytes(4, "big")
    reply += cycles.to_bytes(4, "big")
    return reply


def test_command_is_r_then_batch_size_high_byte_first():
    assert run_board.build_command(360) == b"R\x01\x68"
    assert run_board.build_command(1) == b"R\x00\x01"


def test_parse_reply_splits_digits_and_summary():
    reply = make_reply([7, 6, 3], 0x015A, 0x001B5B5F, 15263)
    digits, correct_count, checksum, cycles = run_board.parse_reply(reply, 3)
    assert digits == [7, 6, 3]
    assert correct_count == 0x015A
    assert checksum == 0x001B5B5F
    assert cycles == 15263


def test_model_reply_passes_the_check():
    expected_digits = run_board.read_hex_file("expected_pred.mem")
    expected_summary = run_board.read_hex_file("expected_summary.mem")
    reply = make_reply(expected_digits, expected_summary[0], expected_summary[1], 15263)
    digits, correct_count, checksum, _ = run_board.parse_reply(reply, len(expected_digits))
    problems = run_board.check_results(digits, correct_count, checksum,
                                       expected_digits, expected_summary)
    assert problems == 0


def test_one_wrong_digit_is_reported():
    expected_digits = run_board.read_hex_file("expected_pred.mem")
    expected_summary = run_board.read_hex_file("expected_summary.mem")
    board_digits = list(expected_digits)
    board_digits[5] = (board_digits[5] + 1) % 10
    problems = run_board.check_results(board_digits, expected_summary[0],
                                       expected_summary[1], expected_digits,
                                       expected_summary)
    assert problems == 1

// int8_mlp: INT8 MLP accelerator on a weight-stationary systolic array (Artix-7)
//
// Description:
//   Self-checking testbench for requant.sv.
//   The model is requantize() from model/golden.py, in 64-bit integers:
//     clamp((acc * m + 2^15) >> 16, 0, 127)
//   One input is driven every cycle; its result is checked 2 cycles later,
//   so the testbench also proves the pipeline depth.
//   Tests, in order:
//     1. Sweeps: every acc in -SweepRange..SweepRange for a set of
//        multipliers. m = 2^15 makes every odd acc an exact tie (x.5),
//        so the sweeps cover rounding ties, the clamp at 0 (ReLU) and the
//        saturation at 127.
//     2. Extremes: the largest and smallest accumulators with the largest
//        and smallest multipliers.
//     3. Random: NumRandom random (acc, m) pairs over the full input range.
//   Prints one line, PASS or FAIL with the first mismatch; FAIL exits non-zero.
//   +seed=N picks the random run.

`default_nettype none

module requant_tb;
    localparam int unsigned AccWidth = 24;  // DUT parameter under test
    localparam int unsigned MultWidth = 16;  // DUT parameter under test
    localparam int unsigned Shift = 16;  // DUT parameter under test
    localparam int unsigned OutWidth = 8;  // DUT parameter under test
    localparam int Latency = 2;  // cycles from acc_i to value_o
    localparam int SweepRange = 1 << 17;  // sweep acc over -2^17..2^17
    localparam int NumRandom = 1000000;  // random pairs in test 3

    // ---- DUT signals ----
    logic                        clk_i = 1'b0;  // 100 MHz, posedge at 5, 15, ...
    logic signed [ AccWidth-1:0] acc_i = '0;  // DUT input, driven at negedge
    logic        [MultWidth-1:0] mult_i = '0;  // DUT input, driven at negedge
    logic        [ OutWidth-1:0] value_o;  // DUT output, checked at negedge

    // ---- Model ----
    int expected[$];  // results still in the pipeline, oldest first
    longint acc_history[$];  // their inputs, for the FAIL message
    longint mult_history[$];  // their inputs, for the FAIL message

    // ---- Bookkeeping ----
    int unsigned seed = 1;  // +seed=N to reproduce a different random run
    int checks = 0;  // outputs checked so far
    int errors = 0;  // mismatching outputs
    string first = "";  // the first mismatch, for the FAIL line

    // Device under test
    requant #(
        .AccWidth (AccWidth),
        .MultWidth(MultWidth),
        .Shift    (Shift),
        .OutWidth (OutWidth)
    ) dut (
        .clk_i  (clk_i),
        .acc_i  (acc_i),
        .mult_i (mult_i),
        .value_o(value_o)
    );

    // svstyle: waive LNG-003 testbench clock generator, a delay loop rather than logic
    always #5 clk_i = ~clk_i;

    // requantize() from golden.py. >>> on a signed longint rounds towards -infinity,
    // like NumPy's >> on int64.
    function automatic int model_requant(input longint acc, input longint mult);
        longint shifted;  // rounded to nearest, before the clamp
        shifted = (acc * mult + (longint'(1) <<< (Shift - 1))) >>> Shift;
        if (shifted < 0) begin
            return 0;
        end
        if (shifted > 127) begin
            return 127;
        end
        return int'(shifted);
    endfunction

    // One cycle: check the result that leaves the pipeline now, then drive a new input.
    task automatic step(input longint acc, input longint mult);
        int want;  // expected value of the oldest input
        longint old_acc;  // its accumulator
        longint old_mult;  // its multiplier

        @(negedge clk_i);
        if (expected.size() == Latency) begin
            want     = expected.pop_front();
            old_acc  = acc_history.pop_front();
            old_mult = mult_history.pop_front();
            checks++;
            if (int'(value_o) != want) begin
                if (errors == 0) begin
                    first = $sformatf("acc %0d, m %0d: value_o=%0d, want %0d", old_acc, old_mult,
                                      value_o, want);
                end
                errors++;
            end
        end
        acc_i  = AccWidth'(acc);
        mult_i = MultWidth'(mult);
        expected.push_back(model_requant(acc, mult));
        acc_history.push_back(acc);
        mult_history.push_back(mult);
    endtask

    longint sweep_mults[6] = '{0, 1, 164, 678, 32768, 65535};  // multipliers for test 1
    longint acc_extremes[4] = '{-(1 << 23), -(1 << 23) + 1, (1 << 23) - 2, (1 << 23) - 1};  // test 2
    longint mult_extremes[3] = '{0, 1, 65535};  // test 2
    int unsigned seed_found;  // 1 if +seed=N was given (xsim needs the result stored)
    int unsigned first_random;  // $urandom(seed) seeds the generator; xsim needs the result stored

    initial begin
        seed_found   = $value$plusargs("seed=%d", seed);
        first_random = $urandom(seed);

        // ---- Test 1: sweeps ----
        foreach (sweep_mults[i]) begin
            for (int acc = -SweepRange; acc <= SweepRange; acc++) begin
                step(acc, sweep_mults[i]);
            end
        end

        // ---- Test 2: extremes ----
        foreach (acc_extremes[i]) begin
            foreach (mult_extremes[j]) begin
                step(acc_extremes[i], mult_extremes[j]);
            end
        end

        // ---- Test 3: random ----
        repeat (NumRandom) begin
            // Sign-extend a random 24-bit pattern to get a signed accumulator
            step(longint'($signed(AccWidth'($urandom))), longint'(MultWidth'($urandom)));
        end

        // Flush the pipeline so the last inputs are checked too
        repeat (Latency) begin
            step(0, 0);
        end

        if (errors != 0) begin
            $fatal(1, "FAIL: %0d of %0d checks, seed %0d, first at %s", errors, checks, seed,
                   first);
        end
        $display("PASS: %0d checks, seed %0d", checks, seed);
        $finish;
    end
endmodule

`resetall

// Copyright (c) 2026 <Company>
// All rights reserved
//
// Description:
//   Self-checking testbench for pe.sv.
//   A model beside the DUT predicts every output from the spec:
//     weight_o = the last weight loaded
//     act_o    = act_i one cycle earlier
//     psum_o   = (psum_i + act_i * weight) one cycle earlier, wrapped to AccWidth bits
//   Tests, in order:
//     1. Exhaustive: every INT8 weight x every INT8 activation (65,536 products),
//        each with a random partial sum.
//     2. Accumulator corners: the largest and smallest sums, and one deliberate wrap.
//     3. Random: weight loads mixed with compute cycles, to check the weight holds.
//   Prints one line, PASS or FAIL with the first mismatch; FAIL exits non-zero.
//   +seed=N picks the random run.

`default_nettype none

module pe_tb;
    localparam int unsigned DataWidth = 8;  // DUT parameter under test
    localparam int unsigned AccWidth = 24;  // DUT parameter under test
    localparam int NumRandom = 20000;  // random cycles in test 3

    // ---- DUT signals ----
    logic                        clk_i = 1'b0;  // 100 MHz, posedge at 5, 15, ...
    logic                        weight_load_i = 1'b0;  // DUT input, driven at negedge
    logic signed [DataWidth-1:0] weight_i = '0;  // DUT input, driven at negedge
    logic signed [DataWidth-1:0] act_i = '0;  // DUT input, driven at negedge
    logic signed [ AccWidth-1:0] psum_i = '0;  // DUT input, driven at negedge
    logic signed [DataWidth-1:0] weight_o;  // DUT output, checked at negedge
    logic signed [DataWidth-1:0] act_o;  // DUT output, checked at negedge
    logic signed [ AccWidth-1:0] psum_o;  // DUT output, checked at negedge

    // ---- Model: the outputs we expect after the next rising edge ----
    logic signed [DataWidth-1:0] weight_m = '0;  // expected weight_o
    logic signed [DataWidth-1:0] act_m = '0;  // expected act_o
    logic signed [ AccWidth-1:0] psum_m = '0;  // expected psum_o
    logic                        check_enabled = 1'b0;  // the PE has no reset: outputs are X at first

    // ---- Bookkeeping ----
    int unsigned seed = 1;  // +seed=N to reproduce a different random run
    int checks = 0;  // checks done so far
    int errors = 0;  // mismatching checks
    string first = "";  // the first mismatch, for the FAIL line

    // Device under test
    pe #(
        .DataWidth(DataWidth),
        .AccWidth (AccWidth)
    ) dut (
        .clk_i        (clk_i),
        .weight_load_i(weight_load_i),
        .weight_i     (weight_i),
        .weight_o     (weight_o),
        .act_i        (act_i),
        .act_o        (act_o),
        .psum_i       (psum_i),
        .psum_o       (psum_o)
    );

    // svstyle: waive LNG-003 testbench clock generator, a delay loop rather than logic
    always #5 clk_i = ~clk_i;

    task automatic fail(input string msg);
        if (errors == 0) begin
            first = $sformatf("check %0d: %s", checks, msg);
        end
        errors++;
    endtask

    // One clock cycle:
    //   1. at the falling edge, compare what the last rising edge produced with the model
    //   2. drive the next inputs
    //   3. advance the model by what the next rising edge will do
    task automatic step(input logic load, input logic signed [DataWidth-1:0] weight,
                        input logic signed [DataWidth-1:0] act,
                        input logic signed [AccWidth-1:0] psum);
        longint full_sum;  // exact sum before wrapping to AccWidth bits

        @(negedge clk_i);
        if (check_enabled) begin
            checks++;
            if (weight_o !== weight_m) begin
                fail($sformatf("weight_o=%0d, want %0d", weight_o, weight_m));
            end else if (act_o !== act_m) begin
                fail($sformatf("act_o=%0d, want %0d", act_o, act_m));
            end else if (psum_o !== psum_m) begin
                fail($sformatf("psum_o=%0d, want %0d", psum_o, psum_m));
            end
        end

        weight_load_i = load;
        weight_i      = weight;
        act_i         = act;
        psum_i        = psum;

        // The product uses the weight held BEFORE this edge; a load takes effect after it.
        full_sum = longint'(psum) + longint'(act) * longint'(weight_m);
        psum_m   = AccWidth'(full_sum);
        act_m    = act;
        if (load) begin
            weight_m = weight;
        end
    endtask

    // A random value for psum_i, kept far enough from the limits that it never wraps.
    function automatic logic signed [AccWidth-1:0] safe_psum();
        safe_psum = AccWidth'($urandom_range(2 ** 22) - 2 ** 21);
    endfunction

    int unsigned seed_found;  // 1 if +seed=N was given (xsim needs the result stored)
    int unsigned first_random;  // $urandom(seed) seeds the generator; xsim needs the result stored

    initial begin
        seed_found   = $value$plusargs("seed=%d", seed);
        first_random = $urandom(seed);

        // ---- Warm-up: the PE has no reset, so its weight starts as X ----
        // Load weight 0, then run one cycle so psum_o no longer depends on the X weight.
        step(1'b1, '0, '0, '0);
        step(1'b0, '0, '0, '0);
        check_enabled = 1'b1;

        // ---- Test 1: every weight x every activation ----
        for (int weight = -128; weight <= 127; weight++) begin
            step(1'b1, DataWidth'(weight), '0, '0);  // load the weight
            for (int act = -128; act <= 127; act++) begin
                step(1'b0, '0, DataWidth'(act), safe_psum());
            end
        end

        // ---- Test 2: accumulator corners ----
        step(1'b1, -8'sd128, '0, '0);  // weight -128
        // largest product +16,384 onto the largest sum that still fits
        step(1'b0, '0, -8'sd128, AccWidth'(2 ** 23 - 1 - 16384));
        step(1'b1, 8'sd127, '0, '0);  // weight 127
        // most negative product -16,256 down to exactly the smallest sum
        step(1'b0, '0, -8'sd128, AccWidth'(-(2 ** 23) + 16256));
        step(1'b1, 8'sd1, '0, '0);  // weight 1
        // one past the largest sum: wraps to the smallest (the design must never get here)
        step(1'b0, '0, 8'sd1, AccWidth'(2 ** 23 - 1));

        // ---- Test 3: random loads and compute cycles ----
        repeat (NumRandom) begin
            step(1'($urandom_range(3) == 0), DataWidth'($urandom), DataWidth'($urandom),
                 safe_psum());
        end
        step(1'b0, '0, '0, '0);  // check the last edge

        if (errors != 0) begin
            $fatal(1, "FAIL: %0d of %0d checks, seed %0d, first at %s", errors, checks, seed,
                   first);
        end
        $display("PASS: %0d checks, seed %0d", checks, seed);
        $finish;
    end
endmodule

`resetall

// int8_mlp: INT8 MLP accelerator on a weight-stationary systolic array (Artix-7)
//
// Description:
//   Self-checking testbench for skew_buffer.sv.
//   The model is a history of every vector driven in: history[0] is the
//   vector driven this cycle, history[d] the one driven d cycles ago.
//   Each cycle the testbench drives a new vector and then checks
//     data_o[k] == lane k of history[k]
//   for every lane k whose input is old enough to have been driven.
//   Tests, in order:
//     1. Impulse: one vector of distinct values, then zeros. Each lane's
//        value appears exactly once, k cycles late. Catches a wrong delay
//        and lanes crossed with each other.
//     2. Extremes: all-ones and all-zeros vectors in turn. Catches stuck
//        or dropped bits.
//     3. Random: NumCycles cycles of random vectors.
//   Run it with several shapes (scripts/sim.ps1 -Generics Lanes=1,Width=24):
//   Lanes = 1 has only the pass-through lane, and Width != 8 catches a
//   width hard-coded in the RTL. Width must be at most 32 ($urandom).
//   Prints one line, PASS or FAIL with the first mismatch; FAIL exits non-zero.
//   +seed=N picks the random run.

`default_nettype none

module skew_buffer_tb #(
    parameter int unsigned Lanes = 8,  // DUT lanes under test
    parameter int unsigned Width = 8   // DUT width under test, at most 32
);
    localparam int NumCycles = 100000;  // random cycles in test 3

    // ---- DUT signals ----
    logic                        clk_i = 1'b0;  // 100 MHz, posedge at 5, 15, ...
    logic [Lanes-1:0][Width-1:0] data_i = '0;  // DUT input, driven at negedge
    logic [Lanes-1:0][Width-1:0] data_o;  // DUT output, checked 1 ns after each drive

    // ---- Model ----
    logic [Lanes-1:0][Width-1:0] history[$];  // history[d] = vector driven d cycles ago

    // ---- Bookkeeping ----
    int unsigned seed = 1;  // +seed=N to reproduce a different random run
    int checks = 0;  // outputs checked so far
    int errors = 0;  // mismatching outputs
    int cycle = 0;  // cycles driven so far
    string first = "";  // the first mismatch, for the FAIL line

    // Device under test
    skew_buffer #(
        .Lanes(Lanes),
        .Width(Width)
    ) dut (
        .clk_i (clk_i),
        .data_i(data_i),
        .data_o(data_o)
    );

    // svstyle: waive LNG-003 testbench clock generator, a delay loop rather than logic
    always #5 clk_i = ~clk_i;

    task automatic fail(input string msg);
        if (errors == 0) begin
            first = $sformatf("check %0d: %s", checks, msg);
        end
        errors++;
    endtask

    // Drive one vector at the falling edge, record it, then check every lane.
    // Lane 0 is a wire, so it shows the new vector at once; lane k shows the
    // vector driven k cycles ago, captured by the last k rising edges.
    // Lanes whose input was never driven (k > cycles so far) still hold X: skip them.
    task automatic step(input logic [Lanes-1:0][Width-1:0] vector);
        @(negedge clk_i);
        data_i = vector;
        history.push_front(vector);
        if (history.size() > Lanes) begin
            void'(history.pop_back());  // older vectors have left every lane
        end
        #1;
        for (int lane = 0; lane < Lanes; lane++) begin
            if (lane < history.size()) begin
                checks++;
                if (data_o[lane] !== history[lane][lane]) begin
                    fail($sformatf("cycle %0d lane %0d: data_o=%0h, want %0h", cycle, lane,
                                   data_o[lane], history[lane][lane]));
                end
            end
        end
        cycle++;
    endtask

    logic [Lanes-1:0][Width-1:0] vector;  // the vector being built for the next step
    int unsigned seed_found;  // 1 if +seed=N was given (xsim needs the result stored)
    int unsigned first_random;  // $urandom(seed) seeds the generator; xsim needs the result stored

    initial begin
        seed_found   = $value$plusargs("seed=%d", seed);
        first_random = $urandom(seed);

        // ---- Test 1: impulse ----
        // Lane k carries k+1, so every lane is different and none is zero.
        for (int lane = 0; lane < Lanes; lane++) begin
            vector[lane] = Width'(lane + 1);
        end
        step(vector);
        repeat (Lanes) begin
            step('0);
        end

        // ---- Test 2: extremes ----
        repeat (Lanes) begin
            step('1);
            step('0);
        end

        // ---- Test 3: random ----
        repeat (NumCycles) begin
            for (int lane = 0; lane < Lanes; lane++) begin
                vector[lane] = Width'($urandom);
            end
            step(vector);
        end

        if (errors != 0) begin
            $fatal(1, "FAIL: %0d of %0d checks, Lanes %0d, Width %0d, seed %0d, first at %s",
                   errors, checks, Lanes, Width, seed, first);
        end
        $display("PASS: %0d checks, Lanes %0d, Width %0d, seed %0d", checks, Lanes, Width, seed);
        $finish;
    end
endmodule

`resetall

// int8_mlp: INT8 MLP accelerator on a weight-stationary systolic array (Artix-7)
//
// Description:
//   Self-checking testbench for argmax.sv.
//   The model is np.argmax written as a plain scan: walk the earlier best
//   (if used) and then the lanes in order, and replace the best only with a
//   strictly larger valid score. That gives the lowest index on every tie.
//   One input is driven every cycle and checked log2(Lanes) + 1 cycles later.
//   Tests, in order:
//     1. All equal: every lane and the earlier best hold the same score, so
//        the lowest index must win.
//     2. One winner: a single lane is larger, in every lane position.
//     3. Random with ties: scores from a range of only 4 values, so most
//        inputs contain ties; random valid masks and use_prev.
//     4. Random wide: scores over the full signed range.
//   Prints one line, PASS or FAIL with the first mismatch; FAIL exits non-zero.
//   +seed=N picks the random run.

`default_nettype none

module argmax_tb;
    localparam int unsigned Lanes = 8;  // DUT parameter under test
    localparam int unsigned ScoreWidth = 24;  // DUT parameter under test
    localparam int unsigned IndexWidth = 4;  // DUT parameter under test
    localparam int Latency = $clog2(Lanes) + 1;  // cycles from inputs to outputs: 4
    localparam int NumRandom = 200000;  // random inputs in tests 3 and 4, each

    // ---- DUT signals ----
    logic                                 clk_i = 1'b0;  // 100 MHz, posedge at 5, 15, ...
    logic        [Lanes-1:0][ScoreWidth-1:0] scores_i = '0;  // DUT input, driven at negedge
    logic        [     Lanes-1:0]          lane_valid_i = '0;  // DUT input, driven at negedge
    logic        [IndexWidth-1:0]          base_index_i = '0;  // DUT input, driven at negedge
    logic                                  use_prev_i = 1'b0;  // DUT input, driven at negedge
    logic signed [ScoreWidth-1:0]          prev_score_i = '0;  // DUT input, driven at negedge
    logic        [IndexWidth-1:0]          prev_index_i = '0;  // DUT input, driven at negedge
    logic signed [ScoreWidth-1:0]          best_score_o;  // DUT output, checked at negedge
    logic        [IndexWidth-1:0]          best_index_o;  // DUT output, checked at negedge

    // ---- Model ----
    longint expected_score[$];  // results still in the pipeline, oldest first
    int expected_index[$];  // their indices

    // ---- Bookkeeping ----
    int unsigned seed = 1;  // +seed=N to reproduce a different random run
    int checks = 0;  // outputs checked so far
    int errors = 0;  // mismatching outputs
    string first = "";  // the first mismatch, for the FAIL line

    // Device under test
    argmax #(
        .Lanes     (Lanes),
        .ScoreWidth(ScoreWidth),
        .IndexWidth(IndexWidth)
    ) dut (
        .clk_i       (clk_i),
        .scores_i    (scores_i),
        .lane_valid_i(lane_valid_i),
        .base_index_i(base_index_i),
        .use_prev_i  (use_prev_i),
        .prev_score_i(prev_score_i),
        .prev_index_i(prev_index_i),
        .best_score_o(best_score_o),
        .best_index_o(best_index_o)
    );

    // svstyle: waive LNG-003 testbench clock generator, a delay loop rather than logic
    always #5 clk_i = ~clk_i;

    // Check the oldest result, drive the inputs already set up, and model them.
    // The caller sets scores_i, lane_valid_i, base_index_i, use_prev_i,
    // prev_score_i and prev_index_i before calling, right after a negedge.
    task automatic step();
        longint best_score;  // model: best score so far
        int best_index;  // model: its index
        logic found;  // model: a valid candidate has been seen

        // Model: earlier best first (lower indices), then the lanes in order
        found      = use_prev_i;
        best_score = prev_score_i;
        best_index = int'(prev_index_i);
        for (int lane = 0; lane < Lanes; lane++) begin
            if (lane_valid_i[lane]) begin
                if (!found || (longint'($signed(scores_i[lane])) > best_score)) begin
                    found      = 1'b1;
                    best_score = longint'($signed(scores_i[lane]));
                    best_index = int'(base_index_i) + lane;
                end
            end
        end
        // Nothing valid at all: the DUT result is meaningless, so do not check it
        if (!found) begin
            best_index = -1;
        end
        expected_score.push_back(best_score);
        expected_index.push_back(best_index);

        @(negedge clk_i);
        if (expected_score.size() >= Latency) begin
            best_score = expected_score.pop_front();
            best_index = expected_index.pop_front();
            if (best_index >= 0) begin
                checks++;
                if (best_score_o !== ScoreWidth'(best_score) ||
                    int'(best_index_o) != best_index) begin
                    if (errors == 0) begin
                        first = $sformatf("got %0d at %0d, want %0d at %0d", best_score_o,
                                          best_index_o, best_score, best_index);
                    end
                    errors++;
                end
            end
        end
    endtask

    // Random inputs. Narrow: scores from -2..1, so ties are common.
    task automatic randomize_inputs(input logic narrow);
        for (int lane = 0; lane < Lanes; lane++) begin
            if (narrow) begin
                scores_i[lane] = ScoreWidth'($signed($urandom_range(3, 0)) - 2);
            end else begin
                scores_i[lane] = ScoreWidth'($urandom);
            end
        end
        lane_valid_i = Lanes'($urandom);
        base_index_i = IndexWidth'($urandom_range(1, 0) * Lanes);
        use_prev_i   = 1'($urandom);
        if (narrow) begin
            prev_score_i = ScoreWidth'($signed($urandom_range(3, 0)) - 2);
        end else begin
            prev_score_i = ScoreWidth'($urandom);
        end
        // The earlier best always comes from lower indices than this block
        if (base_index_i == 0) begin
            use_prev_i = 1'b0;
        end
        prev_index_i = IndexWidth'($urandom_range(Lanes - 1, 0));
    endtask

    int unsigned seed_found;  // 1 if +seed=N was given (xsim needs the result stored)
    int unsigned first_random;  // $urandom(seed) seeds the generator; xsim needs the result stored

    initial begin
        seed_found   = $value$plusargs("seed=%d", seed);
        first_random = $urandom(seed);
        @(negedge clk_i);

        // ---- Test 1: all equal ----
        scores_i     = {Lanes{ScoreWidth'(5)}};
        lane_valid_i = '1;
        base_index_i = IndexWidth'(Lanes);
        use_prev_i   = 1'b1;
        prev_score_i = 5;
        prev_index_i = 3;
        step();  // the earlier best (index 3) must win
        use_prev_i = 1'b0;
        step();  // lane 0 (index 8) must win

        // ---- Test 2: one winner in every lane position ----
        for (int winner = 0; winner < Lanes; winner++) begin
            scores_i         = {Lanes{ScoreWidth'(-7)}};
            scores_i[winner] = ScoreWidth'(100);
            lane_valid_i     = '1;
            base_index_i     = '0;
            use_prev_i       = 1'b0;
            step();
        end

        // ---- Test 3: random with ties ----
        repeat (NumRandom) begin
            randomize_inputs(1'b1);
            step();
        end

        // ---- Test 4: random wide ----
        repeat (NumRandom) begin
            randomize_inputs(1'b0);
            step();
        end

        // Flush the pipeline so the last inputs are checked too
        lane_valid_i = '0;
        use_prev_i   = 1'b0;
        repeat (Latency) begin
            step();
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

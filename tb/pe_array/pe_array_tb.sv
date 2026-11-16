// int8_mlp: INT8 MLP accelerator on a weight-stationary systolic array (Artix-7)
//
// Description:
//   Self-checking testbench for pe_array.sv.
//   It drives and reads the array only at its edges, and checks it against a
//   plain matrix-vector model:
//     output of column n for input vector v = sum over k of x[v][k] * W[n][k]
//   The testbench does by hand what other modules will do later:
//     - loads the weights bottom row first, Rows cycles with weight_load_i high
//     - skews the inputs: row k of vector v is driven in cycle v + k
//     - de-skews the outputs: column n of vector v is read in cycle v + Rows + n
//   A tile is one weight load followed by a batch of input vectors.
//   Tests, in order:
//     1. Wiring: every weight is different and every input vector is one-hot,
//        so each output equals exactly one PE's weight. Catches swapped rows,
//        swapped columns and a reversed weight load order.
//     2. Corners: all weights and inputs at the INT8 extremes.
//     3. Random: NumTiles tiles, each with random weights and a batch of
//        1 to MaxBatch random input vectors.
//   With Rows = Cols a swap of the two would go unseen, so also run it with
//   Rows != Cols (scripts/sim.ps1 -Generics Rows=3,Cols=5).
//   Prints one line, PASS or FAIL with the first mismatch; FAIL exits non-zero.
//   +seed=N picks the random run.

`default_nettype none

module pe_array_tb #(
    parameter int unsigned Rows = 8,  // DUT rows under test
    parameter int unsigned Cols = 8   // DUT columns under test
);
    localparam int unsigned DataWidth = 8;  // DUT parameter under test
    localparam int unsigned AccWidth = 24;  // DUT parameter under test
    localparam int MaxBatch = 16;  // most input vectors in one tile
    localparam int NumTiles = 100000;  // random tiles in test 3 (roadmap exit test)

    // ---- DUT signals ----
    logic                           clk_i = 1'b0;  // 100 MHz, posedge at 5, 15, ...
    logic                           weight_load_i = 1'b0;  // DUT input, driven at negedge
    logic [Cols-1:0][DataWidth-1:0] weight_i = '0;  // DUT input, driven at negedge
    logic [Rows-1:0][DataWidth-1:0] act_i = '0;  // DUT input, driven at negedge
    logic [Cols-1:0][ AccWidth-1:0] psum_o;  // DUT output, checked at negedge

    // ---- Model: the current tile ----
    logic signed [DataWidth-1:0] weight_m[Cols][Rows];  // weight_m[n][k] = W[n, k]
    logic signed [DataWidth-1:0] act_m[MaxBatch][Rows];  // act_m[v][k] = input k of vector v
    int batch_size = 0;  // input vectors in the current tile

    // ---- Bookkeeping ----
    int unsigned seed = 1;  // +seed=N to reproduce a different random run
    int checks = 0;  // outputs checked so far
    int errors = 0;  // mismatching outputs
    string first = "";  // the first mismatch, for the FAIL line

    // Device under test
    pe_array #(
        .Rows     (Rows),
        .Cols     (Cols),
        .DataWidth(DataWidth),
        .AccWidth (AccWidth)
    ) dut (
        .clk_i        (clk_i),
        .weight_load_i(weight_load_i),
        .weight_i     (weight_i),
        .act_i        (act_i),
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

    // The model: the dot product of input vector `vector` with the weights of
    // column `col`, wrapped to AccWidth bits like the hardware.
    function automatic logic signed [AccWidth-1:0] expected_psum(input int vector, input int col);
        longint sum;  // exact sum before wrapping
        sum = 0;
        for (int row = 0; row < Rows; row++) begin
            sum += longint'(act_m[vector][row]) * longint'(weight_m[col][row]);
        end
        expected_psum = AccWidth'(sum);
    endfunction

    // Shift weight_m into the array: one row per cycle, bottom row first,
    // because the first weight pushed in is pushed furthest down.
    task automatic load_weights();
        for (int cycle = 0; cycle < Rows; cycle++) begin
            @(negedge clk_i);
            weight_load_i = 1'b1;
            for (int col = 0; col < Cols; col++) begin
                weight_i[col] = weight_m[col][Rows-1-cycle];
            end
            act_i = '0;
        end
    endtask

    // Stream the batch through the array and check every output.
    // Each cycle, at the falling edge:
    //   1. check: column n now holds vector (cycle - Rows - n), if that vector exists
    //   2. drive: row k takes vector (cycle - k), or zero outside the batch
    // The last output (last vector, last column) is checked in the last cycle.
    task automatic run_batch();
        int last_cycle;  // cycle in which the last output is checked
        int vector;  // which input vector a row or column is handling
        logic signed [AccWidth-1:0] want;  // expected output

        last_cycle = batch_size + Rows + Cols - 2;
        for (int cycle = 0; cycle <= last_cycle; cycle++) begin
            @(negedge clk_i);

            for (int col = 0; col < Cols; col++) begin
                vector = cycle - Rows - col;
                if (vector >= 0 && vector < batch_size) begin
                    checks++;
                    want = expected_psum(vector, col);
                    if (psum_o[col] !== want) begin
                        fail($sformatf("vector %0d column %0d: psum_o=%0d, want %0d", vector,
                                       col, $signed(psum_o[col]), want));
                    end
                end
            end

            weight_load_i = 1'b0;
            for (int row = 0; row < Rows; row++) begin
                vector = cycle - row;
                if (vector >= 0 && vector < batch_size) begin
                    act_i[row] = act_m[vector][row];
                end else begin
                    act_i[row] = '0;
                end
            end
        end
    endtask

    // Fill every weight and every input with the same value.
    task automatic fill_tile(input logic signed [DataWidth-1:0] weight,
                             input logic signed [DataWidth-1:0] act, input int size);
        batch_size = size;
        for (int col = 0; col < Cols; col++) begin
            for (int row = 0; row < Rows; row++) begin
                weight_m[col][row] = weight;
            end
        end
        for (int vector = 0; vector < batch_size; vector++) begin
            for (int row = 0; row < Rows; row++) begin
                act_m[vector][row] = act;
            end
        end
    endtask

    int unsigned seed_found;  // 1 if +seed=N was given (xsim needs the result stored)
    int unsigned first_random;  // $urandom(seed) seeds the generator; xsim needs the result stored

    initial begin
        seed_found   = $value$plusargs("seed=%d", seed);
        first_random = $urandom(seed);

        // ---- Test 1: wiring ----
        // W[n, k] = 16k + n + 1 is different for every PE (largest 16*7 + 7 + 1 = 120).
        // Vector v is one-hot at row v, so column n outputs W[n, v]: one PE's weight.
        batch_size = Rows;
        for (int col = 0; col < Cols; col++) begin
            for (int row = 0; row < Rows; row++) begin
                weight_m[col][row] = DataWidth'(16 * row + col + 1);
            end
        end
        for (int vector = 0; vector < batch_size; vector++) begin
            for (int row = 0; row < Rows; row++) begin
                act_m[vector][row] = (row == vector) ? 8'sd1 : 8'sd0;
            end
        end
        load_weights();
        run_batch();

        // ---- Test 2: corners ----
        fill_tile(-8'sd128, -8'sd128, 2);  // largest sum: +16,384 per row
        load_weights();
        run_batch();
        fill_tile(8'sd127, -8'sd128, 2);  // most negative sum: -16,256 per row
        load_weights();
        run_batch();
        fill_tile(-8'sd128, 8'sd127, 2);  // the same through the other operand
        load_weights();
        run_batch();

        // ---- Test 3: random tiles ----
        repeat (NumTiles) begin
            batch_size = $urandom_range(MaxBatch, 1);
            for (int col = 0; col < Cols; col++) begin
                for (int row = 0; row < Rows; row++) begin
                    weight_m[col][row] = DataWidth'($urandom);
                end
            end
            for (int vector = 0; vector < batch_size; vector++) begin
                for (int row = 0; row < Rows; row++) begin
                    act_m[vector][row] = DataWidth'($urandom);
                end
            end
            load_weights();
            run_batch();
        end

        if (errors != 0) begin
            $fatal(1, "FAIL: %0d of %0d checks, Rows %0d, Cols %0d, seed %0d, first at %s", errors,
                   checks, Rows, Cols, seed, first);
        end
        $display("PASS: %0d checks, Rows %0d, Cols %0d, seed %0d", checks, Rows, Cols, seed);
        $finish;
    end
endmodule

`resetall

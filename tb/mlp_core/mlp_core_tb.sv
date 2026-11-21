// int8_mlp: INT8 MLP accelerator on a weight-stationary systolic array (Artix-7)
//
// Description:
//   Full-model differential testbench for mlp_core.sv: every test image
//   through the RTL, every intermediate value compared with the integer
//   golden model (model/golden.py, via model/gen_vectors.py).
//   For each batch size in BatchSizes it runs the whole test set and checks:
//     - every hidden block written (32 values per image) = expected_hidden
//     - every final layer 2 block (10 scores per image, 6 padding) = expected_scores
//     - every image's hidden blocks, score blocks and prediction are each
//       produced exactly once: no tile skipped or used twice
//     - every weight word is read the same number of times (once per batch)
//     - the prediction of every image, read through result_image_i = expected_pred
//     - correct_count_o and checksum_o = expected_summary
//   and prints the cycle count and array utilisation of each run:
//     utilisation = useful MACs / (64 MACs per cycle * cycles)
//   Run the .mem files through scripts\sim.ps1, which copies model\data\*.mem
//   next to the simulation (generate them with model\gen_vectors.py).
//   OnlyBatch = N (scripts\sim.ps1 -Generics OnlyBatch=N) runs only batch size N.
//   Prints one PASS or FAIL line at the end; FAIL exits non-zero.

`default_nettype none

module mlp_core_tb
    import accel_pkg::*;
#(
    parameter int OnlyBatch = 0  // 0: run every size in batch_sizes; N: run only batch size N
);
    localparam int UsefulMacs = NumInputs * NumHidden + NumHidden * NumOutputs;  // 2,368 per image
    localparam int TimeoutCycles = 2000000;  // per run; B = 1 needs about 400,000

    // ---- DUT signals ----
    logic                  clk_i = 1'b0;  // 100 MHz, posedge at 5, 15, ...
    logic                  rst_ni = 1'b0;  // DUT reset, released after a few cycles
    logic                  start_i = 1'b0;  // DUT input, driven at negedge
    logic [BatchWidth-1:0] batch_size_i = '0;  // DUT input, driven at negedge
    logic                  busy_o;  // DUT output
    logic                  done_o;  // DUT output, sampled at negedge
    logic [          31:0] cycle_count_o;  // DUT output
    logic [          15:0] correct_count_o;  // DUT output
    logic [          31:0] checksum_o;  // DUT output
    logic [ImageWidth-1:0] result_image_i = '0;  // DUT input, driven at negedge
    logic [DigitWidth-1:0] result_digit_o;  // DUT output, one cycle after result_image_i

    // Device under test
    mlp_core dut (
        .clk_i          (clk_i),
        .rst_ni         (rst_ni),
        .start_i        (start_i),
        .batch_size_i   (batch_size_i),
        .busy_o         (busy_o),
        .done_o         (done_o),
        .cycle_count_o  (cycle_count_o),
        .correct_count_o(correct_count_o),
        .checksum_o     (checksum_o),
        .result_image_i (result_image_i),
        .result_digit_o (result_digit_o)
    );

    // svstyle: waive LNG-003 testbench clock generator, a delay loop rather than logic
    always #5 clk_i = ~clk_i;

    // ---- Expected results from the golden model ----
    hidden_vector_t                   expected_hidden [NumImages * L1ColBlocks];  // image*4 + block
    acc_vector_t                      expected_scores [NumImages * L2ColBlocks];  // image*2 + block
    logic           [DigitWidth-1:0] expected_pred   [NumImages];  // predicted digit per image
    logic           [          31:0] expected_summary[2];  // correct count, checksum

    // ---- Bookkeeping ----
    int checks = 0;  // values compared so far
    int errors = 0;  // mismatches
    string first = "";  // the first mismatch, for the FAIL line
    int hidden_writes[NumImages * L1ColBlocks];  // writes per hidden block in this run
    int score_blocks[NumImages * L2ColBlocks];  // final score blocks seen in this run
    int pred_writes[NumImages];  // predictions written per image in this run
    int weight_reads[WeightDepth];  // reads per weight word in this run

    task automatic fail(input string msg);
        if (errors == 0) begin
            first = msg;
        end
        errors++;
    endtask

    // Watch the hidden memory writes (end of layer 1 for one neuron block).
    // svstyle: waive LNG-002 testbench monitor: samples the DUT at the falling edge
    always @(negedge clk_i) begin
        int image;  // test image of the write
        int block;  // neuron block of the write
        if (rst_ni && dut.hidden_write) begin
            image = int'(dut.batch_start) + int'(dut.hidden_write_addr) / L1ColBlocks;
            block = int'(dut.hidden_write_addr) % L1ColBlocks;
            checks++;
            hidden_writes[image*L1ColBlocks+block]++;
            if (dut.hidden_write_data !== expected_hidden[image*L1ColBlocks+block]) begin
                fail($sformatf("hidden image %0d block %0d: got %h, want %h", image, block,
                               dut.hidden_write_data, expected_hidden[image*L1ColBlocks+block]));
            end
        end
    end

    // Watch the final layer 2 sums (scores incl. bias), at stage P1.
    // svstyle: waive LNG-002 testbench monitor: samples the DUT at the falling edge
    always @(negedge clk_i) begin
        int image;  // test image of the scores
        int block;  // neuron block of the scores
        if (rst_ni && dut.p1_valid_q && dut.p1_tag_q.last && dut.p1_tag_q.layer) begin
            image = int'(dut.batch_start) + int'(dut.p1_tag_q.image);
            block = int'(dut.p1_tag_q.block);
            checks++;
            score_blocks[image*L2ColBlocks+block]++;
            if (dut.p1_acc_q !== expected_scores[image*L2ColBlocks+block]) begin
                fail($sformatf("scores image %0d block %0d: got %h, want %h", image, block,
                               dut.p1_acc_q, expected_scores[image*L2ColBlocks+block]));
            end
        end
    end

    // Count prediction writes and weight reads.
    // svstyle: waive LNG-002 testbench monitor: samples the DUT at the falling edge
    always @(negedge clk_i) begin
        if (rst_ni && dut.image_done) begin
            pred_writes[int'(dut.pred_image)]++;
        end
        if (rst_ni && dut.weight_read) begin
            weight_reads[int'(dut.weight_addr)]++;
        end
    end

    // Run the whole test set with one batch size and check everything.
    task automatic run(input int batch_size);
        int num_batches;  // batches in this run
        int cycles;  // cycles the run took
        real utilisation;  // useful MACs / peak MACs over the run

        foreach (hidden_writes[i]) hidden_writes[i] = 0;
        foreach (score_blocks[i]) score_blocks[i] = 0;
        foreach (pred_writes[i]) pred_writes[i] = 0;
        foreach (weight_reads[i]) weight_reads[i] = 0;
        num_batches = (NumImages + batch_size - 1) / batch_size;

        // Start
        @(negedge clk_i);
        batch_size_i = BatchWidth'(batch_size);
        start_i      = 1'b1;
        @(negedge clk_i);
        start_i = 1'b0;

        // Wait for done
        cycles = 0;
        while (!done_o && cycles < TimeoutCycles) begin
            @(negedge clk_i);
            cycles++;
        end
        if (!done_o) begin
            fail($sformatf("batch %0d: timed out after %0d cycles", batch_size, cycles));
            return;
        end

        // Every block produced exactly once, every weight word read once per batch
        foreach (hidden_writes[i]) begin
            checks++;
            if (hidden_writes[i] != 1) begin
                fail($sformatf("batch %0d: hidden block %0d written %0d times", batch_size, i,
                               hidden_writes[i]));
            end
        end
        foreach (score_blocks[i]) begin
            checks++;
            if (score_blocks[i] != 1) begin
                fail($sformatf("batch %0d: score block %0d seen %0d times", batch_size, i,
                               score_blocks[i]));
            end
        end
        foreach (pred_writes[i]) begin
            checks++;
            if (pred_writes[i] != 1) begin
                fail($sformatf("batch %0d: prediction %0d written %0d times", batch_size, i,
                               pred_writes[i]));
            end
        end
        foreach (weight_reads[i]) begin
            checks++;
            if (weight_reads[i] != num_batches) begin
                fail($sformatf("batch %0d: weight word %0d read %0d times, want %0d", batch_size,
                               i, weight_reads[i], num_batches));
            end
        end

        // Predictions, read back through the result port like the host does
        for (int image = 0; image < NumImages; image++) begin
            @(negedge clk_i);
            result_image_i = ImageWidth'(image);
            @(negedge clk_i);
            checks++;
            if (result_digit_o !== expected_pred[image]) begin
                fail($sformatf("batch %0d: image %0d predicted %0d, want %0d", batch_size, image,
                               result_digit_o, expected_pred[image]));
            end
        end

        // Summary
        checks += 2;
        if (32'(correct_count_o) !== expected_summary[0]) begin
            fail($sformatf("batch %0d: correct count %0d, want %0d", batch_size, correct_count_o,
                           expected_summary[0]));
        end
        if (checksum_o !== expected_summary[1]) begin
            fail($sformatf("batch %0d: checksum %h, want %h", batch_size, checksum_o,
                           expected_summary[1]));
        end

        utilisation = 100.0 * real'(UsefulMacs) * real'(NumImages) /
            (real'(Rows * Cols) * real'(cycle_count_o));
        $display("batch %3d: %0d cycles, %0d correct, utilisation %.1f %%", batch_size,
                 cycle_count_o, correct_count_o, utilisation);
    endtask

    int batch_sizes[5] = '{NumImages, 64, 8, 7, 1};  // 7 leaves a short last batch

    initial begin
        $readmemh("expected_hidden.mem", expected_hidden);
        $readmemh("expected_scores.mem", expected_scores);
        $readmemh("expected_pred.mem", expected_pred);
        $readmemh("expected_summary.mem", expected_summary);

        repeat (3) @(negedge clk_i);
        rst_ni = 1'b1;

        if (OnlyBatch != 0) begin
            run(OnlyBatch);
        end else begin
            foreach (batch_sizes[i]) begin
                run(batch_sizes[i]);
            end
        end

        if (errors != 0) begin
            $fatal(1, "FAIL: %0d of %0d checks, first: %s", errors, checks, first);
        end
        $display("PASS: %0d checks, %0d images, %0d of %0d correct, checksum %h", checks,
                 NumImages, correct_count_o, NumImages, checksum_o);
        $finish;
    end
endmodule

`resetall

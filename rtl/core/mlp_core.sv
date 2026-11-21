// int8_mlp: INT8 MLP accelerator on a weight-stationary systolic array (Artix-7)
//
// Description:
//   The whole accelerator without its host link: runs the 64 -> 32 -> 10
//   MLP on every test image held in on-chip memory and keeps the predicted
//   digit of each, the number of correct predictions, a checksum of all
//   scores and the run's cycle count. Bit-exact with model/golden.py.
//   Pipeline, cycle by cycle after the controller issues a read in cycle c:
//     c+1           memories give the weight word or input vector
//     c+2           layer select and a register (feed: FeedLatency = 2)
//     c+2 .. c+17   systolic_core: aligned in, aligned out CoreLatency later
//     c+17    (P0)  add the psum to the running sum, or to the bias on the
//                   first input block (acc, bias and best memories were
//                   read in c+16 using the tag one stage earlier)
//     c+18    (P1)  store the running sum; on the last input block the sum
//                   is final and enters requant (layer 1) or argmax (layer 2)
//     c+22    (PD)  write the hidden values, or the best digit so far; on
//                   the last neuron block of layer 2, the prediction.
//                   PD = P1 + ArgmaxLatency; requant is faster, so its
//                   results wait in a short delay line.
//   The feed register and the argmax pipeline exist for timing at 150 MHz:
//   without them the BRAM-to-PE path and the argmax compares failed.
//   A tag (image in batch, blocks, layer) travels beside each vector in a
//   shift register, so the output side knows what every result belongs to.
//   Memories (all sdp_ram, layouts in model/gen_vectors.py):
//     weights, images, labels, biases, multipliers: ROMs loaded from .mem files
//     acc     running sums, one word of Cols sums per image in the batch
//     hidden  layer 1 results, read back as layer 2 inputs
//     best    best score and digit so far per image (after layer 2 block 0)
//     pred    predicted digit per image, read by the host after the run
//   A new start clears the correct count and the checksum.

// svstyle: waive LNG-014 Vivado xsim rejects `input logic` ports under `default_nettype none
`include "assert_macros.svh"

module mlp_core
    import accel_pkg::*;
#(
    parameter string WeightFile = "weights.mem",  // Weight tiles, in load order
    parameter string ImageFile  = "images.mem",   // Test images, one row block per word
    parameter string LabelFile  = "labels.mem",   // True digit of each test image
    parameter string BiasFile   = "bias.mem",     // Biases, one neuron block per word
    parameter string MultFile   = "mult.mem"      // Requantisation multipliers, one block per word
) (
    input  logic                  clk_i,            // System clock
    input  logic                  rst_ni,           // Asynchronous reset, active-low
    // ---- Run control ----
    input  logic                  start_i,          // Start a run over all test images
    input  logic [BatchWidth-1:0] batch_size_i,     // Images per batch, sampled at start
    output logic                  busy_o,           // A run is in progress
    output logic                  done_o,           // One-cycle pulse: results are ready
    // ---- Results ----
    output logic [          31:0] cycle_count_o,    // Cycles the last run took
    output logic [          15:0] correct_count_o,  // Predictions equal to the label
    output logic [          31:0] checksum_o,       // Sum of all scores, modulo 2^32
    input  logic [ImageWidth-1:0] result_image_i,   // Image whose prediction to read
    output logic [DigitWidth-1:0] result_digit_o    // Its digit, one cycle later
);
    localparam int unsigned FeedLatency = 2;  // controller read to core input
    localparam int unsigned CoreLatency = Rows + Cols - 1;  // systolic_core: act_i to psum_o
    localparam int unsigned TagStages = FeedLatency + CoreLatency;  // controller read to P0
    localparam int unsigned RequantLatency = 2;  // requant: acc_i to value_o
    localparam int unsigned ArgmaxLatency = $clog2(Cols) + 1;  // argmax: inputs to outputs

    // The requant results wait for the argmax, so the argmax must be the slower one.
    if (ArgmaxLatency <= RequantLatency) begin : gen_latency_check
        $error("ArgmaxLatency must exceed RequantLatency");
    end
    localparam int unsigned AccDepth = MaxBatch;  // One running-sum word per image in a batch
    localparam int unsigned BestWidth = AccWidth + DigitWidth;  // Best score and its digit

    // ---- Controller ----
    logic                       pipe_busy;  // Vectors in flight after the controller
    logic                       weight_read;  // Controller reads a weight word
    logic [WeightAddrWidth-1:0] weight_addr;  // Which weight word
    logic                       act_read;  // Controller reads an input vector
    logic [ ImageAddrWidth-1:0] image_addr;  // Image word, layer 1
    logic [HiddenAddrWidth-1:0] hidden_read_addr;  // Hidden word, layer 2
    logic                       issue_layer;  // Tag: layer of the vector
    logic [BatchIndexWidth-1:0] issue_image;  // Tag: image in the batch
    logic [     BlockWidth-1:0] issue_block;  // Tag: neuron block
    logic                       issue_first;  // Tag: first input block
    logic                       issue_last;  // Tag: last input block
    logic [     ImageWidth-1:0] batch_start;  // First image of the current batch

    // Tiling controller: issues one weight or input read per cycle.
    controller u_ctrl (
        .clk_i        (clk_i),
        .rst_ni       (rst_ni),
        .start_i      (start_i),
        .batch_size_i (batch_size_i),
        .busy_o       (busy_o),
        .done_o       (done_o),
        .cycle_count_o(cycle_count_o),
        .pipe_busy_i  (pipe_busy),
        .weight_read_o(weight_read),
        .weight_addr_o(weight_addr),
        .act_read_o   (act_read),
        .image_addr_o (image_addr),
        .hidden_addr_o(hidden_read_addr),
        .layer_o      (issue_layer),
        .batch_index_o(issue_image),
        .block_o      (issue_block),
        .first_o      (issue_first),
        .last_o       (issue_last),
        .batch_start_o(batch_start)
    );

    // ---- Feed: read the memories, one cycle ----
    weight_vector_t weight_word;  // Weight word read last cycle
    act_vector_t    image_word;  // Image row block read last cycle
    act_vector_t    hidden_word;  // Hidden block read last cycle (layer 2 input)

    // Weight tiles, in load order.
    sdp_ram #(
        .Width   ($bits(weight_vector_t)),
        .Depth   (WeightDepth),
        .InitFile(WeightFile)
    ) u_weight_rom (
        .clk_i       (clk_i),
        .write_en_i  (1'b0),
        .write_addr_i('0),
        .write_data_i('0),
        .read_addr_i (weight_addr),
        .read_data_o (weight_word)
    );

    // Test images: the layer 1 inputs.
    sdp_ram #(
        .Width   ($bits(act_vector_t)),
        .Depth   (ImageDepth),
        .InitFile(ImageFile)
    ) u_image_rom (
        .clk_i       (clk_i),
        .write_en_i  (1'b0),
        .write_addr_i('0),
        .write_data_i('0),
        .read_addr_i (image_addr),
        .read_data_o (image_word)
    );

    logic                                 hidden_write;  // Write a block of hidden values (PD)
    logic           [HiddenAddrWidth-1:0] hidden_write_addr;  // Image in batch and neuron block
    hidden_vector_t                       hidden_write_data;  // Requantised values

    // Hidden values: written by layer 1, read as the layer 2 inputs.
    sdp_ram #(
        .Width   ($bits(hidden_vector_t)),
        .Depth   (HiddenDepth),
        .InitFile("")
    ) u_hidden_ram (
        .clk_i       (clk_i),
        .write_en_i  (hidden_write),
        .write_addr_i(hidden_write_addr),
        .write_data_i(hidden_write_data),
        .read_addr_i (hidden_read_addr),
        .read_data_o (hidden_word)
    );

    logic           read_weight_load_q;  // weight_read, one cycle on: beside weight_word
    logic           read_layer_q;  // issue_layer, one cycle on: beside the input words
    logic           core_weight_load_q;  // Load strobe into the core
    weight_vector_t core_weight_q;  // Weight word into the core
    act_vector_t    core_act_q;  // Input vector into the core: image or hidden word

    // Delay the load strobe (control, reset) beside the memory read and the feed register.
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            read_weight_load_q <= 1'b0;
            core_weight_load_q <= 1'b0;
        end else begin
            read_weight_load_q <= weight_read;
            core_weight_load_q <= read_weight_load_q;
        end
    end

    // Feed register: pick this layer's input, so the BRAM output does not
    // drive the multipliers directly.
    always_ff @(posedge clk_i) begin
        read_layer_q    <= issue_layer;
        core_weight_q <= weight_word;
        core_act_q    <= read_layer_q ? hidden_word : image_word;
    end

    // ---- The array ----
    acc_vector_t core_psum;  // Dot products, CoreLatency cycles after core_act_q

    // Systolic array with skew and de-skew: aligned vectors in and out.
    systolic_core #(
        .Rows     (Rows),
        .Cols     (Cols),
        .DataWidth(DataWidth),
        .AccWidth (AccWidth)
    ) u_core (
        .clk_i        (clk_i),
        .weight_load_i(core_weight_load_q),
        .weight_i     (core_weight_q),
        .act_i        (core_act_q),
        .psum_o       (core_psum)
    );

    // ---- Tag line: what each vector in the pipeline belongs to ----
    typedef struct packed {
        logic                       layer;  // 0: layer 1, 1: layer 2
        logic [BatchIndexWidth-1:0] image;  // Image within the batch
        logic [BlockWidth-1:0]      block;  // Neuron block within the layer
        logic                       first;  // First input block
        logic                       last;   // Last input block
    } tag_t;

    tag_t issue_tag;  // The controller's tag for this cycle's read
    assign issue_tag = '{
            layer: issue_layer,
            image: issue_image,
            block: issue_block,
            first: issue_first,
            last: issue_last
        };

    // Stage s holds the tag of the vector issued s+1 cycles ago: stage
    // TagStages-1 lines up with core_psum, and stage TagStages-2 is one cycle
    // earlier, in time to read the memories the output side needs.
    logic [TagStages-1:0] tag_valid_q;  // A real vector is at this stage
    tag_t                 tag_q                                          [TagStages];  // Its tag

    // Shift the valid bits (control, reset).
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            tag_valid_q <= '0;
        end else begin
            tag_valid_q <= {tag_valid_q[TagStages-2:0], act_read};
        end
    end

    // Shift the tags (datapath).
    always_ff @(posedge clk_i) begin
        tag_q[0] <= issue_tag;
        for (int stage = 1; stage < TagStages; stage++) begin
            tag_q[stage] <= tag_q[stage-1];
        end
    end

    // Only the image, block and layer fields address memories; first and last are
    // used later, from the tag line itself.
    /* verilator lint_off UNUSEDSIGNAL */
    tag_t read_tag;  // One cycle before the result: read acc, bias, best, label
    /* verilator lint_on UNUSEDSIGNAL */
    assign read_tag = tag_q[TagStages-2];

    // ---- P0: add to the running sum ----
    acc_vector_t                   acc_word;  // Running sums of this image, read last cycle
    acc_vector_t                   bias_word;  // Biases of this neuron block, read last cycle
    mult_vector_t                  mult_word;  // Multipliers of this neuron block, read last cycle
    logic         [ BestWidth-1:0] best_word;  // Best score and digit so far, read last cycle
    logic         [DigitWidth-1:0] label_word;  // True digit of this image, read last cycle

    logic                          p1_acc_write;  // Store p1_acc_q as the running sum
    tag_t                          p1_tag_q;  // Tag of the P1 stage
    acc_vector_t                   p1_acc_q;  // Running sum including this tile

    // Running sums: one word per image in the batch.
    sdp_ram #(
        .Width   ($bits(acc_vector_t)),
        .Depth   (AccDepth),
        .InitFile("")
    ) u_acc_ram (
        .clk_i       (clk_i),
        .write_en_i  (p1_acc_write),
        .write_addr_i(p1_tag_q.image),
        .write_data_i(p1_acc_q),
        .read_addr_i (read_tag.image),
        .read_data_o (acc_word)
    );

    // Bias word: layer 1 blocks are 0..3, layer 2 blocks follow.
    logic [BlockWidth-1:0] bias_addr;  // Bias word of the vector one cycle before P0
    assign
        bias_addr = read_tag.layer ? (BlockWidth'(L1ColBlocks) + read_tag.block) : read_tag.block;

    // Biases, one neuron block per word.
    sdp_ram #(
        .Width   ($bits(acc_vector_t)),
        .Depth   (NumBlocks),
        .InitFile(BiasFile)
    ) u_bias_rom (
        .clk_i       (clk_i),
        .write_en_i  (1'b0),
        .write_addr_i('0),
        .write_data_i('0),
        .read_addr_i (bias_addr),
        .read_data_o (bias_word)
    );

    // Requantisation multipliers of layer 1, one neuron block per word.
    sdp_ram #(
        .Width   ($bits(mult_vector_t)),
        .Depth   (L1ColBlocks),
        .InitFile(MultFile)
    ) u_mult_rom (
        .clk_i       (clk_i),
        .write_en_i  (1'b0),
        .write_addr_i('0),
        .write_data_i('0),
        .read_addr_i (read_tag.block[$clog2(L1ColBlocks)-1:0]),
        .read_data_o (mult_word)
    );

    logic                       best_write;  // Store the best score and digit so far (PD)
    logic [BatchIndexWidth-1:0] best_write_image;  // Which image
    logic [      BestWidth-1:0] best_write_data;  // {score, digit}

    // Best score and digit so far, per image, between the two layer 2 blocks.
    sdp_ram #(
        .Width   (BestWidth),
        .Depth   (MaxBatch),
        .InitFile("")
    ) u_best_ram (
        .clk_i       (clk_i),
        .write_en_i  (best_write),
        .write_addr_i(best_write_image),
        .write_data_i(best_write_data),
        .read_addr_i (read_tag.image),
        .read_data_o (best_word)
    );

    logic [ImageWidth-1:0] label_addr;  // Test image of the vector one cycle before P0
    assign label_addr = batch_start + ImageWidth'(read_tag.image);

    // True digit of each test image, to count correct predictions.
    sdp_ram #(
        .Width   (DigitWidth),
        .Depth   (NumImages),
        .InitFile(LabelFile)
    ) u_label_rom (
        .clk_i       (clk_i),
        .write_en_i  (1'b0),
        .write_addr_i('0),
        .write_data_i('0),
        .read_addr_i (label_addr[$clog2(NumImages)-1:0]),
        .read_data_o (label_word)
    );

    logic p0_valid;  // A real result leaves the core this cycle
    tag_t p0_tag;  // Its tag
    assign p0_valid = tag_valid_q[TagStages-1];
    assign p0_tag   = tag_q[TagStages-1];

    acc_vector_t p0_sum;  // Running sum after this tile

    // On the first input block start from the bias, otherwise from the stored sum.
    always_comb begin
        for (int col = 0; col < Cols; col++) begin
            if (p0_tag.first) begin
                p0_sum[col] = bias_word[col] + core_psum[col];
            end else begin
                p0_sum[col] = acc_word[col] + core_psum[col];
            end
        end
    end

    // ---- P1: store the sum; a final sum enters requant or argmax ----
    logic                          p1_valid_q;  // P1 holds a real result
    mult_vector_t                  p1_mult_q;  // Multipliers, beside p1_acc_q
    logic         [ BestWidth-1:0] p1_best_q;  // Best so far, beside p1_acc_q
    logic         [DigitWidth-1:0] p1_label_q;  // True digit, beside p1_acc_q

    // P1 valid bit (control, reset).
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            p1_valid_q <= 1'b0;
        end else begin
            p1_valid_q <= p0_valid;
        end
    end

    // P1 data (datapath).
    always_ff @(posedge clk_i) begin
        p1_tag_q   <= p0_tag;
        p1_acc_q   <= p0_sum;
        p1_mult_q  <= mult_word;
        p1_best_q  <= best_word;
        p1_label_q <= label_word;
    end

    // Store every partial sum; a final one is never read back, so skip it.
    assign p1_acc_write = p1_valid_q && !p1_tag_q.last;

    // Layer 1: requantise each column's final sum into a hidden value.
    hidden_vector_t requant_hidden;  // Requantised values, RequantLatency cycles after P1
    for (genvar col = 0; col < Cols; col++) begin : gen_requant
        // One requantiser per column: multiply by m, round, shift, clamp.
        requant #(
            .AccWidth (AccWidth),
            .MultWidth(MultWidth),
            .Shift    (Shift),
            .OutWidth (DataWidth)
        ) u_requant (
            .clk_i  (clk_i),
            .acc_i  (p1_acc_q[col]),
            .mult_i (p1_mult_q[col]),
            .value_o(requant_hidden[col])
        );
    end

    // Layer 2: a lane is a real digit only below NumOutputs (the rest is padding).
    logic [Cols-1:0] score_valid;  // Lane holds a real digit's score
    for (genvar col = 0; col < Cols; col++) begin : gen_score_valid
        assign score_valid[col] = (32'(p1_tag_q.block) * Cols + col) < NumOutputs;
    end

    logic signed [  AccWidth-1:0] best_score;  // Best score so far, at stage PD
    logic        [DigitWidth-1:0] best_digit;  // Its digit

    // Find the best digit of this block and merge it with the earlier blocks.
    argmax #(
        .Lanes     (Cols),
        .ScoreWidth(AccWidth),
        .IndexWidth(DigitWidth)
    ) u_argmax (
        .clk_i       (clk_i),
        .scores_i    (p1_acc_q),
        .lane_valid_i(score_valid),
        .base_index_i(DigitWidth'(32'(p1_tag_q.block) * Cols)),
        .use_prev_i  (p1_tag_q.block != '0),
        .prev_score_i(p1_best_q[BestWidth-1:DigitWidth]),
        .prev_index_i(p1_best_q[DigitWidth-1:0]),
        .best_score_o(best_score),
        .best_index_o(best_digit)
    );

    logic [31:0] score_sum;  // Sign-extended sum of the valid lanes

    // Sum of the real scores in this block, for the checksum.
    always_comb begin
        score_sum = '0;
        for (int col = 0; col < Cols; col++) begin
            if (score_valid[col]) begin
                score_sum = score_sum + 32'($signed(p1_acc_q[col]));
            end
        end
    end

    // ---- P2 .. PD: tags beside the requant and argmax pipelines ----
    // post_*_q[s] belongs to stage P(2+s); the last entry is stage PD.
    logic [ArgmaxLatency-1:0] post_valid_q;  // Stage holds a real result
    tag_t post_tag_q[ArgmaxLatency];  // Its tag
    logic [DigitWidth-1:0] post_label_q[ArgmaxLatency];  // True digit
    logic [31:0] post_score_sum_q[ArgmaxLatency];  // Checksum contribution

    // Post-processing valid bits (control, reset).
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            post_valid_q <= '0;
        end else begin
            post_valid_q <= {post_valid_q[ArgmaxLatency-2:0], p1_valid_q};
        end
    end

    // Post-processing data (datapath).
    always_ff @(posedge clk_i) begin
        post_tag_q[0]       <= p1_tag_q;
        post_label_q[0]     <= p1_label_q;
        post_score_sum_q[0] <= score_sum;
        for (int stage = 1; stage < ArgmaxLatency; stage++) begin
            post_tag_q[stage]       <= post_tag_q[stage-1];
            post_label_q[stage]     <= post_label_q[stage-1];
            post_score_sum_q[stage] <= post_score_sum_q[stage-1];
        end
    end

    // The requant results are ready RequantLatency cycles after P1: hold them until PD.
    localparam int unsigned HiddenDelay = ArgmaxLatency - RequantLatency;  // 2 cycles
    hidden_vector_t hidden_delay_q[HiddenDelay];  // Requantised values on their way to PD

    // Delay line for the requantised values.
    always_ff @(posedge clk_i) begin
        hidden_delay_q[0] <= requant_hidden;
        for (int stage = 1; stage < HiddenDelay; stage++) begin
            hidden_delay_q[stage] <= hidden_delay_q[stage-1];
        end
    end

    // ---- PD: write the results ----
    logic                  done_valid;  // PD holds a real result
    // done_tag.first is not needed once the sum is final.
    /* verilator lint_off UNUSEDSIGNAL */
    tag_t                  done_tag;  // Its tag
    /* verilator lint_on UNUSEDSIGNAL */
    logic [DigitWidth-1:0] done_label;  // Its true digit
    logic [          31:0] done_score_sum;  // Its checksum contribution
    assign done_valid     = post_valid_q[ArgmaxLatency-1];
    assign done_tag       = post_tag_q[ArgmaxLatency-1];
    assign done_label     = post_label_q[ArgmaxLatency-1];
    assign done_score_sum = post_score_sum_q[ArgmaxLatency-1];

    logic layer1_done;  // A block of final layer 1 sums, requantised
    logic layer2_done;  // A block of final layer 2 scores, argmaxed
    logic image_done;  // The last layer 2 block: the prediction is final
    assign layer1_done = done_valid && done_tag.last && !done_tag.layer;
    assign layer2_done = done_valid && done_tag.last && done_tag.layer;
    assign image_done = layer2_done && (done_tag.block == BlockWidth'(L2ColBlocks - 1));

    assign hidden_write = layer1_done;
    assign hidden_write_addr = HiddenAddrWidth'(done_tag.image) * HiddenAddrWidth'(L1ColBlocks) +
        HiddenAddrWidth'(done_tag.block);
    assign hidden_write_data = hidden_delay_q[HiddenDelay-1];

    assign best_write = layer2_done;
    assign best_write_image = done_tag.image;
    assign best_write_data = {best_score, best_digit};

    logic [ImageWidth-1:0] pred_image;  // Test image whose prediction is final
    assign pred_image = batch_start + ImageWidth'(done_tag.image);

    // Predicted digit of every test image, read by the host.
    sdp_ram #(
        .Width   (DigitWidth),
        .Depth   (NumImages),
        .InitFile("")
    ) u_pred_ram (
        .clk_i       (clk_i),
        .write_en_i  (image_done),
        .write_addr_i(pred_image[$clog2(NumImages)-1:0]),
        .write_data_i(best_digit),
        .read_addr_i (result_image_i[$clog2(NumImages)-1:0]),
        .read_data_o (result_digit_o)
    );

    logic [15:0] correct_count_q;  // Predictions that equal the label
    logic [31:0] checksum_q;  // Sum of every real score, modulo 2^32

    // Result counters: cleared by a new start, updated by finished images.
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            correct_count_q <= '0;
            checksum_q      <= '0;
        end else if (start_i && !busy_o) begin
            correct_count_q <= '0;
            checksum_q      <= '0;
        end else begin
            if (image_done && (best_digit == done_label)) begin
                correct_count_q <= correct_count_q + 16'd1;
            end
            if (layer2_done) begin
                checksum_q <= checksum_q + done_score_sum;
            end
        end
    end

    assign correct_count_o = correct_count_q;
    assign checksum_o      = checksum_q;

    // The controller may finish a layer only when nothing is left in flight.
    assign pipe_busy       = act_read || (|tag_valid_q) || p1_valid_q || (|post_valid_q);

    // ---- Assertions ----
    // A finished image always comes from inside the test set.
    `ASSERT(PredInRange_A, image_done |-> pred_image < ImageWidth'(NumImages))
    // A weight load never reaches the array while a real vector still needs
    // a weight. The vector at tag stage FeedLatency-1 is at the core input now;
    // one at stage FeedLatency-1+k entered k cycles ago, and uses its last PE
    // at k = Rows + Cols - 2, the same edge as the load (old weight, still fine).
    `ASSERT(NoLoadInFlight_A,
            core_weight_load_q |-> !(|tag_valid_q[FeedLatency+Rows+Cols-4:FeedLatency-1]))
    // Control outputs are never X once out of reset (VER-002).
    `ASSERT_KNOWN(PipeBusyKnown_A, pipe_busy)
    `ASSERT_KNOWN(CountsKnown_A, {correct_count_o, checksum_o})
endmodule

`resetall

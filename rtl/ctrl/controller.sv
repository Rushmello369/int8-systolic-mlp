// int8_mlp: INT8 MLP accelerator on a weight-stationary systolic array (Artix-7)
//
// Description:
//   Tiling controller. Walks the test set through both layers, one batch of
//   images at a time, and issues one memory read per cycle: a weight word
//   while loading a tile, or an input vector while streaming one.
//   Order, for each batch:
//     layer 1: for each neuron block nb (4), for each input block kb (8):
//     layer 2: for each neuron block nb (2), for each input block kb (4):
//       load the tile  (Rows cycles, weight words bottom row first)
//       stream the batch (one input vector per cycle)
//       drain          (DrainCycles idle cycles)
//     then wait until the post-processing pipeline is empty, so layer 2
//     reads only finished hidden values and the next batch starts clean.
//   kb is the inner loop, so the partial sums of one neuron block are
//   finished (bias added, requantised or argmaxed) before the next block.
//   Each streamed vector carries a tag (image in batch, block, first and
//   last input block, layer) that travels with it to the output side.
//   Why DrainCycles = Rows + Cols - 3: a weight load changes every PE's
//   weight at once, so it must wait until the last vector of the tile has
//   passed PE[Rows-1][Cols-1]. That vector reaches the array's input
//   Rows + Cols - 2 cycles before it reaches that PE; the first load word
//   reaches the array DrainCycles + 1 cycles after the last vector does.
//   Every issue output is registered (one cycle after the state that made it).
//   batch_size_i of 0 or above MaxBatch is treated as MaxBatch; the last
//   batch is shorter when NumImages is not a multiple of the batch size.

// svstyle: waive LNG-014 Vivado xsim rejects `input logic` ports under `default_nettype none
`include "assert_macros.svh"

module controller
    import accel_pkg::*;
(
    input  logic                       clk_i,          // System clock
    input  logic                       rst_ni,         // Asynchronous reset, active-low
    // ---- Host side ----
    input  logic                       start_i,        // Start a run; ignored while busy
    input  logic [     BatchWidth-1:0] batch_size_i,   // Images per batch, sampled at start
    output logic                       busy_o,         // A run is in progress
    output logic                       done_o,         // One-cycle pulse: the run has finished
    output logic [               31:0] cycle_count_o,  // Cycles the last run took
    // ---- Pipeline status ----
    input  logic                       pipe_busy_i,    // Vectors still in flight after the issue
    // ---- Weight read, while loading a tile ----
    output logic                       weight_read_o,  // A weight word is read this cycle
    output logic [WeightAddrWidth-1:0] weight_addr_o,  // Which word
    // ---- Input vector read, while streaming, and its tag ----
    output logic                       act_read_o,     // An input vector is read this cycle
    output logic [ ImageAddrWidth-1:0] image_addr_o,   // Image memory word (layer 1)
    output logic [HiddenAddrWidth-1:0] hidden_addr_o,  // Hidden memory word (layer 2)
    output logic                       layer_o,        // 0: layer 1, 1: layer 2
    output logic [BatchIndexWidth-1:0] batch_index_o,  // Image within the batch
    output logic [     BlockWidth-1:0] block_o,        // Neuron block within the layer
    output logic                       first_o,        // First input block: start from the bias
    output logic                       last_o,         // Last input block: the sum is complete
    output logic [     ImageWidth-1:0] batch_start_o   // First image of the batch
);
    localparam int unsigned DrainCycles = Rows + Cols - 3;  // Idle cycles between tiles
    localparam int unsigned CountWidth = BatchWidth;  // Counts to MaxBatch, Rows or DrainCycles

    // The counter must reach every limit it counts to.
    if ((Rows > MaxBatch) || (DrainCycles > MaxBatch) || (DrainCycles < 1)) begin : gen_count_check
        $error("count_q is too narrow for Rows or DrainCycles");
    end

    // Tiling state; each is described on its case arm.
    typedef enum logic [2:0] {
        StIdle,
        StLoad,
        StStream,
        StDrain,
        StWaitEmpty
    } state_e;

    state_e                    state_q, state_d;  // Where in the tile sequence we are
    logic [    CountWidth-1:0] count_q, count_d;  // Cycle within the current state
    logic                      layer_q, layer_d;  // 0: layer 1, 1: layer 2
    logic [    BlockWidth-1:0] block_q, block_d;  // Neuron block nb within the layer
    logic [    BlockWidth-1:0] row_block_q, row_block_d;  // Input block kb within the layer
    logic [    BatchWidth-1:0] batch_size_q, batch_size_d;  // Images per batch for this run
    logic [    ImageWidth-1:0] batch_start_q, batch_start_d;  // First image of the batch
    logic [    BatchWidth-1:0] batch_len_q, batch_len_d;  // Images in this batch
    logic                      done_q, done_d;  // Registered done pulse
    logic [              31:0] cycle_count_q, cycle_count_d;  // Cycles since start

    // Last input block and last neuron block of the current layer.
    logic [BlockWidth-1:0] last_row_block;  // kb of the last tile in a neuron block
    logic [BlockWidth-1:0] last_block;  // nb of the last neuron block
    assign last_row_block = layer_q ? BlockWidth'(L2RowBlocks - 1) : BlockWidth'(L1RowBlocks - 1);
    assign last_block     = layer_q ? BlockWidth'(L2ColBlocks - 1) : BlockWidth'(L1ColBlocks - 1);

    // Images left from a batch start: the next batch is the smaller of that and the batch size.
    function automatic logic [BatchWidth-1:0] batch_length(input logic [ImageWidth-1:0] start,
                                                           input logic [BatchWidth-1:0] size);
        logic [ImageWidth-1:0] remaining;  // images from start to the end of the test set
        remaining = ImageWidth'(NumImages) - start;
        batch_length = size;
        if (BatchWidth'(remaining) < size) begin
            batch_length = BatchWidth'(remaining);
        end
    endfunction

    // A batch size of 0 or above MaxBatch means MaxBatch.
    logic [BatchWidth-1:0] start_batch_size;  // batch_size_i, made legal
    assign start_batch_size = ((batch_size_i == '0) || (batch_size_i > BatchWidth'(MaxBatch))) ?
        BatchWidth'(MaxBatch) : batch_size_i;

    // Next state and counters.
    always_comb begin
        state_d       = state_q;
        count_d       = count_q;
        layer_d       = layer_q;
        block_d       = block_q;
        row_block_d   = row_block_q;
        batch_size_d  = batch_size_q;
        batch_start_d = batch_start_q;
        batch_len_d   = batch_len_q;
        done_d        = 1'b0;
        cycle_count_d = cycle_count_q;
        if (state_q != StIdle) begin
            cycle_count_d = cycle_count_q + 32'd1;
        end

        unique case (state_q)
            // Wait for start; set up the first batch of layer 1.
            StIdle: begin
                if (start_i) begin
                    state_d       = StLoad;
                    count_d       = '0;
                    layer_d       = 1'b0;
                    block_d       = '0;
                    row_block_d   = '0;
                    batch_size_d  = start_batch_size;
                    batch_start_d = '0;
                    batch_len_d   = batch_length('0, start_batch_size);
                    cycle_count_d = 32'd1;
                end
            end
            // Read the tile's Rows weight words, one per cycle.
            StLoad: begin
                count_d = count_q + 1'b1;
                if (count_q == CountWidth'(Rows - 1)) begin
                    state_d = StStream;
                    count_d = '0;
                end
            end
            // Read one input vector per image of the batch.
            StStream: begin
                count_d = count_q + 1'b1;
                if (count_q == CountWidth'(batch_len_q - 1'b1)) begin
                    state_d = StDrain;
                    count_d = '0;
                end
            end
            // Let the last vector clear the array, then move to the next tile.
            StDrain: begin
                count_d = count_q + 1'b1;
                if (count_q == CountWidth'(DrainCycles - 1)) begin
                    count_d = '0;
                    state_d = StLoad;
                    if (row_block_q != last_row_block) begin
                        row_block_d = row_block_q + 1'b1;
                    end else begin
                        row_block_d = '0;
                        if (block_q != last_block) begin
                            block_d = block_q + 1'b1;
                        end else begin
                            block_d = '0;
                            state_d = StWaitEmpty;
                        end
                    end
                end
            end
            // A layer is finished: wait for its results, then start the next layer or batch.
            StWaitEmpty: begin
                if (!pipe_busy_i) begin
                    if (!layer_q) begin
                        state_d = StLoad;
                        layer_d = 1'b1;
                    end else if ((batch_start_q + ImageWidth'(batch_len_q)) >=
                                 ImageWidth'(NumImages)) begin
                        state_d = StIdle;
                        done_d  = 1'b1;
                    end else begin
                        state_d       = StLoad;
                        layer_d       = 1'b0;
                        batch_start_d = batch_start_q + ImageWidth'(batch_len_q);
                        batch_len_d   = batch_length(batch_start_d, batch_size_q);
                    end
                end
            end
            default: begin
                state_d = StIdle;
            end
        endcase
    end

    // Control registers.
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q       <= StIdle;
            count_q       <= '0;
            layer_q       <= 1'b0;
            block_q       <= '0;
            row_block_q   <= '0;
            batch_size_q  <= '0;
            batch_start_q <= '0;
            batch_len_q   <= '0;
            done_q        <= 1'b0;
            cycle_count_q <= '0;
        end else begin
            state_q       <= state_d;
            count_q       <= count_d;
            layer_q       <= layer_d;
            block_q       <= block_d;
            row_block_q   <= row_block_d;
            batch_size_q  <= batch_size_d;
            batch_start_q <= batch_start_d;
            batch_len_q   <= batch_len_d;
            done_q        <= done_d;
            cycle_count_q <= cycle_count_d;
        end
    end

    // ---- Issue: the memory reads for this cycle, registered ----

    logic [WeightAddrWidth-1:0] tile;  // Current tile, 0..NumTiles-1

    // Tile number in the weight memory: layer 1 tiles first, kb inner.
    always_comb begin
        if (layer_q) begin
            tile = WeightAddrWidth'(L1Tiles) + WeightAddrWidth'(block_q) *
                WeightAddrWidth'(L2RowBlocks) + WeightAddrWidth'(row_block_q);
        end else begin
            tile = WeightAddrWidth'(block_q) * WeightAddrWidth'(L1RowBlocks) +
                WeightAddrWidth'(row_block_q);
        end
    end

    logic weight_read_q, act_read_q;  // Read strobes, reset so they are never X
    logic [WeightAddrWidth-1:0] weight_addr_q;  // Registered weight address
    logic [ ImageAddrWidth-1:0] image_addr_q;  // Registered image address
    logic [HiddenAddrWidth-1:0] hidden_addr_q;  // Registered hidden address
    logic                       layer_tag_q;  // Registered tag: layer
    logic [BatchIndexWidth-1:0] batch_index_q;  // Registered tag: image in batch
    logic [     BlockWidth-1:0] block_tag_q;  // Registered tag: neuron block
    logic                       first_q;  // Registered tag: first input block
    logic                       last_q;  // Registered tag: last input block

    // Read strobes are control: reset them.
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            weight_read_q <= 1'b0;
            act_read_q    <= 1'b0;
        end else begin
            weight_read_q <= (state_q == StLoad);
            act_read_q    <= (state_q == StStream);
        end
    end

    // Addresses and tags are datapath, qualified by the strobes.
    always_ff @(posedge clk_i) begin
        weight_addr_q <= tile * WeightAddrWidth'(Rows) + WeightAddrWidth'(count_q);
        image_addr_q  <= (ImageAddrWidth'(batch_start_q) + ImageAddrWidth'(count_q)) *
            ImageAddrWidth'(L1RowBlocks) + ImageAddrWidth'(row_block_q);
        hidden_addr_q <= HiddenAddrWidth'(count_q) * HiddenAddrWidth'(L1ColBlocks) +
            HiddenAddrWidth'(row_block_q);
        layer_tag_q   <= layer_q;
        batch_index_q <= BatchIndexWidth'(count_q);
        block_tag_q   <= block_q;
        first_q       <= (row_block_q == '0);
        last_q        <= (row_block_q == last_row_block);
    end

    assign busy_o        = (state_q != StIdle);
    assign done_o        = done_q;
    assign cycle_count_o = cycle_count_q;
    assign weight_read_o = weight_read_q;
    assign weight_addr_o = weight_addr_q;
    assign act_read_o    = act_read_q;
    assign image_addr_o  = image_addr_q;
    assign hidden_addr_o = hidden_addr_q;
    assign layer_o       = layer_tag_q;
    assign batch_index_o = batch_index_q;
    assign block_o       = block_tag_q;
    assign first_o       = first_q;
    assign last_o        = last_q;
    assign batch_start_o = batch_start_q;

    // ---- Assertions ----
    // Never a weight read and an input read in the same cycle.
    `ASSERT(OneReadPerCycle_A, !(weight_read_o && act_read_o))
    // Streamed images stay inside the test set and the batch memories.
    `ASSERT(ImageInRange_A, act_read_o && !layer_o |-> image_addr_o < ImageAddrWidth'(ImageDepth))
    `ASSERT(WeightInRange_A, weight_read_o |-> weight_addr_o < WeightAddrWidth'(WeightDepth))
    // A batch is never empty.
    `ASSERT(BatchNotEmpty_A, busy_o |-> batch_len_q != '0)
    // Control outputs are never X once out of reset (VER-002).
    `ASSERT_KNOWN(BusyKnown_A, busy_o)
    `ASSERT_KNOWN(DoneKnown_A, done_o)
    `ASSERT_KNOWN(ReadsKnown_A, {weight_read_o, act_read_o})
endmodule

`resetall

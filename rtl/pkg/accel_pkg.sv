// int8_mlp: INT8 MLP accelerator on a weight-stationary systolic array (Artix-7)
//
// Description:
//   Sizes shared by the accelerator: the array shape, the data widths, the
//   MLP (64 -> 32 -> 10) and how its two layers are cut into 8x8 tiles.
//   The memory layouts that use these numbers are described in
//   model/gen_vectors.py, which writes the files the memories load.

`default_nettype none

package accel_pkg;
    // ---- Array ----
    localparam int unsigned Rows = 8;  // PE rows: inputs per tile
    localparam int unsigned Cols = 8;  // PE columns: neurons per tile
    localparam int unsigned DataWidth = 8;  // Weights, pixels and hidden values (signed)
    localparam int unsigned AccWidth = 24;  // Accumulators and biases (signed)

    // ---- Requantisation: hidden = clamp((acc * m + 2^(Shift-1)) >> Shift, 0, 127) ----
    localparam int unsigned MultWidth = 16;  // Width of the multipliers m (unsigned)
    localparam int unsigned Shift = 16;  // Right shift after the multiply

    // ---- The MLP ----
    localparam int unsigned NumInputs = 64;  // Pixels per image
    localparam int unsigned NumHidden = 32;  // Layer 1 neurons
    localparam int unsigned NumOutputs = 10;  // Layer 2 neurons: one score per digit

    // ---- Tiling: each layer is cut into blocks of Rows inputs and Cols neurons ----
    localparam int unsigned L1RowBlocks = NumInputs / Rows;  // 8 input blocks in layer 1
    localparam int unsigned L1ColBlocks = NumHidden / Cols;  // 4 neuron blocks in layer 1
    localparam int unsigned L2RowBlocks = NumHidden / Rows;  // 4 input blocks in layer 2
    localparam int unsigned L2ColBlocks = (NumOutputs + Cols - 1) / Cols;  // 2, 10 padded to 16
    localparam int unsigned L1Tiles = L1RowBlocks * L1ColBlocks;  // 32 layer 1 tiles
    localparam int unsigned L2Tiles = L2RowBlocks * L2ColBlocks;  // 8 layer 2 tiles
    localparam int unsigned NumTiles = L1Tiles + L2Tiles;  // 40 tiles, stored in this order
    localparam int unsigned NumBlocks = L1ColBlocks + L2ColBlocks;  // 6 bias words

    // ---- Data set and batching ----
    localparam int unsigned NumImages = 360;  // Test images in the image memory
    localparam int unsigned MaxBatch = 512;  // Most images per batch (memory depth)
    localparam int unsigned BatchWidth = $clog2(MaxBatch + 1);  // Holds 1..MaxBatch
    localparam int unsigned DigitWidth = $clog2(NumOutputs);  // Holds a digit 0..9
    localparam int unsigned ImageWidth = $clog2(NumImages + 1);  // Holds an image count 0..360
    localparam int unsigned BatchIndexWidth = $clog2(MaxBatch);  // Image within a batch

    // ---- Memories: depth in words, and address widths ----
    localparam int unsigned WeightDepth = NumTiles * Rows;  // One word per weight-load cycle
    localparam int unsigned ImageDepth = NumImages * L1RowBlocks;  // One word per image row block
    localparam int unsigned HiddenDepth = MaxBatch * L1ColBlocks;  // One word per hidden block
    localparam int unsigned WeightAddrWidth = $clog2(WeightDepth);  // Weight memory address
    localparam int unsigned ImageAddrWidth = $clog2(ImageDepth);  // Image memory address
    localparam int unsigned HiddenAddrWidth = $clog2(HiddenDepth);  // Hidden memory address
    localparam int unsigned BlockWidth = $clog2(NumBlocks);  // Bias word address; holds a block

    // ---- Vector types: one lane per array row or column, lane 0 in the low bits ----
    typedef logic [Rows-1:0][DataWidth-1:0] act_vector_t;  // One input vector into the array
    typedef logic [Cols-1:0][DataWidth-1:0] weight_vector_t;  // One weight row, top edge
    typedef logic [Cols-1:0][AccWidth-1:0] acc_vector_t;  // One partial sum per column
    typedef logic [Cols-1:0][MultWidth-1:0] mult_vector_t;  // One multiplier per column
    typedef logic [Cols-1:0][DataWidth-1:0] hidden_vector_t;  // One requantised value per column
endpackage

`resetall

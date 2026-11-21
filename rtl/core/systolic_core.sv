// int8_mlp: INT8 MLP accelerator on a weight-stationary systolic array (Artix-7)
//
// Description:
//   The PE array with its skew and de-skew buffers, so the outside sees
//   aligned vectors: a whole input vector enters in one cycle, and all Cols
//   dot products leave together Rows + Cols - 1 cycles later.
//   act_i entered in cycle c gives psum_o in cycle c + Rows + Cols - 1.
//   The de-skew buffer is a second skew_buffer with the columns connected in
//   reverse, so column n waits Cols - 1 - n cycles for the slowest column.
//   Weights bypass the buffers: they are loaded before any input arrives.
//   Do not load weights while inputs are still in flight: weight_load_i
//   changes every PE's weight in the same cycle.

// svstyle: waive LNG-014 Vivado xsim rejects `input logic` ports under `default_nettype none
module systolic_core #(
    parameter int unsigned Rows      = 8,  // PE rows: inputs per tile
    parameter int unsigned Cols      = 8,  // PE columns: neurons per tile
    parameter int unsigned DataWidth = 8,  // Width of weights and activations (signed)
    parameter int unsigned AccWidth  = 24  // Width of the partial sums (signed)
) (
    input  logic                           clk_i,          // System clock
    input  logic                           weight_load_i,  // Shift weights down every column
    input  logic [Cols-1:0][DataWidth-1:0] weight_i,       // Per-column weight, bottom row first
    input  logic [Rows-1:0][DataWidth-1:0] act_i,          // One input vector, aligned
    output logic [Cols-1:0][ AccWidth-1:0] psum_o          // Its dot products, aligned
);
    logic [Rows-1:0][DataWidth-1:0] act_skewed;  // Row k delayed k cycles, into the array
    logic [Cols-1:0][ AccWidth-1:0] array_psum;  // Column n leaves n cycles after column 0
    logic [Cols-1:0][ AccWidth-1:0] deskew_in;  // array_psum in reverse column order
    logic [Cols-1:0][ AccWidth-1:0] deskew_out;  // Still reversed, now aligned

    // Skew the inputs: row k enters the array k cycles after row 0.
    skew_buffer #(
        .Lanes(Rows),
        .Width(DataWidth)
    ) u_skew (
        .clk_i (clk_i),
        .data_i(act_i),
        .data_o(act_skewed)
    );

    // The PE array: column n computes the dot product of neuron n.
    pe_array #(
        .Rows     (Rows),
        .Cols     (Cols),
        .DataWidth(DataWidth),
        .AccWidth (AccWidth)
    ) u_array (
        .clk_i        (clk_i),
        .weight_load_i(weight_load_i),
        .weight_i     (weight_i),
        .act_i        (act_skewed),
        .psum_o       (array_psum)
    );

    // Reverse the columns around the de-skew buffer: column n uses lane
    // Cols-1-n, so the earliest column waits longest.
    for (genvar col = 0; col < Cols; col++) begin : gen_reverse
        assign deskew_in[Cols-1-col] = array_psum[col];
        assign psum_o[col]           = deskew_out[Cols-1-col];
    end

    // De-skew the outputs: line every column up with the last one.
    skew_buffer #(
        .Lanes(Cols),
        .Width(AccWidth)
    ) u_deskew (
        .clk_i (clk_i),
        .data_i(deskew_in),
        .data_o(deskew_out)
    );
endmodule

`resetall

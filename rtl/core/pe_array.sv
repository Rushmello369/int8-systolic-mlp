// int8_mlp: INT8 MLP accelerator on a weight-stationary systolic array (Artix-7)
//
// Description:
//   Weight-stationary systolic array of Rows x Cols PEs (pe.sv).
//   Column n holds the weights of one neuron; row k carries one input.
//   PE[k][n] holds weight W[n, k], multiplies it by the activation arriving
//   from the left, and adds the partial sum arriving from above.
//   Activations flow right and partial sums flow down, one PE per cycle.
//   Weights load by shifting down each column while weight_load_i is high:
//   send the bottom row's weight first and row 0's last (Rows cycles).
//   act_i must already be skewed: row k enters k cycles after row 0.
//   psum_o is still skewed: column n leaves n cycles after column 0.
//   The array is wiring only; every register is inside the PEs.

// svstyle: waive LNG-014 Vivado xsim rejects `input logic` ports under `default_nettype none
module pe_array #(
    parameter int unsigned Rows      = 8,  // PE rows: inputs per tile
    parameter int unsigned Cols      = 8,  // PE columns: neurons per tile
    parameter int unsigned DataWidth = 8,  // Width of weights and activations (signed)
    parameter int unsigned AccWidth  = 24  // Width of the partial sums (signed)
) (
    input  logic                           clk_i,          // System clock
    input  logic                           weight_load_i,  // Shift weights down every column
    input  logic [Cols-1:0][DataWidth-1:0] weight_i,       // Top edge, one weight per column
    input  logic [Rows-1:0][DataWidth-1:0] act_i,          // Left edge, one input per row, skewed
    output logic [Cols-1:0][ AccWidth-1:0] psum_o          // Bottom edge, one sum per column
);
    // Each wire group has one more position than PEs along its flow direction:
    // position 0 is fed from the edge, and PE i drives position i+1.
    logic [Rows-1:0][  Cols:0][DataWidth-1:0] act;  // Activations, flowing right
    // The last row of positions (weight[Rows]) is the bottom PEs' weight_o: unused.
    /* verilator lint_off UNUSEDSIGNAL */
    logic [  Rows:0][Cols-1:0][DataWidth-1:0] weight;  // Weights, shifting down while loading
    /* verilator lint_on UNUSEDSIGNAL */
    logic [  Rows:0][Cols-1:0][ AccWidth-1:0] psum;  // Partial sums, flowing down

    // Left edge: each row's first act position comes from the input port.
    for (genvar row = 0; row < Rows; row++) begin : gen_left_edge
        assign act[row][0] = act_i[row];
    end

    // Top edge: weights enter from the input port; partial sums start at zero.
    for (genvar col = 0; col < Cols; col++) begin : gen_top_edge
        assign weight[0][col] = weight_i[col];
        assign psum[0][col]   = '0;
    end

    // Bottom edge: each column's final partial sum leaves through the output port.
    // The last act position of each row and the last weight position of each
    // column are left unread; synthesis removes the registers that drive only them.
    for (genvar col = 0; col < Cols; col++) begin : gen_bottom_edge
        assign psum_o[col] = psum[Rows][col];
    end

    // The grid: every PE reads its current position and drives the next one,
    // so edge PEs need no special case.
    for (genvar row = 0; row < Rows; row++) begin : gen_row
        for (genvar col = 0; col < Cols; col++) begin : gen_col
            // PE[row][col]: holds W[col, row] for the current tile
            pe #(
                .DataWidth(DataWidth),
                .AccWidth (AccWidth)
            ) u_pe (
                .clk_i        (clk_i),
                .weight_load_i(weight_load_i),
                .weight_i     (weight[row][col]),
                .weight_o     (weight[row+1][col]),
                .act_i        (act[row][col]),
                .act_o        (act[row][col+1]),
                .psum_i       (psum[row][col]),
                .psum_o       (psum[row+1][col])
            );
        end
    end
endmodule

`resetall

// int8_mlp: INT8 MLP accelerator on a weight-stationary systolic array (Artix-7)
//
// Description:
//   Requantiser for one neuron: turns a layer 1 accumulator into an INT8
//   hidden value, with ReLU folded into the clamp. Same arithmetic as
//   requantize() in model/golden.py:
//     value = clamp((acc * mult + 2^(Shift-1)) >>> Shift, 0, 2^(OutWidth-1) - 1)
//   Adding half a step before the arithmetic right shift rounds to nearest,
//   with ties rounded up (towards +infinity), exactly like NumPy's >>.
//   Two pipeline stages, so value_o is valid 2 cycles after acc_i:
//     1. the multiply (one DSP48 slice)
//     2. round, shift and clamp
//   No reset: datapath only; the caller tracks which cycles are valid.

// svstyle: waive LNG-014 Vivado xsim rejects `input logic` ports under `default_nettype none
module requant #(
    parameter int unsigned AccWidth  = 24,  // Accumulator width (signed)
    parameter int unsigned MultWidth = 16,  // Multiplier width (unsigned)
    parameter int unsigned Shift     = 16,  // Right shift; at least 1
    parameter int unsigned OutWidth  = 8    // Output width; the result is 0..2^(OutWidth-1)-1
) (
    input  logic                        clk_i,   // System clock
    input  logic signed [ AccWidth-1:0] acc_i,   // Accumulator, bias already added
    input  logic        [MultWidth-1:0] mult_i,  // This neuron's multiplier m
    output logic        [ OutWidth-1:0] value_o  // Requantised value, 2 cycles after acc_i
);
    // Rounding adds 2^(Shift-1), which needs Shift >= 1.
    if (Shift < 1) begin : gen_shift_check
        $error("Shift must be at least 1");
    end

    // The product needs AccWidth + MultWidth bits, plus one because the
    // unsigned multiplier gains a sign bit when it becomes a signed operand.
    localparam int unsigned ProductWidth = AccWidth + MultWidth + 1;  // 41 bits by default

    // ---- Stage 1: multiply ----
    logic signed [ProductWidth-1:0] product_q;  // acc_i * mult_i, exact

    // Register the product. {1'b0, mult_i} makes the multiplier a positive signed number.
    always_ff @(posedge clk_i) begin
        product_q <= ProductWidth'(acc_i) * ProductWidth'($signed({1'b0, mult_i}));
    end

    // ---- Stage 2: round, shift, clamp ----
    localparam logic signed [ProductWidth-1:0] Half = ProductWidth'(1) <<< (Shift - 1);  // 0.5 LSB
    localparam logic signed [ProductWidth-1:0] MaxValue = (2 ** (OutWidth - 1)) - 1;  // 127

    logic signed [ProductWidth-1:0] rounded;  // product scaled down and rounded to nearest
    assign rounded = (product_q + Half) >>> Shift;

    logic [OutWidth-1:0] value_q, value_d;  // Clamped result

    // Clamp to 0..MaxValue: the lower bound is the ReLU, the upper one saturates.
    always_comb begin
        value_d = OutWidth'(rounded);
        if (rounded < 0) begin
            value_d = '0;
        end else if (rounded > MaxValue) begin
            value_d = OutWidth'(MaxValue);
        end
    end

    // Register the clamped value.
    always_ff @(posedge clk_i) begin
        value_q <= value_d;
    end

    assign value_o = value_q;
endmodule

`resetall

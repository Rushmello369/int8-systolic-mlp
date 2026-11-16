// int8_mlp: INT8 MLP accelerator on a weight-stationary systolic array (Artix-7)
//
// Description:
//   Processing element (PE) of the weight-stationary systolic array.
//   Holds one signed INT8 weight. Every cycle it multiplies the activation
//   arriving from the left by that weight, adds the partial sum arriving from
//   above, and registers the result for the PE below. The activation is
//   registered and passed on to the PE on the right.
//   Weights are loaded by shifting them down the column: while weight_load_i
//   is high, the PE takes weight_i and offers its old weight on weight_o.
//   Latency: act_o and psum_o appear one cycle after act_i and psum_i.
//   The partial sum wraps on overflow; AccWidth must be chosen so that the
//   largest possible sum fits (24 bits for 64 terms: |sum| <= 2^20).
//   No reset: every flop is datapath. The array controller drives zeros
//   into act_i and psum_i until real data arrives.

// svstyle: waive LNG-014 Vivado xsim rejects `input logic` ports under `default_nettype none
module pe #(
    parameter int unsigned DataWidth = 8,  // Width of weights and activations (signed)
    parameter int unsigned AccWidth  = 24  // Width of the partial sums (signed)
) (
    input  logic                        clk_i,          // System clock
    input  logic                        weight_load_i,  // Take weight_i into the weight register
    input  logic signed [DataWidth-1:0] weight_i,       // Weight shifted in from the PE above
    output logic signed [DataWidth-1:0] weight_o,       // Stored weight, shifted on to the PE below
    input  logic signed [DataWidth-1:0] act_i,          // Activation from the PE on the left
    output logic signed [DataWidth-1:0] act_o,          // act_i delayed one cycle, to the right
    input  logic signed [ AccWidth-1:0] psum_i,         // Partial sum from the PE above
    output logic signed [ AccWidth-1:0] psum_o          // psum_i + act_i * weight, one cycle later
);
    // The accumulator must at least hold one full product.
    if (AccWidth < 2 * DataWidth) begin : gen_width_check
        $error("AccWidth must be at least 2 * DataWidth");
    end

    logic signed [DataWidth-1:0] weight_q;  // The stationary weight of this PE

    // Load a new weight only while the column is shifting weights in.
    always_ff @(posedge clk_i) begin
        if (weight_load_i) begin
            weight_q <= weight_i;
        end
    end

    // The product gets its own signal, 2*DataWidth wide, to hold every result
    // exactly (-128 * -128 = 16384 needs 16 bits). A narrower signal would drop the
    // top bits; the testbench catches that (tested by making this 8 bits wide).
    logic signed [2*DataWidth-1:0] product;  // act_i * weight_q, exact
    assign product = act_i * weight_q;

    logic signed [AccWidth-1:0] psum_q, psum_d;  // Registered partial sum for the PE below
    assign psum_d = psum_i + AccWidth'(product);  // the cast sign-extends the signed product

    logic signed [DataWidth-1:0] act_q;  // Activation register, passed to the right

    // Datapath registers: the new partial sum and the passed-on activation.
    always_ff @(posedge clk_i) begin
        psum_q <= psum_d;
        act_q  <= act_i;
    end

    assign weight_o = weight_q;
    assign act_o    = act_q;
    assign psum_o   = psum_q;
endmodule

`resetall

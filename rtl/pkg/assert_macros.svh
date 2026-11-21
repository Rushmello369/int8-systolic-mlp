// int8_mlp: INT8 MLP accelerator on a weight-stationary systolic array (Artix-7)
//
// Description:
//   Assertion macros for synthesizable RTL (VER-001). Checks run in
//   simulation; synthesis tools define SYNTHESIS and see nothing.

`ifndef ASSERT_MACROS_SVH
`define ASSERT_MACROS_SVH

`ifndef SYNTHESIS

// Concurrent assertion, clocked on clk and disabled while rst_n is low or
// unknown. name_rst_q remembers a reset since the last edge and disables
// every attempt still running at that edge. Using rst_n both as the async
// reset of that flop and inside the property is intended, so Verilator's
// SYNCASYNCNET warning is switched off around it.
`define ASSERT(name, prop, clk = clk_i, rst_n = rst_ni) \
    /* verilator lint_off SYNCASYNCNET */ \
    logic name``_rst_q; \
    always_ff @(posedge clk or negedge rst_n) name``_rst_q <= !rst_n; \
    name : assert property (@(posedge clk) disable iff ( \
        $isunknown(rst_n) || !rst_n || $sampled(name``_rst_q)) (prop)) \
    else $error("assertion %s failed", `"name`"); \
    /* verilator lint_on SYNCASYNCNET */

// A signal is never X or Z outside reset (VER-002).
`define ASSERT_KNOWN(name, sig, clk = clk_i, rst_n = rst_ni) \
    `ASSERT(name, !$isunknown(sig), clk, rst_n)

// Checked once at time zero, e.g. on parameters.
`define ASSERT_INIT(name, prop) \
    initial begin : name \
        assert (prop) else $error("assertion %s failed", `"name`"); \
    end

`else

`define ASSERT(name, prop, clk = clk_i, rst_n = rst_ni)
`define ASSERT_KNOWN(name, sig, clk = clk_i, rst_n = rst_ni)
`define ASSERT_INIT(name, prop)

`endif  // SYNTHESIS

`endif  // ASSERT_MACROS_SVH

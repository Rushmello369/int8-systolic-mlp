// int8_mlp: INT8 MLP accelerator on a weight-stationary systolic array (Artix-7)
//
// Description:
//   Delay line for Lanes parallel inputs: lane k leaves k cycles after it
//   enters, so a vector that enters in one cycle leaves as a diagonal.
//   Skew: feeds the array's left edge (Width = 8), since row k must enter
//   k cycles after row 0.
//   De-skew: the same module on the array's outputs (Width = 24), with the
//   columns connected in reverse order so column n gets Cols-1-n cycles.
//   Lane 0 is a wire; lane k is a chain of k registers.
//   No reset: the registers hold data only, and without a reset Vivado can
//   map each chain into LUT shift registers (SRL) instead of flip-flops.

// svstyle: waive LNG-014 Vivado xsim rejects `input logic` ports under `default_nettype none
module skew_buffer #(
    parameter int unsigned Lanes = 8,  // Independent channels; lane k is delayed k cycles
    parameter int unsigned Width = 8   // Bits per lane
) (
    input  logic                        clk_i,   // System clock
    input  logic [Lanes-1:0][Width-1:0] data_i,  // All lanes enter in the same cycle
    output logic [Lanes-1:0][Width-1:0] data_o   // Lane k leaves k cycles after it entered
);
    // Lanes must be positive: with none there is nothing to delay.
    if (Lanes < 1) begin : gen_lanes_check
        $error("Lanes must be at least 1");
    end

    // One delay chain per lane, each one register longer than the last.
    for (genvar lane = 0; lane < Lanes; lane++) begin : gen_lane
        if (lane == 0) begin : gen_pass
            // Lane 0 has no delay: a plain wire.
            assign data_o[0] = data_i[0];
        end else begin : gen_delay
            // Separate branch because [lane-1:0] with lane = 0 would be [-1:0], two registers.
            logic [lane-1:0][Width-1:0] delay_q;  // Stage 0 is newest, stage lane-1 is oldest

            // Shift the chain by one stage every cycle.
            always_ff @(posedge clk_i) begin
                delay_q[0] <= data_i[lane];
                for (int stage = 1; stage < lane; stage++) begin
                    delay_q[stage] <= delay_q[stage-1];
                end
            end

            assign data_o[lane] = delay_q[lane-1];
        end
    end
endmodule

`resetall

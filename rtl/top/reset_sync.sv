// int8_mlp: INT8 MLP accelerator on a weight-stationary systolic array (Artix-7)
//
// Description:
//   Reset synchroniser: asserts rst_no as soon as rst_async_ni goes low
//   (asynchronously, even with no clock), and releases it only on a clock
//   edge, two edges after rst_async_ni rises. Every flop in the clock
//   domain then leaves reset in the same cycle (RST-002).

// svstyle: waive LNG-014 Vivado xsim rejects `input logic` ports under `default_nettype none
module reset_sync (
    input  logic clk_i,         // Clock of the domain being reset
    input  logic rst_async_ni,  // Asynchronous reset request, active-low
    output logic rst_no         // Reset for the domain: async assert, sync release
);
    (* ASYNC_REG = "TRUE" *) logic [1:0] sync_q;  // Shifts in ones after the request ends

    // Clear both flops at once on a request; shift a 1 through on release.
    always_ff @(posedge clk_i or negedge rst_async_ni) begin
        if (!rst_async_ni) begin
            sync_q <= 2'b00;
        end else begin
            sync_q <= {sync_q[0], 1'b1};
        end
    end

    assign rst_no = sync_q[1];
endmodule

`resetall

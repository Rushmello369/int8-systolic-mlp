// int8_mlp: INT8 MLP accelerator on a weight-stationary systolic array (Artix-7)
//
// Description:
//   UART transmitter, 8N1, LSB first. ClksPerBit = clock / baud rate.
//   Pulse send_i with a byte on data_i while busy_o is low; busy_o is high
//   from the next cycle until the stop bit has been sent. A send_i while
//   busy is ignored. tx_o idles high and is driven straight from a flop,
//   so it never glitches.

// svstyle: waive LNG-014 Vivado xsim rejects `input logic` ports under `default_nettype none
module uart_tx #(
    parameter int unsigned ClksPerBit = 1302  // Clock cycles per bit; at least 2
) (
    input  logic       clk_i,   // System clock
    input  logic       rst_ni,  // Asynchronous reset, active-low
    input  logic       send_i,  // Start sending data_i (when not busy)
    input  logic [7:0] data_i,  // Byte to send, sampled with send_i
    output logic       busy_o,  // A frame is being sent
    output logic       tx_o     // Serial output, idle high
);
    // The bit counter needs at least two cycles per bit.
    if (ClksPerBit < 2) begin : gen_clks_check
        $error("ClksPerBit must be at least 2");
    end

    localparam int unsigned CountWidth = $clog2(ClksPerBit);  // Counts cycles within a bit

    logic                  busy_q;  // A frame is in progress
    logic [CountWidth-1:0] count_q;  // Cycles into the current bit
    logic [           3:0] bit_q;  // Bits sent: 0 start, 1..8 data, 9 stop
    logic [           8:0] frame_q;  // Stop bit and data still to send; bit 0 goes out next
    logic                  tx_q;  // Line level

    // Load a frame on send_i, then shift it out one bit every ClksPerBit cycles.
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            busy_q  <= 1'b0;
            count_q <= '0;
            bit_q   <= '0;
            frame_q <= '1;
            tx_q    <= 1'b1;
        end else if (!busy_q) begin
            if (send_i) begin
                busy_q  <= 1'b1;
                count_q <= '0;
                bit_q   <= '0;
                frame_q <= {1'b1, data_i};
                tx_q    <= 1'b0;
            end
        end else if (count_q == CountWidth'(ClksPerBit - 1)) begin
            count_q <= '0;
            if (bit_q == 4'd9) begin
                busy_q <= 1'b0;
                tx_q   <= 1'b1;
            end else begin
                bit_q   <= bit_q + 4'd1;
                frame_q <= {1'b1, frame_q[8:1]};
                tx_q    <= frame_q[0];
            end
        end else begin
            count_q <= count_q + 1'b1;
        end
    end

    assign busy_o = busy_q;
    assign tx_o   = tx_q;
endmodule

`resetall

// int8_mlp: INT8 MLP accelerator on a weight-stationary systolic array (Artix-7)
//
// Description:
//   UART receiver, 8 data bits, no parity, 1 stop bit (8N1), LSB first.
//   ClksPerBit = clock frequency / baud rate (1302 for 150 MHz, 115200 baud).
//   The serial input comes from another chip with no shared clock, so it
//   first passes through a two-flop synchroniser. A falling edge starts a
//   frame; each bit is sampled in its middle (half a bit after the start
//   edge, then once per bit). A frame whose stop bit is 0 is dropped.
//   valid_o pulses for one cycle with the byte on data_o.

// svstyle: waive LNG-014 Vivado xsim rejects `input logic` ports under `default_nettype none
module uart_rx #(
    parameter int unsigned ClksPerBit = 1302  // Clock cycles per bit; at least 4
) (
    input  logic       clk_i,   // System clock
    input  logic       rst_ni,  // Asynchronous reset, active-low
    input  logic       rx_i,    // Serial input, asynchronous, idle high
    output logic       valid_o, // One-cycle pulse: data_o holds a received byte
    output logic [7:0] data_o   // The received byte
);
    // The mid-bit sample point needs a few cycles per bit.
    if (ClksPerBit < 4) begin : gen_clks_check
        $error("ClksPerBit must be at least 4");
    end

    localparam int unsigned CountWidth = $clog2(ClksPerBit);  // Counts cycles within a bit

    // ---- Synchroniser ----
    (* ASYNC_REG = "TRUE" *) logic [1:0] rx_sync_q;  // Two flops: rx_i into the clock domain

    // Two-flop synchroniser, reset to the idle level so reset never looks like a start bit.
    // ASYNC_REG keeps both flops together and tells timing analysis they are a synchroniser.
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            rx_sync_q <= 2'b11;
        end else begin
            rx_sync_q <= {rx_sync_q[0], rx_i};
        end
    end

    logic rx;  // rx_i, synchronised
    assign rx = rx_sync_q[1];

    // ---- Receiver ----
    // Frame position; each is described on its case arm.
    typedef enum logic [1:0] {
        StIdle,
        StStart,
        StData,
        StStop
    } state_e;

    state_e                  state_q, state_d;  // Position in the frame
    logic   [CountWidth-1:0] count_q, count_d;  // Cycles since the last sample point
    logic   [           2:0] bit_q, bit_d;  // Data bit being received
    logic   [           7:0] shift_q, shift_d;  // Data bits so far, LSB arrives first
    logic                    valid_q, valid_d;  // Registered byte-ready pulse

    // Next state, counters and data.
    always_comb begin
        state_d = state_q;
        count_d = count_q + 1'b1;
        bit_d   = bit_q;
        shift_d = shift_q;
        valid_d = 1'b0;
        unique case (state_q)
            // Line idle: a falling edge is the start bit.
            StIdle: begin
                count_d = '0;
                if (!rx) begin
                    state_d = StStart;
                end
            end
            // Half a bit into the start bit: still low means a real start.
            StStart: begin
                if (count_q == CountWidth'(ClksPerBit / 2 - 1)) begin
                    count_d = '0;
                    bit_d   = '0;
                    state_d = rx ? StIdle : StData;
                end
            end
            // Sample each data bit in its middle, LSB first.
            StData: begin
                if (count_q == CountWidth'(ClksPerBit - 1)) begin
                    count_d = '0;
                    shift_d = {rx, shift_q[7:1]};
                    bit_d   = bit_q + 1'b1;
                    if (bit_q == 3'd7) begin
                        state_d = StStop;
                    end
                end
            end
            // Middle of the stop bit: keep the byte only if the stop bit is 1.
            StStop: begin
                if (count_q == CountWidth'(ClksPerBit - 1)) begin
                    valid_d = rx;
                    state_d = StIdle;
                end
            end
            default: begin
                state_d = StIdle;
            end
        endcase
    end

    // Receiver registers.
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= StIdle;
            count_q <= '0;
            bit_q   <= '0;
            shift_q <= '0;
            valid_q <= 1'b0;
        end else begin
            state_q <= state_d;
            count_q <= count_d;
            bit_q   <= bit_d;
            shift_q <= shift_d;
            valid_q <= valid_d;
        end
    end

    assign valid_o = valid_q;
    assign data_o  = shift_q;
endmodule

`resetall

// int8_mlp: INT8 MLP accelerator on a weight-stationary systolic array (Artix-7)
//
// Description:
//   Host command handler between the UART and mlp_core.
//   Protocol (bytes, big-endian numbers):
//     host -> board:  'R' (0x52), batch size high byte, batch size low byte
//     board -> host:  one byte per test image: its predicted digit (0..9),
//                     then correct count (2 bytes), checksum (4 bytes),
//                     cycle count (4 bytes): NumImages + 10 bytes in all.
//   Any other first byte is ignored, so the host can resynchronise by
//   sending 'R' again after a garbled byte. Commands that arrive while a
//   run or its reply is in progress are ignored.
//   The predictions are read from mlp_core's result port one image at a
//   time (one-cycle read latency), each sent when the UART is free.

// svstyle: waive LNG-014 Vivado xsim rejects `input logic` ports under `default_nettype none
module host_if
    import accel_pkg::*;
(
    input  logic                  clk_i,            // System clock
    input  logic                  rst_ni,           // Asynchronous reset, active-low
    // ---- UART receiver ----
    input  logic                  rx_valid_i,       // A byte has arrived
    input  logic [           7:0] rx_data_i,        // The byte
    // ---- UART transmitter ----
    output logic                  tx_send_o,        // Send tx_data_o now
    output logic [           7:0] tx_data_o,        // Byte to send
    input  logic                  tx_busy_i,        // Transmitter is sending a byte
    // ---- mlp_core ----
    output logic                  start_o,          // One-cycle pulse: start a run
    output logic [BatchWidth-1:0] batch_size_o,     // Images per batch for the run
    input  logic                  done_i,           // One-cycle pulse: the run has finished
    input  logic [          31:0] cycle_count_i,    // Cycles the run took
    input  logic [          15:0] correct_count_i,  // Correct predictions
    input  logic [          31:0] checksum_i,       // Sum of all scores
    output logic [ImageWidth-1:0] result_image_o,   // Image whose prediction to read
    input  logic [DigitWidth-1:0] result_digit_i    // Its digit, one cycle later
);
    localparam logic [7:0] CmdRun = 8'h52;  // 'R': start a run
    localparam int unsigned SummaryBytes = 10;  // 2 correct + 4 checksum + 4 cycles

    // Command and reply phases; each is described on its case arm.
    typedef enum logic [2:0] {
        StIdle,
        StSizeHigh,
        StSizeLow,
        StRun,
        StFetch,
        StSendDigit,
        StSendSummary
    } state_e;

    state_e                 state_q, state_d;  // Where in the command or reply we are
    logic   [          7:0] size_high_q, size_high_d;  // Batch size high byte
    logic   [BatchWidth-1:0] batch_size_q, batch_size_d;  // Batch size for the run
    logic                   start_q, start_d;  // Registered start pulse
    logic   [ImageWidth-1:0] image_q, image_d;  // Next image to report
    logic   [          3:0] byte_q, byte_d;  // Summary bytes sent so far
    logic   [SummaryBytes*8-1:0] summary_q, summary_d;  // Summary, next byte at the top
    logic                   send_q, send_d;  // Registered send pulse
    logic   [          7:0] tx_data_q, tx_data_d;  // Registered byte to send

    // Next state and reply data.
    always_comb begin
        state_d      = state_q;
        size_high_d  = size_high_q;
        batch_size_d = batch_size_q;
        start_d      = 1'b0;
        image_d      = image_q;
        byte_d       = byte_q;
        summary_d    = summary_q;
        send_d       = 1'b0;
        tx_data_d    = tx_data_q;
        unique case (state_q)
            // Wait for the 'R' command byte.
            StIdle: begin
                if (rx_valid_i && (rx_data_i == CmdRun)) begin
                    state_d = StSizeHigh;
                end
            end
            // Receive the batch size, high byte first.
            StSizeHigh: begin
                if (rx_valid_i) begin
                    size_high_d = rx_data_i;
                    state_d     = StSizeLow;
                end
            end
            // Receive the low byte and start the run.
            StSizeLow: begin
                if (rx_valid_i) begin
                    batch_size_d = BatchWidth'({size_high_q, rx_data_i});
                    start_d      = 1'b1;
                    state_d      = StRun;
                end
            end
            // Wait for the run; latch the summary when it is done.
            StRun: begin
                if (done_i) begin
                    summary_d = {correct_count_i, checksum_i, cycle_count_i};
                    image_d   = '0;
                    state_d   = StFetch;
                end
            end
            // result_image_o = image_q this cycle: the digit arrives next cycle.
            StFetch: begin
                state_d = StSendDigit;
            end
            // Send the digit when the UART is free, then fetch the next one.
            StSendDigit: begin
                if (!tx_busy_i && !send_q) begin
                    send_d    = 1'b1;
                    tx_data_d = 8'(result_digit_i);
                    image_d   = image_q + 1'b1;
                    if (image_q == ImageWidth'(NumImages - 1)) begin
                        byte_d  = '0;
                        state_d = StSendSummary;
                    end else begin
                        state_d = StFetch;
                    end
                end
            end
            // Send the summary, most significant byte first.
            StSendSummary: begin
                if (!tx_busy_i && !send_q) begin
                    send_d    = 1'b1;
                    tx_data_d = summary_q[SummaryBytes*8-1-:8];
                    summary_d = {summary_q[SummaryBytes*8-9:0], 8'h00};
                    byte_d    = byte_q + 1'b1;
                    if (byte_q == 4'(SummaryBytes - 1)) begin
                        state_d = StIdle;
                    end
                end
            end
            default: begin
                state_d = StIdle;
            end
        endcase
    end

    // Handler registers.
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q      <= StIdle;
            size_high_q  <= '0;
            batch_size_q <= '0;
            start_q      <= 1'b0;
            image_q      <= '0;
            byte_q       <= '0;
            summary_q    <= '0;
            send_q       <= 1'b0;
            tx_data_q    <= '0;
        end else begin
            state_q      <= state_d;
            size_high_q  <= size_high_d;
            batch_size_q <= batch_size_d;
            start_q      <= start_d;
            image_q      <= image_d;
            byte_q       <= byte_d;
            summary_q    <= summary_d;
            send_q       <= send_d;
            tx_data_q    <= tx_data_d;
        end
    end

    assign start_o        = start_q;
    assign batch_size_o   = batch_size_q;
    assign result_image_o = image_q;
    assign tx_send_o      = send_q;
    assign tx_data_o      = tx_data_q;
endmodule

`resetall

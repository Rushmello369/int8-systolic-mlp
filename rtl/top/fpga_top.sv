// int8_mlp: INT8 MLP accelerator on a weight-stationary systolic array (Artix-7)
//
// Description:
//   Top level for the ALINX AX7A200 board (XC7A200T-2FBG484I). Pins are
//   in fpga/constraints/ax7a200.xdc, taken from the board's user manual.
//   Clocking: the 200 MHz differential board clock drives an MMCM that
//   makes the 150 MHz system clock (VCO 1200 MHz = 200 x 6; 1200 / 8).
//   Reset: the RESET key (active-low) or a lost MMCM lock resets the
//   whole design through one reset synchroniser.
//   The host talks to mlp_core over the on-board USB-UART (8N1, BaudRate);
//   see host_if.sv for the protocol.
//   LEDs (active-low on the board): LED1 = run in progress,
//   LED2 = results ready, LED3 = heartbeat (about 1 Hz), LED4 = MMCM locked.

// svstyle: waive LNG-014 Vivado xsim rejects `input logic` ports under `default_nettype none
module fpga_top
    import accel_pkg::*;
#(
    parameter int unsigned ClockHz  = 150_000_000,  // System clock from the MMCM
    parameter int unsigned BaudRate = 115_200       // UART speed; the testbench raises it
) (
    input  logic       sys_clk_p_i,  // 200 MHz board clock, positive side
    input  logic       sys_clk_n_i,  // 200 MHz board clock, negative side
    input  logic       reset_ni,     // RESET key, active-low, asynchronous
    input  logic       uart_rx_i,    // From the USB-UART chip, asynchronous
    output logic       uart_tx_o,    // To the USB-UART chip
    output logic [3:0] led_no        // User LEDs 1..4, active-low
);
    localparam int unsigned ClksPerBit = (ClockHz + BaudRate / 2) / BaudRate;  // 1302 at 115200

    // ---- Clocking ----
    logic sys_clk;  // 200 MHz from the differential input buffer
    logic mmcm_clk;  // 150 MHz from the MMCM, before the global buffer
    logic mmcm_feedback;  // MMCM feedback loop
    logic mmcm_locked;  // MMCM output is stable
    logic clk;  // 150 MHz system clock

    // Differential input buffer for the board clock.
    // svstyle: waive EXT-VLTR-MODMISSING Xilinx unisim primitive, compiled from unisims_ver
    IBUFDS u_sys_clk_buf (
        .I (sys_clk_p_i),
        .IB(sys_clk_n_i),
        .O (sys_clk)
    );

    // 200 MHz in, 150 MHz out: VCO = 200 x 6 = 1200 MHz, output = 1200 / 8.
    // svstyle: waive EXT-VLTR-MODMISSING Xilinx unisim primitive, compiled from unisims_ver
    MMCME2_BASE #(
        .CLKIN1_PERIOD   (5.000),
        .DIVCLK_DIVIDE   (1),
        .CLKFBOUT_MULT_F (6.000),
        .CLKOUT0_DIVIDE_F(8.000)
    ) u_mmcm (
        .CLKIN1   (sys_clk),
        .CLKFBIN  (mmcm_feedback),
        .CLKFBOUT (mmcm_feedback),
        .CLKFBOUTB(),
        .CLKOUT0  (mmcm_clk),
        .CLKOUT0B (),
        .CLKOUT1  (),
        .CLKOUT1B (),
        .CLKOUT2  (),
        .CLKOUT2B (),
        .CLKOUT3  (),
        .CLKOUT3B (),
        .CLKOUT4  (),
        .CLKOUT5  (),
        .CLKOUT6  (),
        .LOCKED   (mmcm_locked),
        .PWRDWN   (1'b0),
        .RST      (1'b0)
    );

    // Global clock buffer: drives the system clock network.
    // svstyle: waive EXT-VLTR-MODMISSING Xilinx unisim primitive, compiled from unisims_ver
    BUFG u_clk_buf (
        .I(mmcm_clk),
        .O(clk)
    );

    // ---- Reset ----
    logic rst_n;  // System reset, active-low, released on a clock edge

    // Hold the design in reset while the key is pressed or the MMCM is unlocked.
    reset_sync u_reset_sync (
        .clk_i       (clk),
        .rst_async_ni(reset_ni && mmcm_locked),
        .rst_no      (rst_n)
    );

    // ---- Host link ----
    logic       rx_valid;  // A byte has arrived from the host
    logic [7:0] rx_data;  // The byte
    logic       tx_send;  // Send tx_data to the host
    logic [7:0] tx_data;  // The byte to send
    logic       tx_busy;  // UART is sending

    // UART receiver from the host.
    uart_rx #(
        .ClksPerBit(ClksPerBit)
    ) u_uart_rx (
        .clk_i  (clk),
        .rst_ni (rst_n),
        .rx_i   (uart_rx_i),
        .valid_o(rx_valid),
        .data_o (rx_data)
    );

    // UART transmitter to the host.
    uart_tx #(
        .ClksPerBit(ClksPerBit)
    ) u_uart_tx (
        .clk_i (clk),
        .rst_ni(rst_n),
        .send_i(tx_send),
        .data_i(tx_data),
        .busy_o(tx_busy),
        .tx_o  (uart_tx_o)
    );

    logic                  start;  // Start a run
    logic [BatchWidth-1:0] batch_size;  // Images per batch
    logic                  busy;  // Run in progress
    logic                  done;  // Run finished
    logic [          31:0] cycle_count;  // Cycles the run took
    logic [          15:0] correct_count;  // Correct predictions
    logic [          31:0] checksum;  // Sum of all scores
    logic [ImageWidth-1:0] result_image;  // Image whose prediction is read
    logic [DigitWidth-1:0] result_digit;  // Its digit

    // Command handler: 'R' + batch size starts a run, then the results go back.
    host_if u_host_if (
        .clk_i          (clk),
        .rst_ni         (rst_n),
        .rx_valid_i     (rx_valid),
        .rx_data_i      (rx_data),
        .tx_send_o      (tx_send),
        .tx_data_o      (tx_data),
        .tx_busy_i      (tx_busy),
        .start_o        (start),
        .batch_size_o   (batch_size),
        .done_i         (done),
        .cycle_count_i  (cycle_count),
        .correct_count_i(correct_count),
        .checksum_i     (checksum),
        .result_image_o (result_image),
        .result_digit_i (result_digit)
    );

    // ---- Accelerator ----
    // The MLP accelerator: all test images, both layers, on chip.
    mlp_core u_mlp_core (
        .clk_i          (clk),
        .rst_ni         (rst_n),
        .start_i        (start),
        .batch_size_i   (batch_size),
        .busy_o         (busy),
        .done_o         (done),
        .cycle_count_o  (cycle_count),
        .correct_count_o(correct_count),
        .checksum_o     (checksum),
        .result_image_i (result_image),
        .result_digit_o (result_digit)
    );

    // ---- LEDs ----
    logic        results_ready_q;  // A run has finished since reset
    logic [26:0] heartbeat_q;  // Free-running counter; bit 26 toggles every 0.45 s

    // Results-ready flag and heartbeat counter.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            results_ready_q <= 1'b0;
            heartbeat_q     <= '0;
        end else begin
            heartbeat_q <= heartbeat_q + 27'd1;
            if (start) begin
                results_ready_q <= 1'b0;
            end else if (done) begin
                results_ready_q <= 1'b1;
            end
        end
    end

    // The LEDs light when their pin is low.
    assign led_no = ~{mmcm_locked, heartbeat_q[26], results_ready_q, busy};
endmodule

`resetall

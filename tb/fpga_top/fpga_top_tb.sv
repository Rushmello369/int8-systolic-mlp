// int8_mlp: INT8 MLP accelerator on a weight-stationary systolic array (Artix-7)
//
// Description:
//   End-to-end testbench for fpga_top.sv: the board clock, the MMCM, the
//   reset key, the UART and the host protocol, around the real accelerator.
//   It plays the host: drives the 200 MHz differential clock, presses and
//   releases RESET, then sends commands over a UART model and decodes the
//   reply (NumImages digits, correct count, checksum, cycle count).
//   The UART runs at BaudRate = 150 MHz / 16 to keep the simulation short;
//   the logic is the same as at 115200 baud, only ClksPerBit differs.
//   Runs, in order:
//     1. A stray byte, then 'R' with batch size 360: the stray byte must be
//        ignored and the reply must match the golden model.
//     2. 'R' with batch size 64: same predictions and summary, more cycles.
//   Needs the Xilinx primitive library: scripts\sim.ps1 fpga_top -Unisims
//   Prints one PASS or FAIL line at the end; FAIL exits non-zero.

`default_nettype none
`timescale 1ns / 1ps

module fpga_top_tb;
    import accel_pkg::*;

    localparam int unsigned ClockHz = 150_000_000;  // MMCM output in fpga_top
    localparam int unsigned ClksPerBit = 16;  // fast UART for simulation
    localparam int unsigned BaudRate = ClockHz / ClksPerBit;  // 9.375 Mbaud
    localparam realtime BitTime = 1.0e9 / BaudRate * 1ns;  // 106.67 ns per bit
    localparam int ReplyBytes = NumImages + 10;  // digits, then the summary
    localparam realtime ReplyTimeout = 50ms;  // generous: a run plus its reply is about 1 ms

    // ---- DUT signals ----
    logic       sys_clk_p_i = 1'b0;  // 200 MHz board clock
    logic       sys_clk_n_i;  // Its complement
    logic       reset_ni = 1'b0;  // RESET key, pressed at the start
    logic       uart_rx_i = 1'b1;  // Host to board, idle high
    logic       uart_tx_o;  // Board to host
    logic [3:0] led_no;  // LEDs, active-low

    assign sys_clk_n_i = ~sys_clk_p_i;

    // Device under test, with a fast UART
    fpga_top #(
        .ClockHz (ClockHz),
        .BaudRate(BaudRate)
    ) dut (
        .sys_clk_p_i(sys_clk_p_i),
        .sys_clk_n_i(sys_clk_n_i),
        .reset_ni   (reset_ni),
        .uart_rx_i  (uart_rx_i),
        .uart_tx_o  (uart_tx_o),
        .led_no     (led_no)
    );

    // svstyle: waive LNG-003 testbench clock generator, a delay loop rather than logic
    always #2.5ns sys_clk_p_i = ~sys_clk_p_i;

    // ---- Expected results ----
    logic [DigitWidth-1:0] expected_pred[NumImages];  // predicted digit per image
    logic [          31:0] expected_summary[2];  // correct count, checksum

    // ---- Bookkeeping ----
    int checks = 0;  // values compared so far
    int errors = 0;  // mismatches
    string first = "";  // the first mismatch, for the FAIL line

    task automatic fail(input string msg);
        if (errors == 0) begin
            first = msg;
        end
        errors++;
    endtask

    // Host UART transmitter: start bit, 8 data bits LSB first, stop bit.
    task automatic send_byte(input logic [7:0] data);
        uart_rx_i = 1'b0;
        #(BitTime);
        for (int bit_index = 0; bit_index < 8; bit_index++) begin
            uart_rx_i = data[bit_index];
            #(BitTime);
        end
        uart_rx_i = 1'b1;
        #(BitTime);
    endtask

    // Host UART receiver: wait for a start bit, sample each bit in its middle.
    task automatic receive_byte(output logic [7:0] data, output logic timed_out);
        timed_out = 1'b0;
        fork
            begin
                @(negedge uart_tx_o);
            end
            begin
                #(ReplyTimeout);
                timed_out = 1'b1;
            end
        join_any
        disable fork;
        if (timed_out) begin
            return;
        end
        #(BitTime * 1.5);
        for (int bit_index = 0; bit_index < 8; bit_index++) begin
            data[bit_index] = uart_tx_o;
            #(BitTime);
        end
        if (uart_tx_o !== 1'b1) begin
            fail("framing error: stop bit is 0");
        end
    endtask

    // Send 'R' and a batch size, then read and check the whole reply.
    task automatic run(input int batch_size);
        logic [7:0] reply[ReplyBytes];  // the board's reply
        logic timed_out;  // no start bit within ReplyTimeout
        logic [15:0] correct_count;  // decoded summary
        logic [31:0] checksum;  // decoded summary
        logic [31:0] cycle_count;  // decoded summary

        send_byte(8'h52);
        send_byte(8'(batch_size >> 8));
        send_byte(8'(batch_size));

        for (int index = 0; index < ReplyBytes; index++) begin
            receive_byte(reply[index], timed_out);
            if (timed_out) begin
                fail($sformatf("batch %0d: reply byte %0d never arrived", batch_size, index));
                return;
            end
        end

        for (int image = 0; image < NumImages; image++) begin
            checks++;
            if (reply[image] !== 8'(expected_pred[image])) begin
                fail($sformatf("batch %0d: image %0d reported %0d, want %0d", batch_size, image,
                               reply[image], expected_pred[image]));
            end
        end
        correct_count = {reply[NumImages], reply[NumImages+1]};
        checksum      = {reply[NumImages+2], reply[NumImages+3], reply[NumImages+4],
                         reply[NumImages+5]};
        cycle_count   = {reply[NumImages+6], reply[NumImages+7], reply[NumImages+8],
                         reply[NumImages+9]};
        checks += 2;
        if (32'(correct_count) !== expected_summary[0]) begin
            fail($sformatf("batch %0d: correct count %0d, want %0d", batch_size, correct_count,
                           expected_summary[0]));
        end
        if (checksum !== expected_summary[1]) begin
            fail($sformatf("batch %0d: checksum %h, want %h", batch_size, checksum,
                           expected_summary[1]));
        end
        $display("batch %3d over UART: %0d correct, checksum %h, %0d cycles", batch_size,
                 correct_count, checksum, cycle_count);
    endtask

    initial begin
        $readmemh("expected_pred.mem", expected_pred);
        $readmemh("expected_summary.mem", expected_summary);

        // Hold RESET, release it, and give the MMCM time to lock
        #200ns;
        reset_ni = 1'b1;
        wait (led_no[3] === 1'b0);  // LED4 on: MMCM locked
        #1us;

        // Run 1: a stray byte first, which must be ignored
        send_byte(8'h00);
        run(NumImages);

        // Run 2: smaller batches
        run(64);

        checks++;
        if (led_no[1] !== 1'b0) begin
            fail("LED2 (results ready) is off after a run");
        end

        if (errors != 0) begin
            $fatal(1, "FAIL: %0d of %0d checks, first: %s", errors, checks, first);
        end
        $display("PASS: %0d checks over UART", checks);
        $finish;
    end
endmodule

`resetall

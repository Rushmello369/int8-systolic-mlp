// int8_mlp: INT8 MLP accelerator on a weight-stationary systolic array (Artix-7)
//
// Description:
//   Simple dual-port RAM: one write port and one read port on the same
//   clock, written so that Vivado infers block RAM (or LUT RAM when small).
//   Read latency is one cycle: read_data_o shows the word at read_addr_i
//   from the previous cycle. Reading an address in the same cycle it is
//   written returns the old word.
//   InitFile, if not empty, names a hex file loaded at start with $readmemh
//   (one word per line). With write_en_i tied low the RAM is a ROM.
//   No reset: memory contents and read data are datapath.

// svstyle: waive LNG-014 Vivado xsim rejects `input logic` ports under `default_nettype none
module sdp_ram #(
    parameter int unsigned Width     = 8,             // Bits per word
    parameter int unsigned Depth     = 256,           // Words; at least 2
    parameter string       InitFile  = "",            // Hex file loaded at start, or "" for none
    parameter int unsigned AddrWidth = $clog2(Depth)  // Derived: do not override
) (
    input  logic                 clk_i,         // System clock
    input  logic                 write_en_i,    // Write write_data_i this cycle
    input  logic [AddrWidth-1:0] write_addr_i,  // Word to write
    input  logic [    Width-1:0] write_data_i,  // Data to write
    input  logic [AddrWidth-1:0] read_addr_i,   // Word to read
    output logic [    Width-1:0] read_data_o    // Word at last cycle's read_addr_i
);
    // $clog2(1) = 0 would give a zero-width address.
    if (Depth < 2) begin : gen_depth_check
        $error("Depth must be at least 2");
    end

    logic [Width-1:0] mem                                  [Depth];  // The storage array
    logic [Width-1:0] read_data_q;  // Registered read port

    // svstyle: waive SYN-006 Vivado synthesis loads $readmemh files into the RAM's initial contents
    initial begin
        if (InitFile != "") begin
            $readmemh(InitFile, mem);
        end
    end

    // One write and one registered read per cycle.
    always_ff @(posedge clk_i) begin
        if (write_en_i) begin
            mem[write_addr_i] <= write_data_i;
        end
        read_data_q <= mem[read_addr_i];
    end

    assign read_data_o = read_data_q;
endmodule

`resetall

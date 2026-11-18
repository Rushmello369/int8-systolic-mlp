# int8_mlp: Vivado non-project build of fpga_top, up to a routed design.
#
# Run from the project root (or use scripts\build.ps1):
#     vivado -mode batch -source fpga/scripts/build.tcl
#
# Steps: synthesis, optimisation, placement, physical optimisation, routing.
# It stops before write_bitstream on purpose: the bitstream is the next step,
# once the reports below have been reviewed.
# Writes:
#     fpga/reports/*.rpt     timing, utilisation, clocks, DRC, methodology (kept in git)
#     build/fpga/*.dcp       design checkpoints (ignored by git)
# Needs the .mem files from model/gen_vectors.py: they become the initial
# contents of the weight, image, label, bias and multiplier memories.

set root [file normalize [file join [file dirname [info script]] .. ..]]
set part xc7a200tfbg484-2
set top fpga_top
set report_dir [file join $root fpga reports]
set checkpoint_dir [file join $root build fpga]
file mkdir $report_dir
file mkdir $checkpoint_dir

# ---- Sources: the package first, then every other RTL file ----
read_verilog -sv [glob [file join $root rtl pkg *.sv]]
foreach dir {core ctrl io mem top} {
    read_verilog -sv [glob [file join $root rtl $dir *.sv]]
}

# Memory initialisation files, found by name by $readmemh during synthesis
foreach mem_file {weights images labels bias mult} {
    read_mem [file join $root model data $mem_file.mem]
}

read_xdc [file join $root fpga constraints ax7a200.xdc]

# ---- Synthesis ----
synth_design -top $top -part $part -include_dirs [file join $root rtl pkg]
write_checkpoint -force [file join $checkpoint_dir synth.dcp]
report_utilization -file [file join $report_dir synth_utilization.rpt]

# ---- Implementation ----
opt_design
place_design
phys_opt_design
route_design
write_checkpoint -force [file join $checkpoint_dir routed.dcp]

# ---- Reports on the routed design ----
report_timing_summary -max_paths 10 -report_unconstrained -file [file join $report_dir timing_summary.rpt]
report_utilization -file [file join $report_dir utilization.rpt]
report_utilization -hierarchical -file [file join $report_dir utilization_hierarchical.rpt]
report_clocks -file [file join $report_dir clocks.rpt]
report_drc -file [file join $report_dir drc.rpt]
report_methodology -file [file join $report_dir methodology.rpt]

# One line for the log: worst negative slack of the setup and hold checks
set wns [get_property SLACK [get_timing_paths -delay_type max]]
set whs [get_property SLACK [get_timing_paths -delay_type min]]
puts "BUILD DONE: WNS $wns ns, WHS $whs ns (bitstream not written)"

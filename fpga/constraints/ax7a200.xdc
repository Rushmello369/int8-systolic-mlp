## int8_mlp: INT8 MLP accelerator on a weight-stationary systolic array (Artix-7)
##
## Pin and timing constraints for fpga_top on the ALINX AX7A200 board.
## Pins are from the ALINX "AX7A200 user manual" REV 1.0:
##   200 MHz clock  sec. 2.3  (R4/T4, bank 34; bank 34 VCCO is 1.5 V for the DDR3)
##   USB-UART       sec. 3.8  (UART1_RXD = L14 into the FPGA, UART1_TXD = L15 out)
##   Keys           sec. 3.13 (RESET = F15, low when pressed)
##   LEDs           sec. 3.14 (LED1..4 = L13, M13, K14, K13, lit when low)
## Banks other than 34/35 run at 3.3 V on this board, hence LVCMOS33.

## ---- Clock: 200 MHz differential ----
set_property PACKAGE_PIN R4 [get_ports sys_clk_p_i]
set_property PACKAGE_PIN T4 [get_ports sys_clk_n_i]
set_property IOSTANDARD DIFF_SSTL15 [get_ports {sys_clk_p_i sys_clk_n_i}]
create_clock -name sys_clk -period 5.000 [get_ports sys_clk_p_i]
## The 150 MHz system clock is derived automatically from the MMCM settings.

## ---- Reset key ----
set_property PACKAGE_PIN F15 [get_ports reset_ni]
set_property IOSTANDARD LVCMOS33 [get_ports reset_ni]

## ---- USB-UART ----
set_property PACKAGE_PIN L14 [get_ports uart_rx_i]
set_property PACKAGE_PIN L15 [get_ports uart_tx_o]
set_property IOSTANDARD LVCMOS33 [get_ports {uart_rx_i uart_tx_o}]

## ---- User LEDs ----
set_property PACKAGE_PIN L13 [get_ports {led_no[0]}]
set_property PACKAGE_PIN M13 [get_ports {led_no[1]}]
set_property PACKAGE_PIN K14 [get_ports {led_no[2]}]
set_property PACKAGE_PIN K13 [get_ports {led_no[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led_no[*]}]

## ---- Timing exceptions ----
## The key and the UART input are asynchronous: they enter through synchronisers
## (reset_sync, and the two flops in uart_rx), so there is no path to time.
set_false_path -from [get_ports {reset_ni uart_rx_i}]
## The UART line (8.7 us per bit) and the LEDs are far slower than the clock.
set_false_path -to [get_ports {uart_tx_o led_no[*]}]

## ---- Configuration ----
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]

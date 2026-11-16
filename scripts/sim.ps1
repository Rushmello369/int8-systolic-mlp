# Run one testbench in the Vivado simulator (xsim).
#
# Usage, from the project root:
#     powershell -ExecutionPolicy Bypass -File scripts\sim.ps1 pe
#     powershell -ExecutionPolicy Bypass -File scripts\sim.ps1 pe_array -Generics Rows=3,Cols=5
#
# -Generics overrides parameters of the testbench's top module, comma-separated.
# Compiles every .sv file under rtl\ plus tb\<name>\<name>_tb.sv, then runs <name>_tb.
# Build files go to build\sim\<name>\ (ignored by git).
# Exits 0 only if the testbench printed PASS.

param(
    [Parameter(Mandatory = $true)][string]$Name,          # testbench folder, e.g. pe
    [string]$Generics = "",                               # e.g. Rows=3,Cols=5
    [string]$VivadoBin = "C:\AMDDesignTools\2025.2\Vivado\bin"
)

# Turn "Rows=3,Cols=5" into: --generic_top Rows=3 --generic_top Cols=5
$genericArgs = @()
if ($Generics -ne "") {
    foreach ($generic in $Generics.Split(",")) {
        $genericArgs += "--generic_top"
        $genericArgs += "`"$generic`""   # quoted: xelab.bat would split at the '='
    }
}

$root = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $root "build\sim\$Name"
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

$rtlFiles = Get-ChildItem -Path (Join-Path $root "rtl") -Recurse -Filter *.sv | ForEach-Object { $_.FullName }
$tbFile = Join-Path $root "tb\$Name\${Name}_tb.sv"
$top = "${Name}_tb"

Push-Location $buildDir
try {
    # 1. Compile the SystemVerilog sources
    & "$VivadoBin\xvlog.bat" --sv $rtlFiles $tbFile
    if ($LASTEXITCODE -ne 0) { throw "xvlog failed" }

    # 2. Elaborate: connect the modules into one simulation snapshot
    & "$VivadoBin\xelab.bat" $top -s $top -timescale 1ns/1ps @genericArgs
    if ($LASTEXITCODE -ne 0) { throw "xelab failed" }

    # 3. Run the simulation to the end
    & "$VivadoBin\xsim.bat" $top -R | Tee-Object -Variable simOutput
    if (-not ($simOutput -match "^PASS")) { throw "testbench did not pass" }
}
finally {
    Pop-Location
}

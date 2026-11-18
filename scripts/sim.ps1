# Run one testbench in the Vivado simulator (xsim).
#
# Usage, from the project root:
#     powershell -ExecutionPolicy Bypass -File scripts\sim.ps1 pe
#     powershell -ExecutionPolicy Bypass -File scripts\sim.ps1 pe_array -Generics Rows=3,Cols=5
#     powershell -ExecutionPolicy Bypass -File scripts\sim.ps1 mlp_core -Generics OnlyBatch=64
#
# -Generics overrides parameters of the testbench's top module, comma-separated.
# (Plus-args such as +seed=N cannot be passed: xsim.bat splits arguments at '='.)
# -Unisims links the Xilinx primitive library (MMCM, BUFG); needed for fpga_top only.
# Compiles rtl\pkg first, then every other .sv file under rtl\, plus tb\<name>\<name>_tb.sv,
# then runs <name>_tb. The model\data\*.mem files (from model\gen_vectors.py) are copied
# next to the simulation so $readmemh finds them by name.
# Build files go to build\sim\<name>\ (ignored by git).
# Exits 0 only if the testbench printed PASS.

param(
    [Parameter(Mandatory = $true)][string]$Name,          # testbench folder, e.g. pe
    [string]$Generics = "",                               # e.g. Rows=3,Cols=5
    [switch]$Unisims,                                     # design uses Xilinx primitives (fpga_top)
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

# Packages must be compiled before the modules that import them
$pkgDir = Join-Path $root "rtl\pkg"
$pkgFiles = Get-ChildItem -Path $pkgDir -Filter *.sv | ForEach-Object { $_.FullName }
$otherFiles = Get-ChildItem -Path (Join-Path $root "rtl") -Recurse -Filter *.sv |
    Where-Object { $_.DirectoryName -ne $pkgDir } | ForEach-Object { $_.FullName }
$tbFile = Join-Path $root "tb\$Name\${Name}_tb.sv"
$top = "${Name}_tb"

# glbl drives the global set/reset of Xilinx primitives (MMCM, BUFG) in fpga_top
$glblFile = Join-Path (Split-Path -Parent $VivadoBin) "data\verilog\src\glbl.v"

# Memory initialisation files, read by name from the simulation directory
$memFiles = Get-ChildItem -Path (Join-Path $root "model\data") -Filter *.mem -ErrorAction SilentlyContinue
foreach ($memFile in $memFiles) {
    Copy-Item $memFile.FullName $buildDir
}

Push-Location $buildDir
try {
    # 1. Compile the SystemVerilog sources (rtl\pkg holds the include files too)
    & "$VivadoBin\xvlog.bat" --sv -i $pkgDir $pkgFiles $otherFiles $tbFile
    if ($LASTEXITCODE -ne 0) { throw "xvlog failed" }
    $tops = @($top)
    if ($Unisims) {
        & "$VivadoBin\xvlog.bat" $glblFile
        if ($LASTEXITCODE -ne 0) { throw "xvlog failed on glbl.v" }
        $tops += "glbl"
    }

    # 2. Elaborate: connect the modules into one simulation snapshot
    & "$VivadoBin\xelab.bat" @tops -s $top -timescale 1ns/1ps -L unisims_ver @genericArgs
    if ($LASTEXITCODE -ne 0) { throw "xelab failed" }

    # 3. Run the simulation to the end
    & "$VivadoBin\xsim.bat" $top -R | Tee-Object -Variable simOutput
    if (-not ($simOutput -match "^PASS")) { throw "testbench did not pass" }
}
finally {
    Pop-Location
}

# Build fpga_top in Vivado up to a routed design (no bitstream).
#
# Usage, from the project root:
#     powershell -ExecutionPolicy Bypass -File scripts\build.ps1
#
# Regenerates the .mem files first, so the memories hold the current model.
# Reports go to fpga\reports\, the Vivado log and checkpoints to build\fpga\.

param(
    [string]$VivadoBin = "C:\AMDDesignTools\2025.2\Vivado\bin"
)

$root = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $root "build\fpga"
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

# 1. Memory files from the integer model
& (Join-Path $root ".venv\Scripts\python.exe") (Join-Path $root "model\gen_vectors.py")
if ($LASTEXITCODE -ne 0) { throw "gen_vectors.py failed" }

# 2. Vivado in batch mode; the log and journal stay in build\fpga
Push-Location $buildDir
try {
    & "$VivadoBin\vivado.bat" -mode batch -nojournal -log vivado.log -source (Join-Path $root "fpga\scripts\build.tcl")
    if ($LASTEXITCODE -ne 0) { throw "Vivado build failed" }
}
finally {
    Pop-Location
}

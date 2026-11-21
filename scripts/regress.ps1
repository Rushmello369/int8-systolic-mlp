# Run every verification level, bottom up, and stop at the first failure.
#
# Usage, from the project root:
#     powershell -ExecutionPolicy Bypass -File scripts\regress.ps1
#
# Levels, as in docs/project_plan.md:
#   0. Python: golden-model unit tests, then regenerate the .mem files
#   1. PE: exhaustive INT8 x INT8
#   2. Array, skew buffer, systolic core: random tiles, several shapes
#   3. Requantiser and argmax: boundaries, ties, random
#   5. Full model: every test image, every value, five batch sizes
#   +  Board level: the same through the MMCM, UART and host protocol
# Exits non-zero if any level fails. Takes about 10 minutes.

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$sim = Join-Path $root "scripts\sim.ps1"

# Each run: testbench name, generics (or ""), and whether it needs the Xilinx primitives
$runs = @(
    @("pe", "", $false),
    @("pe_array", "Rows=8,Cols=8", $false),
    @("pe_array", "Rows=3,Cols=5", $false),
    @("skew_buffer", "Lanes=8,Width=8", $false),
    @("skew_buffer", "Lanes=8,Width=24", $false),
    @("skew_buffer", "Lanes=1,Width=8", $false),
    @("systolic_core", "Rows=8,Cols=8", $false),
    @("systolic_core", "Rows=3,Cols=5", $false),
    @("requant", "", $false),
    @("argmax", "", $false),
    @("mlp_core", "", $false),
    @("fpga_top", "", $true)
)

Push-Location $root
try {
    & $python -m pytest model -q
    if ($LASTEXITCODE -ne 0) { throw "Python tests failed" }
    & $python (Join-Path $root "model\gen_vectors.py")
    if ($LASTEXITCODE -ne 0) { throw "gen_vectors.py failed" }
    # The host tests read the .mem files, so they run after gen_vectors.py.
    & $python -m pytest host -q
    if ($LASTEXITCODE -ne 0) { throw "Host script tests failed" }

    foreach ($run in $runs) {
        $name = $run[0]
        $generics = $run[1]
        $simArgs = @("-ExecutionPolicy", "Bypass", "-File", $sim, $name)
        if ($generics -ne "") { $simArgs += @("-Generics", $generics) }
        if ($run[2]) { $simArgs += "-Unisims" }

        Write-Host "==== $name $generics"
        $output = powershell @simArgs 2>&1
        $output | Select-String -Pattern "^PASS|FAIL|^batch|Error:" | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE -ne 0) { throw "$name $generics failed" }
    }
    Write-Host "REGRESSION PASSED: $($runs.Count) simulations"
}
finally {
    Pop-Location
}

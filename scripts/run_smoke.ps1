param(
    [string]$DataRoot = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot

if ($DataRoot) {
    python -m src.run_experiments --task task2 --mode core --feature-type provided_engineered --model-key logistic_regression --quick --max-rows 120 --cv 3 --data-root $DataRoot --output-dir "outputs/smoke"
} else {
    python -m src.synthetic_smoke
}
if ($LASTEXITCODE -ne 0) { throw "Smoke check failed." }

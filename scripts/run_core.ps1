param(
    [string]$DataRoot = ".\data"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot

python -m src.run_experiments --task task1 --mode core --cv 3 --data-root $DataRoot
if ($LASTEXITCODE -ne 0) { throw "Task 1 failed." }
python -m src.run_experiments --task task2 --mode core --cv 5 --data-root $DataRoot
if ($LASTEXITCODE -ne 0) { throw "Task 2 failed." }

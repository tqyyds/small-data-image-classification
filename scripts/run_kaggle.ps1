param(
    [string]$DataRoot = ".\data"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot

python -m src.run_experiments --task task1 --mode kaggle --cv 3 --device auto --data-root $DataRoot
if ($LASTEXITCODE -ne 0) { throw "Task 1 failed." }
python -m src.run_experiments --task task2 --mode kaggle --cv 5 --device auto --data-root $DataRoot
if ($LASTEXITCODE -ne 0) { throw "Task 2 failed." }
python -m src.run_experiments --task task1 --mode kaggle --feature-type efficientnet_b0_plus_provided --include-random-forest --model-key random_forest --cv 3 --device auto --data-root $DataRoot
python -m src.run_experiments --task task2 --mode kaggle --feature-type cnn_plus_provided --include-random-forest --model-key random_forest --model-key svm_random_forest --cv 5 --device auto --data-root $DataRoot
python -m src.make_submissions --task task1 --feature-type efficientnet_b0 --model-key logistic_regression --transform-mode official --cv 3 --device auto --data-root $DataRoot
python -m src.make_submissions --task task2 --feature-type cnn_plus_provided --model-key random_forest --transform-mode official --cv 5 --device auto --data-root $DataRoot

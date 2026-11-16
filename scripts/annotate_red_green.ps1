$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$so101 = Split-Path -Parent $repo
$lelab = Join-Path $so101 'sources/lelab-main'
$python = Join-Path $lelab '.venv/Scripts/python.exe'
$dataset = Join-Path $lelab '.cache/huggingface/lerobot/BlossomsGarden/pick-and-place-v3_20261025_051506'
$output = Join-Path $repo 'annotations_red_green_alignment_30.json'
$sheets = Join-Path $repo 'annotation_sheets_alignment_30'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "LeLab Python not found: $python" }
if (-not (Test-Path -LiteralPath $dataset -PathType Container)) { throw "Dataset not found: $dataset" }

Set-Location -LiteralPath $repo
Write-Host 'Annotating red-on-green placement alignment: episodes 0-29.'
Write-Host "JSON: $output"
& $python (Join-Path $PSScriptRoot 'annotate_dataset.py') `
    --dataset_path $dataset `
    --annotations_out $output `
    --camera observation.images.front `
    --sheet_dir $sheets
if ($LASTEXITCODE -ne 0) { throw "Annotator exited with code $LASTEXITCODE" }

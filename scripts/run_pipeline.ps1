# Full offline pipeline on Windows PowerShell:  .\scripts\run_pipeline.ps1
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
python -m frauddet.data;      if (-not $?) { exit 1 }
python -m frauddet.research;  if (-not $?) { exit 1 }
python -m frauddet.optimize;  if (-not $?) { exit 1 }
python -m frauddet.report;    if (-not $?) { exit 1 }
Write-Host "Done. Next: docker compose up -d db; python -m frauddet.db init; python -m frauddet.db load-research"

# Live demo on Windows: PostgreSQL + scoring API + stream replay + Streamlit (+ Power BI when installed).
#   .\scripts\run_live.ps1 [-Tps 25] [-NoPowerBI]
param([double]$Tps = 25, [switch]$NoPowerBI)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = "src"

docker compose up -d db
Start-Sleep -Seconds 3
python -m frauddet.db init
python -m frauddet.db reset            # fresh demo: stream restarts at the beginning of the test period
python -m frauddet.db load-research    # r_* tables for the Research page

Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root'; `$env:PYTHONPATH='src'; python -m frauddet.serving.api"
Start-Sleep -Seconds 8
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root'; `$env:PYTHONPATH='src'; python -u -m frauddet.stream.replay --tps $Tps --label-delay 20"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root'; streamlit run dashboard/streamlit/app.py"

$pbi = "C:\Program Files\Microsoft Power BI Desktop\bin\PBIDesktop.exe"
if (-not $NoPowerBI -and (Test-Path $pbi)) {
    Start-Process -FilePath $pbi -ArgumentList "`"$root\dashboard\powerbi\FraudOps.pbip`""
    Write-Host "Power BI: login database pertama kali -> fraud / fraud"
}
Write-Host "API http://127.0.0.1:8010/docs  |  Streamlit http://localhost:8501"

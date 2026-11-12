# Live demo on Windows: DB + API + stream replay + Streamlit, each in its own window.
#   .\scripts\run_live.ps1 [-Tps 40]
param([double]$Tps = 40)
$env:PYTHONPATH = "src"
docker compose up -d db
python -m frauddet.db init
python -m frauddet.db reset
python -m frauddet.db load-research
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$env:PYTHONPATH='src'; python -m frauddet.serving.api"
Start-Sleep -Seconds 6
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$env:PYTHONPATH='src'; python -m frauddet.stream.replay --tps $Tps --label-delay 20"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "streamlit run dashboard/streamlit/app.py"
Write-Host "API http://127.0.0.1:8010/docs  |  Streamlit http://localhost:8501  |  Power BI: dashboard/powerbi/README.md"

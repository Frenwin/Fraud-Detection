# Linux/macOS/Git-Bash. Windows PowerShell: scripts/run_pipeline.ps1
PY ?= python
export PYTHONPATH := src

.PHONY: all data research optimize report db api replay loadtest dashboard test clean

all: data research optimize report          ## full offline research pipeline (~25 min)

data:      ; $(PY) -m frauddet.data          ## simulate + feature replay + state snapshot
research:  ; $(PY) -m frauddet.research      ## E1-E8
optimize:  ; $(PY) -m frauddet.optimize      ## E9-E10 + production bundle
report:    ; $(PY) -m frauddet.report        ## reports/RESEARCH_REPORT.md
db:        ; docker compose up -d db && $(PY) -m frauddet.db init && $(PY) -m frauddet.db load-research
api:       ; $(PY) -m frauddet.serving.api
replay:    ; $(PY) -m frauddet.stream.replay --tps 40 --label-delay 20
loadtest:  ; FRAUDDET_SINK=off $(PY) -m frauddet.loadtest
dashboard: ; streamlit run dashboard/streamlit/app.py
test:      ; $(PY) -m pytest -q
clean:     ; rm -rf data artifacts catboost_info

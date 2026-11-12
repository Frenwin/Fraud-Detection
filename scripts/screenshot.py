"""Capture dashboard screenshots for the README (uses the locally installed Chrome).
    python scripts/screenshot.py [url] [out_dir]
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/img")
out.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    pg = b.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1)
    pg.goto(url)
    pg.wait_for_selector("[data-testid='stMetricValue']", timeout=60000)
    pg.wait_for_timeout(5000)
    pg.screenshot(path=str(out / "streamlit_live.png"), full_page=True)
    for i, name in ((1, "streamlit_health.png"), (2, "streamlit_research.png")):
        pg.get_by_role("tab").nth(i).click()
        pg.wait_for_timeout(5000)
        pg.screenshot(path=str(out / name), full_page=True)
    b.close()
print("saved to", out)

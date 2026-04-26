"""capture_thumbs.py — Capture map thumbnails for product cards.

Uses Playwright to headlessly render each data_products HTML file and
save a cropped screenshot of just the map area as a JPEG thumbnail.

Run once:  python capture_thumbs.py
Requires:  pip install playwright && playwright install chromium
"""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

HERE = Path(__file__).parent
OUT  = HERE / "images" / "thumbs"
OUT.mkdir(parents=True, exist_ok=True)

# (filename, map_element_selector, output_name)
TARGETS = [
    ("data_products/spatial_risk.html",  "#map",    "thumb_spatial.jpg"),
    ("data_products/dynamic_risk.html",  "#map",    "thumb_dynamic.jpg"),
    ("data_products/sunglare.html",      "#map",    "thumb_sunglare.jpg"),
    ("data_products/absolute_risk.html", "#map",    "thumb_mvkt.jpg"),
    ("data_products/whatif.html",        "#wi-map", "thumb_whatif.jpg"),
    ("data_products/intervention.html",  "#map",    "thumb_intervention.jpg"),
]

# Wait this long (ms) after page load for tiles + JS render to settle
SETTLE_MS = 4000


async def capture(pw, path: Path, selector: str, out: Path):
    browser = await pw.chromium.launch()
    page    = await browser.new_page(viewport={"width": 1280, "height": 800})
    await page.goto(path.as_uri(), wait_until="networkidle")
    await page.wait_for_timeout(SETTLE_MS)

    el = page.locator(selector).first
    await el.screenshot(path=str(out), type="jpeg", quality=88)
    await browser.close()
    print(f"  ✓  {out.name}")


async def main():
    async with async_playwright() as pw:
        for html_name, selector, thumb_name in TARGETS:
            html_path = HERE / html_name
            if not html_path.exists():
                print(f"  ⚠  skipped (not found): {html_name}")
                continue
            out_path = OUT / thumb_name
            print(f"  → capturing {html_name} …")
            try:
                await capture(pw, html_path, selector, out_path)
            except Exception as e:
                print(f"  ✗  {html_name}: {e}")


if __name__ == "__main__":
    asyncio.run(main())

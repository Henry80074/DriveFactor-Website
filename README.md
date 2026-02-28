# DriveFactor-Website

Static marketing site for DriveFactor (Predict / React / Coach) plus a sample driver report.

## Structure

- `index.html` — home page
- `predict/index.html` — Predict product page
- `react/index.html` — React product page
- `coach/index.html` — Coach product page
- `coach/driver-report/index.html` — sample driver report
- `styles.css` — shared site styling tokens + layout helpers
- `report-styles.css` — driver report specific styling
- `report-script.js` — shared UI + chart initialization helpers
- `advanced.html`, `crash_density.html` — embedded Leaflet map sources

## Local preview

From the repo root, run a static server and open the URL in your browser:

```bash
python3 -m http.server 8081
```

Then visit:

- `http://localhost:8081/`
- `http://localhost:8081/predict/`
- `http://localhost:8081/react/`
- `http://localhost:8081/coach/`
- `http://localhost:8081/coach/driver-report/`

## GitHub Pages notes

All internal links and asset references are relative so the site works when hosted at a project subpath (for example: `https://username.github.io/DriveFactor-Website/`).

Redirect pages (`predict.html`, `react.html`, `coach.html`, `coach/driver-report/driver-report.html`) are kept for backward compatibility.

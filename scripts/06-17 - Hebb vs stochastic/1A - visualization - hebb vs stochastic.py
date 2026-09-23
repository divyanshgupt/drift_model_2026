### This script is Claude generated.
### It generates an interactive HTML visualization for the Hebb vs Stochastic simulation results, allowing users to explore the drift differences between baseline and CNO conditions across different Hebb_k and Eta_k parameter combinations.
### The baseline/CNO drift_metrics.png plots are embedded directly in the HTML (base64), so the output file is fully self-contained and can be shared on its own.

import base64
import json

import numpy as np
import h5py
import plotly.graph_objects as go

save_loc_general = "../../results/06-17 - hebb vs stochastic/1A. F to E plastic - co-tuned inh - hebb vs stochastic/"

hebb_range = np.round(np.arange(0, 1.01, 0.1), 1)
eta_range = np.round(np.arange(0, 1.01, 0.1), 1)

# --- Load final-day mean drift magnitude for every (hebb_k, eta_k) pair, and
#     embed each condition's drift_metrics.png as a base64 data URI so the
#     resulting HTML has no external file dependencies. ---
diff_matrix = np.full((len(hebb_range), len(eta_range)), np.nan)
embedded_images = {}

for i, hebb_k in enumerate(hebb_range):
    for j, eta_k in enumerate(eta_range):
        baseline_dir = save_loc_general + f"baseline hebb_{hebb_k:.1f}_eta_{eta_k:.1f}/"
        cno_dir = save_loc_general + f"cno hebb_{hebb_k:.1f}_eta_{eta_k:.1f}/"

        with h5py.File(baseline_dir + "results.h5", "r") as f:
            baseline_final = f["drift_mag"][-1].mean()
        with h5py.File(cno_dir + "results.h5", "r") as f:
            cno_final = f["drift_mag"][-1].mean()

        diff_matrix[i, j] = cno_final - baseline_final

        key = f"{hebb_k:.1f}_{eta_k:.1f}"
        with open(baseline_dir + "drift_metrics.png", "rb") as f:
            baseline_b64 = base64.b64encode(f.read()).decode("ascii")
        with open(cno_dir + "drift_metrics.png", "rb") as f:
            cno_b64 = base64.b64encode(f.read()).decode("ascii")
        embedded_images[key] = {
            "baseline": "data:image/png;base64," + baseline_b64,
            "cno": "data:image/png;base64," + cno_b64,
        }

# --- Heatmap figure ---
fig = go.Figure(data=go.Heatmap(
    z=diff_matrix,
    x=[f"{v:.1f}" for v in eta_range],
    y=[f"{v:.1f}" for v in hebb_range],
    colorscale="RdBu",
    reversescale=True,
    zmid=0,
    colorbar=dict(title="CNO - Baseline<br>drift (deg)"),
    hovertemplate="Eta_k=%{x}<br>Hebb_k=%{y}<br>Diff=%{z:.3f}°<extra></extra>",
))
fig.update_layout(
    title="Drift Difference (CNO - Baseline)",
    xaxis_title="Eta_k",
    yaxis_title="Hebb_k",
    width=520,
    height=520,
    template="plotly_white",
)
# fixedrange (not dragmode=False) is what disables zoom/pan here -- setting
# dragmode=False also removes Plotly's click-detection layer, which silently
# breaks plotly_click. fixedrange blocks range changes per-axis without
# touching click handling.
fig.update_xaxes(fixedrange=True)
fig.update_yaxes(fixedrange=True)

heatmap_div = fig.to_html(
    full_html=False,
    include_plotlyjs=True,
    div_id="heatmapDiv",
    config={"displayModeBar": False, "scrollZoom": False},
)
embedded_images_json = json.dumps(embedded_images)

# --- Full page: heatmap + click-driven baseline/CNO drift metric plots ---
html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Hebb vs Stochastic - Drift Visualization</title>
<style>
  body {{ font-family: -apple-system, Arial, sans-serif; margin: 24px 32px 64px; color: #1a1a1a; }}
  .plots-row {{ display: flex; gap: 24px; flex-wrap: wrap; margin-top: 24px; }}
  .plot-card {{ flex: 1 1 480px; min-width: 320px; border: 1px solid #ddd; border-radius: 12px; padding: 16px; }}
  .plot-card h3.baseline {{ color: #2563eb; margin-top: 0; }}
  .plot-card h3.cno {{ color: #dc2626; margin-top: 0; }}
  .plot-card img {{ width: 100%; height: auto; border-radius: 6px; }}
  #readout {{ margin-top: 12px; color: #555; font-size: 0.9rem; }}
</style>
</head>
<body>

<h1>Hebb vs Stochastic — F to E plastic, co-tuned inh</h1>
<p>Click a heatmap cell to load the baseline &amp; CNO drift metric plots for that (Hebb_k, Eta_k) pair. This file is self-contained — all plots are embedded, so it can be shared as a single .html.</p>

{heatmap_div}

<div id="readout">Click a cell above to select a (Hebb_k, Eta_k) pair.</div>

<div class="plots-row">
  <div class="plot-card">
    <h3 class="baseline">Baseline</h3>
    <img id="baselineImg" src="" onerror="this.style.display='none'">
  </div>
  <div class="plot-card">
    <h3 class="cno">CNO</h3>
    <img id="cnoImg" src="" onerror="this.style.display='none'">
  </div>
</div>

<script>
  var embeddedImages = {embedded_images_json};
  var heatmapDiv = document.getElementById('heatmapDiv');
  heatmapDiv.on('plotly_click', function(data) {{
    var pt = data.points[0];
    // Plotly may hand back pt.x/pt.y as numbers rather than the exact
    // "0.0".."1.0" strings (e.g. 0 instead of "0.0"), so normalize before
    // building the lookup key.
    var eta = parseFloat(pt.x).toFixed(1);
    var hebb = parseFloat(pt.y).toFixed(1);
    var key = hebb + '_' + eta;
    var entry = embeddedImages[key];
    var baselineImg = document.getElementById('baselineImg');
    var cnoImg = document.getElementById('cnoImg');
    if (entry) {{
      baselineImg.style.display = '';
      cnoImg.style.display = '';
      baselineImg.src = entry.baseline;
      cnoImg.src = entry.cno;
    }}
    document.getElementById('readout').innerHTML =
      'Selected: <b>Hebb_k = ' + hebb + '</b>, <b>Eta_k = ' + eta +
      '</b> — drift difference (CNO − baseline) = <b>' + pt.z.toFixed(3) + '°</b>';
  }});
</script>

</body>
</html>
"""

out_path = save_loc_general + "interactive_visualization_plotly_standalone.html"
with open(out_path, "w") as f:
    f.write(html)

print(f"Saved self-contained interactive visualization to {out_path}")
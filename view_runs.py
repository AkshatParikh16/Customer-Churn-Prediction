"""
Standalone run viewer — opens a simple HTML page in your browser showing
all MLflow runs from the SQLite database. No MLflow UI needed.
Run: python view_runs.py
"""
import os, sys, sqlite3, json, webbrowser, http.server, threading
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
DB_PATH  = ROOT / "mlruns" / "mlflow.db"
PORT     = 7860


def read_runs_from_sqlite(db: Path) -> list[dict]:
    if not db.exists():
        return []
    try:
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row

        # Get experiments
        try:
            exps = {r["experiment_id"]: r["name"]
                    for r in conn.execute("SELECT experiment_id, name FROM experiments")}
        except Exception:
            exps = {}

        # Get runs + metrics
        # Get column names so we handle both old (run_name) and new (name) schemas
        cols = [c[1] for c in conn.execute("PRAGMA table_info(runs)").fetchall()]
        name_col = "name" if "name" in cols else "run_name"
        try:
            runs_raw = conn.execute(f"""
                SELECT run_uuid, experiment_id, {name_col} as run_name, status,
                       start_time, end_time
                FROM runs ORDER BY start_time DESC
            """).fetchall()
        except Exception as e:
            print(f"  runs query failed: {e}")
            runs_raw = []

        results = []
        for r in runs_raw:
            rid = r["run_uuid"]
            try:
                metrics = {m["key"]: round(m["value"], 4)
                           for m in conn.execute(
                               "SELECT key, value FROM latest_metrics WHERE run_uuid=?", (rid,))}
            except Exception:
                metrics = {}
            try:
                params = {p["key"]: p["value"]
                          for p in conn.execute(
                              "SELECT key, value FROM params WHERE run_uuid=?", (rid,))
                          if p["key"] in ("model", "decision_threshold", "optuna_trials")}
            except Exception:
                params = {}
            results.append({
                "run_id":    rid[:8],
                "name":      r["run_name"] or "—",
                "exp":       exps.get(r["experiment_id"], r["experiment_id"]),
                "status":    r["status"],
                "metrics":   metrics,
                "params":    params,
            })
        conn.close()
        return results
    except Exception as e:
        return [{"error": str(e)}]


def read_runs_from_file_store(mlruns: Path) -> list[dict]:
    """Fallback: read from old file-based mlruns directories."""
    results = []
    for exp_dir in sorted(mlruns.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name in ("models", "artifacts"):
            continue
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name == "models":
                continue
            metrics = {}
            metrics_dir = run_dir / "metrics"
            if metrics_dir.exists():
                for mf in metrics_dir.iterdir():
                    try:
                        val = mf.read_text().strip().split()[-2]
                        metrics[mf.name] = round(float(val), 4)
                    except Exception:
                        pass
            tags_dir = run_dir / "tags"
            run_name = "—"
            if tags_dir.exists():
                name_file = tags_dir / "mlflow.runName"
                if name_file.exists():
                    run_name = name_file.read_text().strip()
            results.append({
                "run_id":  run_dir.name[:8],
                "name":    run_name,
                "exp":     exp_dir.name,
                "status":  "FINISHED",
                "metrics": metrics,
                "params":  {},
            })
    return results


def make_html(runs: list[dict], source: str) -> str:
    key_metrics = ["val/roc_auc", "val/recall", "val/pr_auc", "val/f1",
                   "val/expected_savings", "best_threshold"]

    rows = ""
    for r in runs:
        if "error" in r:
            rows += f"<tr><td colspan='10' style='color:red'>{r['error']}</td></tr>"
            continue
        m = r.get("metrics", {})
        def mv(k):
            v = m.get(k, "")
            return f"{v:.4f}" if isinstance(v, float) else str(v)
        status_color = "#2ecc71" if r["status"] == "FINISHED" else "#f39c12"
        rows += f"""<tr>
          <td>{r['name']}</td>
          <td>{r['exp']}</td>
          <td style='color:{status_color}'>{r['status']}</td>
          <td><b>{mv('val/roc_auc')}</b></td>
          <td>{mv('val/recall')}</td>
          <td>{mv('val/pr_auc')}</td>
          <td>{mv('val/f1')}</td>
          <td>{mv('val/expected_savings')}</td>
          <td>{mv('best_threshold')}</td>
          <td style='font-size:11px'>{r['run_id']}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html><head>
<meta charset='utf-8'>
<title>Churn Model Runs</title>
<style>
  body  {{ font-family: Arial, sans-serif; background: #1a1a2e; color: #eee; margin: 20px; }}
  h1   {{ color: #00d4aa; }}
  .src {{ color: #aaa; font-size: 13px; margin-bottom: 16px; }}
  table{{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th   {{ background: #16213e; color: #00d4aa; padding: 10px 12px; text-align: left; }}
  td   {{ padding: 8px 12px; border-bottom: 1px solid #2a2a4a; }}
  tr:hover {{ background: #0f3460; }}
  b    {{ color: #00d4aa; }}
</style>
</head><body>
<h1>Customer Churn — Model Runs</h1>
<div class='src'>Source: {source} &nbsp;|&nbsp; {len(runs)} runs &nbsp;|&nbsp;
<a href='javascript:location.reload()' style='color:#00d4aa'>Refresh</a></div>
<table>
<tr>
  <th>Run Name</th><th>Experiment</th><th>Status</th>
  <th>ROC-AUC</th><th>Recall</th><th>PR-AUC</th><th>F1</th>
  <th>Exp Savings $</th><th>Threshold</th><th>Run ID</th>
</tr>
{rows if rows else "<tr><td colspan='10'>No runs found — training may still be starting</td></tr>"}
</table>
</body></html>"""


# ── Try SQLite first, fall back to file store ────────────────────────────────
mlruns = ROOT / "mlruns"
if DB_PATH.exists():
    print(f"DB found: {DB_PATH}  ({DB_PATH.stat().st_size:,} bytes)")
    runs  = read_runs_from_sqlite(DB_PATH)
    source = f"SQLite ({DB_PATH.name})"
    print(f"Runs found: {len(runs)}")
    for r in runs[:5]:
        print(f"  - {r.get('name','?')}  ROC-AUC={r.get('metrics',{}).get('val/roc_auc','?')}")
elif mlruns.exists():
    runs  = read_runs_from_file_store(mlruns)
    source = f"File store ({mlruns})"
    print(f"Reading from file store: {mlruns}  ({len(runs)} runs)")
else:
    runs   = []
    source = "No tracking store found"
    print("No mlruns directory found — run training first")

html = make_html(runs, source)

# ── Serve it ─────────────────────────────────────────────────────────────────
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):                               # noqa: N802
        # Re-read runs on every refresh so live Optuna runs appear
        if DB_PATH.exists():
            fresh = read_runs_from_sqlite(DB_PATH)
            src   = f"SQLite ({DB_PATH.name})"
        elif mlruns.exists():
            fresh = read_runs_from_file_store(mlruns)
            src   = "File store"
        else:
            fresh, src = [], "none"
        content = make_html(fresh, src).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)
    def log_message(self, *args):                   # silence access logs
        pass

print(f"\nStarting viewer on http://127.0.0.1:{PORT}")
print("Refresh the browser tab to see new Optuna runs appear live.")
print("Press Ctrl+C to stop.\n")

threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
with http.server.HTTPServer(("127.0.0.1", PORT), Handler) as srv:
    srv.serve_forever()

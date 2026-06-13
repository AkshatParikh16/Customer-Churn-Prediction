"""
MLflow URI diagnostic — run from project root with venv activated.
python debug_mlflow.py
"""
import json, time, os, sys
from pathlib import Path

# Write immediately — absolute path to avoid any CWD issues
LOG = Path(__file__).resolve().parent / "debug-1ecc3f.log"


def w(hyp, msg, data):
    entry = json.dumps({"sessionId": "1ecc3f", "runId": "run2", "hypothesisId": hyp,
                        "message": msg, "data": data, "timestamp": int(time.time() * 1000)})
    LOG.open("a").write(entry + "\n")
    print(f"[{hyp}] {msg}: {data}")


print(f"Log path: {LOG}")
w("INIT", "script started", {"cwd": os.getcwd(), "log": str(LOG)})

# Check what tracking URI settings produces
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
try:
    from configs.settings import settings
    uri = settings.mlflow.tracking_uri
    w("URI", "tracking_uri from settings", {"uri": uri, "has_backslash": "\\" in uri})
except Exception as e:
    w("URI", "settings load failed", {"error": str(e)})
    uri = None

# Check if the db file exists
db_path = Path(__file__).resolve().parent / "mlruns" / "mlflow.db"
w("DB", "mlflow.db existence check", {
    "path": str(db_path),
    "exists": db_path.exists(),
    "size_bytes": db_path.stat().st_size if db_path.exists() else 0,
})

# Try opening the DB directly with sqlite3
if db_path.exists():
    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        exp_count = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] if ("experiments",) in tables else "no table"
        run_count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] if ("runs",) in tables else "no table"
        conn.close()
        w("DB", "sqlite3 direct read", {"tables": [t[0] for t in tables], "experiments": exp_count, "runs": run_count})
    except Exception as e:
        w("DB", "sqlite3 read failed", {"error": str(e)})

# Try MLflow client with the corrected URI
if uri:
    try:
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        import mlflow
        mlflow.set_tracking_uri(uri)
        client = mlflow.tracking.MlflowClient()
        exps = client.search_experiments()
        runs = client.search_runs([e.experiment_id for e in exps]) if exps else []
        w("CLIENT", "mlflow client result", {
            "uri": uri,
            "experiments": len(exps),
            "exp_names": [e.name for e in exps],
            "runs": len(runs),
        })
    except Exception as e:
        w("CLIENT", "mlflow client failed", {"error": str(e), "uri": uri})

# Also try with as_posix URI explicitly
posix_uri = "sqlite:///" + Path(__file__).resolve().parent.as_posix() + "/mlruns/mlflow.db"
w("POSIX", "posix URI value", {"posix_uri": posix_uri})
if db_path.exists():
    try:
        import mlflow as mlf2
        mlf2.set_tracking_uri(posix_uri)
        c2 = mlf2.tracking.MlflowClient()
        exps2 = c2.search_experiments()
        w("POSIX", "posix URI client result", {"experiments": len(exps2), "runs": len(c2.search_runs([e.experiment_id for e in exps2])) if exps2 else 0})
    except Exception as e:
        w("POSIX", "posix URI client failed", {"error": str(e)})

print(f"\nDone. Check {LOG}")

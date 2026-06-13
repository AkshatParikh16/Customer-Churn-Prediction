"""
src/churn/models/train.py
"""
from __future__ import annotations
import sys, time, tempfile
from pathlib import Path
from typing import Any
import joblib, mlflow, mlflow.sklearn
import numpy as np, pandas as pd
import optuna, typer
from loguru import logger
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from configs.settings import settings
from churn.data.preprocess import prepare_datasets
from churn.features.engineer import apply_smote

app = typer.Typer(rich_markup_mode="rich")
optuna.logging.set_verbosity(optuna.logging.WARNING)

AVG_ANNUAL_CUSTOMER_VALUE = 48.46 * 12
RETENTION_CALL_COST       = 10.0
RETENTION_SUCCESS_RATE    = 0.30
THRESHOLD_GRID            = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def expected_savings(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> float:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return round(
        tp * AVG_ANNUAL_CUSTOMER_VALUE * RETENTION_SUCCESS_RATE
        - (tp + fp) * RETENTION_CALL_COST,
        2,
    )


def threshold_sweep(
    y_true: np.ndarray, y_proba: np.ndarray, prefix: str = "val"
) -> tuple[pd.DataFrame, float]:
    rows = []
    for t in THRESHOLD_GRID:
        y_pred = (y_proba >= t).astype(int)
        rows.append({
            "threshold":        t,
            "roc_auc":          round(roc_auc_score(y_true, y_proba), 4),
            "pr_auc":           round(average_precision_score(y_true, y_proba), 4),
            "recall":           round(recall_score(y_true, y_pred, zero_division=0), 4),
            "precision":        round(precision_score(y_true, y_pred, zero_division=0), 4),
            "f1":               round(f1_score(y_true, y_pred, zero_division=0), 4),
            "n_flagged":        int(y_pred.sum()),
            "expected_savings": expected_savings(y_true, y_proba, t),
        })
    df = pd.DataFrame(rows)
    best_t = float(df.loc[df["expected_savings"].idxmax(), "threshold"])
    return df, best_t


def evaluate(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float,
    prefix: str = "val",
) -> dict[str, float]:
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        f"{prefix}/roc_auc":          round(roc_auc_score(y, y_prob), 4),
        f"{prefix}/pr_auc":           round(average_precision_score(y, y_prob), 4),
        f"{prefix}/recall":           round(recall_score(y, y_pred, zero_division=0), 4),
        f"{prefix}/precision":        round(precision_score(y, y_pred, zero_division=0), 4),
        f"{prefix}/f1":               round(f1_score(y, y_pred, zero_division=0), 4),
        f"{prefix}/expected_savings": expected_savings(y, y_prob, threshold),
    }
    logger.info("{} metrics @ threshold={}: {}", prefix, threshold, metrics)
    return metrics


def log_threshold_sweep(
    model: Any, X_val: np.ndarray, y_val: np.ndarray, model_name: str
) -> float:
    y_proba = model.predict_proba(X_val)[:, 1]
    sweep_df, best_t = threshold_sweep(y_val, y_proba)
    csv_path = str(
        Path(tempfile.gettempdir())
        / f"threshold_sweep_{model_name.lower().replace(' ', '_')}.csv"
    )
    sweep_df.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path, artifact_path="threshold_sweep")
    mlflow.log_metric("best_threshold", best_t)
    logger.info(
        "{} best threshold={} savings=${:,.0f} recall={}",
        model_name, best_t,
        sweep_df.loc[sweep_df["threshold"] == best_t, "expected_savings"].values[0],
        sweep_df.loc[sweep_df["threshold"] == best_t, "recall"].values[0],
    )
    return best_t


def save_shap_explainer(model: Any, X_val: np.ndarray, model_name: str) -> None:
    try:
        import shap
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val[:500])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        feat_names = getattr(model, "feature_names_in_", None) or [
            f"f{i}" for i in range(X_val.shape[1])
        ]
        importance = pd.DataFrame({
            "feature":   feat_names,
            "mean_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_shap", ascending=False)
        csv_path = str(
            Path(tempfile.gettempdir())
            / f"shap_{model_name.lower().replace(' ', '_')}.csv"
        )
        importance.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path="shap")
        explainer_path = (
            Path(settings.model.models_dir)
            / f"shap_explainer_{model_name.lower().replace(' ', '_')}.joblib"
        )
        joblib.dump(explainer, explainer_path)
        logger.info("SHAP explainer saved → {}", explainer_path)
    except Exception as e:
        logger.warning("SHAP skipped for {}: {}", model_name, e)


# ── Optuna objectives ─────────────────────────────────────────────────────────

def _xgb_objective(
    trial: optuna.Trial,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
) -> float:
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 800),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight":  settings.model.xgb_scale_pos_weight,
        "tree_method":       "hist",
        "eval_metric":       "auc",
        "random_state":      settings.model.random_state,
        "n_jobs":            -1,
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])


def _lgb_objective(
    trial: optuna.Trial,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
) -> float:
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 20, 150),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": settings.model.xgb_scale_pos_weight,
        "random_state":     settings.model.random_state,
        "n_jobs":           -1,
        "verbosity":        -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])


def _cat_objective(
    trial: optuna.Trial,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
) -> float:
    params = {
        "iterations":       trial.suggest_int("iterations", 200, 800),
        "depth":            trial.suggest_int("depth", 4, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg":      trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "border_count":     trial.suggest_int("border_count", 32, 255),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "scale_pos_weight": settings.model.xgb_scale_pos_weight,
        "eval_metric":      "AUC",
        "random_seed":      settings.model.random_state,
        "verbose":          0,
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])


def _rf_objective(
    trial: optuna.Trial,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
) -> float:
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
        "max_depth":        trial.suggest_int("max_depth", 5, 25),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        "class_weight":     "balanced",
        "random_state":     settings.model.random_state,
        "n_jobs":           -1,
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])


def _et_objective(
    trial: optuna.Trial,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
) -> float:
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
        "max_depth":        trial.suggest_int("max_depth", 5, 25),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        "class_weight":     "balanced",
        "random_state":     settings.model.random_state,
        "n_jobs":           -1,
    }
    model = ExtraTreesClassifier(**params)
    model.fit(X_train, y_train)
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])


# ── Model builders ────────────────────────────────────────────────────────────

def build_xgb(p: dict[str, Any]) -> XGBClassifier:
    return XGBClassifier(
        **p,
        scale_pos_weight=settings.model.xgb_scale_pos_weight,
        tree_method="hist",
        eval_metric="auc",
        random_state=settings.model.random_state,
        n_jobs=-1,
    )


def build_lgb(p: dict[str, Any]) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        **p,
        scale_pos_weight=settings.model.xgb_scale_pos_weight,
        random_state=settings.model.random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def build_lr_baseline() -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=settings.model.random_state,
    )


def _run_optuna(
    objective_fn: Any,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
    trials: int,
) -> tuple[dict[str, Any], float]:
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: objective_fn(t, X_train, y_train, X_val, y_val),
        n_trials=trials,
        timeout=settings.model.optuna_timeout_secs,
        show_progress_bar=True,
    )
    return study.best_params, study.best_value


# ── Main training command ─────────────────────────────────────────────────────

@app.command()
def main(
    trials: int = typer.Option(settings.model.optuna_n_trials, help="Optuna trials per model"),
    skip_tuning: bool = typer.Option(False, help="Skip Optuna, use default params"),
) -> None:
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment)

    data = prepare_datasets()
    X_train, y_train = apply_smote(data["X_train"], data["y_train"])
    X_val,  y_val    = data["X_val"],  data["y_val"]
    X_test, y_test   = data["X_test"], data["y_test"]
    results: list[dict[str, Any]] = []

    # ── 1. Logistic Regression baseline ──────────────────────────────────────
    with mlflow.start_run(run_name="LogisticRegression-Baseline"):
        lr = build_lr_baseline()
        lr.fit(X_train, y_train)
        best_t = log_threshold_sweep(lr, X_val, y_val, "LogisticRegression")
        val_m  = evaluate(lr, X_val,  y_val,  best_t, "val")
        tst_m  = evaluate(lr, X_test, y_test, best_t, "test") if len(y_test) > 0 else {}
        mlflow.log_params({"model": "LR", "best_threshold": best_t})
        mlflow.log_metrics({**val_m, **tst_m})
        mlflow.sklearn.log_model(lr, artifact_path="model")
        results.append({"name": "logisticregression", "model": lr,
                        "roc_auc": val_m["val/roc_auc"], "best_threshold": best_t})
        logger.success("LR  ROC-AUC:{}", val_m["val/roc_auc"])

    # ── 2. Random Forest ──────────────────────────────────────────────────────
    with mlflow.start_run(run_name="RandomForest-Optuna"):
        t0 = time.time()
        if skip_tuning:
            rf_params: dict[str, Any] = {
                "n_estimators": 300, "max_depth": 15,
                "min_samples_leaf": 2, "max_features": "sqrt",
                "class_weight": "balanced",
                "random_state": settings.model.random_state, "n_jobs": -1,
            }
        else:
            best_p, best_v = _run_optuna(_rf_objective, X_train, y_train, X_val, y_val, trials)
            rf_params = {**best_p, "class_weight": "balanced",
                         "random_state": settings.model.random_state, "n_jobs": -1}
            mlflow.log_params({"optuna_trials": trials, "optuna_best_auc": round(best_v, 4)})
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_train, y_train)
        best_t = log_threshold_sweep(rf, X_val, y_val, "RandomForest")
        val_m  = evaluate(rf, X_val,  y_val,  best_t, "val")
        tst_m  = evaluate(rf, X_test, y_test, best_t, "test") if len(y_test) > 0 else {}
        mlflow.log_params({**rf_params, "decision_threshold": best_t, "smote": True})
        mlflow.log_metrics({**val_m, **tst_m, "train_time_s": round(time.time() - t0, 1)})
        mlflow.sklearn.log_model(rf, artifact_path="model")
        save_shap_explainer(rf, X_val, "RandomForest")
        logger.success("RF   ROC-AUC:{} Recall:{}", val_m["val/roc_auc"], val_m["val/recall"])
        results.append({"name": "randomforest", "model": rf,
                        "roc_auc": val_m["val/roc_auc"], "best_threshold": best_t})

    # ── 3. Extra Trees ────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="ExtraTrees-Optuna"):
        t0 = time.time()
        if skip_tuning:
            et_params: dict[str, Any] = {
                "n_estimators": 300, "max_depth": 15,
                "min_samples_leaf": 2, "max_features": "sqrt",
                "class_weight": "balanced",
                "random_state": settings.model.random_state, "n_jobs": -1,
            }
        else:
            best_p, best_v = _run_optuna(_et_objective, X_train, y_train, X_val, y_val, trials)
            et_params = {**best_p, "class_weight": "balanced",
                         "random_state": settings.model.random_state, "n_jobs": -1}
            mlflow.log_params({"optuna_trials": trials, "optuna_best_auc": round(best_v, 4)})
        et = ExtraTreesClassifier(**et_params)
        et.fit(X_train, y_train)
        best_t = log_threshold_sweep(et, X_val, y_val, "ExtraTrees")
        val_m  = evaluate(et, X_val,  y_val,  best_t, "val")
        tst_m  = evaluate(et, X_test, y_test, best_t, "test") if len(y_test) > 0 else {}
        mlflow.log_params({**et_params, "decision_threshold": best_t, "smote": True})
        mlflow.log_metrics({**val_m, **tst_m, "train_time_s": round(time.time() - t0, 1)})
        mlflow.sklearn.log_model(et, artifact_path="model")
        save_shap_explainer(et, X_val, "ExtraTrees")
        logger.success("ET   ROC-AUC:{} Recall:{}", val_m["val/roc_auc"], val_m["val/recall"])
        results.append({"name": "extratrees", "model": et,
                        "roc_auc": val_m["val/roc_auc"], "best_threshold": best_t})

    # ── 4. XGBoost ────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="XGBoost-Optuna"):
        t0 = time.time()
        if skip_tuning:
            xgb_params: dict[str, Any] = {
                "n_estimators":     settings.model.xgb_n_estimators,
                "max_depth":        settings.model.xgb_max_depth,
                "learning_rate":    settings.model.xgb_learning_rate,
                "subsample":        settings.model.xgb_subsample,
                "colsample_bytree": settings.model.xgb_colsample_bytree,
            }
        else:
            xgb_params, best_v = _run_optuna(_xgb_objective, X_train, y_train, X_val, y_val, trials)
            mlflow.log_params({"optuna_trials": trials, "optuna_best_auc": round(best_v, 4)})
        xgb = build_xgb(xgb_params)
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        best_t = log_threshold_sweep(xgb, X_val, y_val, "XGBoost")
        val_m  = evaluate(xgb, X_val,  y_val,  best_t, "val")
        tst_m  = evaluate(xgb, X_test, y_test, best_t, "test") if len(y_test) > 0 else {}
        mlflow.log_params({**xgb_params, "decision_threshold": best_t, "smote": True})
        mlflow.log_metrics({**val_m, **tst_m, "train_time_s": round(time.time() - t0, 1)})
        mlflow.sklearn.log_model(
            xgb, artifact_path="model",
            registered_model_name=settings.mlflow.registry_name,
        )
        save_shap_explainer(xgb, X_val, "XGBoost")
        logger.success("XGB  ROC-AUC:{} PR-AUC:{} Recall:{} Savings:${:,.0f}",
                       val_m["val/roc_auc"], val_m["val/pr_auc"],
                       val_m["val/recall"], val_m["val/expected_savings"])
        results.append({"name": "xgboost", "model": xgb,
                        "roc_auc": val_m["val/roc_auc"], "best_threshold": best_t})

    # ── 5. LightGBM ───────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="LightGBM-Optuna"):
        t0 = time.time()
        if skip_tuning:
            lgb_params: dict[str, Any] = {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05}
        else:
            lgb_params, best_v = _run_optuna(_lgb_objective, X_train, y_train, X_val, y_val, trials)
            mlflow.log_params({"optuna_trials": trials, "optuna_best_auc": round(best_v, 4)})
        lgb_model = build_lgb(lgb_params)
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        best_t = log_threshold_sweep(lgb_model, X_val, y_val, "LightGBM")
        val_m  = evaluate(lgb_model, X_val,  y_val,  best_t, "val")
        tst_m  = evaluate(lgb_model, X_test, y_test, best_t, "test") if len(y_test) > 0 else {}
        mlflow.log_params({**lgb_params, "decision_threshold": best_t})
        mlflow.log_metrics({**val_m, **tst_m, "train_time_s": round(time.time() - t0, 1)})
        mlflow.sklearn.log_model(lgb_model, artifact_path="model")
        save_shap_explainer(lgb_model, X_val, "LightGBM")
        logger.success("LGB  ROC-AUC:{} Recall:{}", val_m["val/roc_auc"], val_m["val/recall"])
        results.append({"name": "lightgbm", "model": lgb_model,
                        "roc_auc": val_m["val/roc_auc"], "best_threshold": best_t})

    # ── 6. CatBoost (Optuna) ──────────────────────────────────────────────────
    with mlflow.start_run(run_name="CatBoost-Optuna"):
        t0 = time.time()
        if skip_tuning:
            cat_params: dict[str, Any] = {
                "iterations": 500, "learning_rate": 0.05, "depth": 6,
                "scale_pos_weight": settings.model.xgb_scale_pos_weight,
                "eval_metric": "AUC", "random_seed": settings.model.random_state, "verbose": 0,
            }
        else:
            best_p, best_v = _run_optuna(_cat_objective, X_train, y_train, X_val, y_val, trials)
            cat_params = {
                **best_p,
                "scale_pos_weight": settings.model.xgb_scale_pos_weight,
                "eval_metric":      "AUC",
                "random_seed":      settings.model.random_state,
                "verbose":          0,
            }
            mlflow.log_params({"optuna_trials": trials, "optuna_best_auc": round(best_v, 4)})
        cat_model = CatBoostClassifier(**cat_params)
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
        best_t = log_threshold_sweep(cat_model, X_val, y_val, "CatBoost")
        val_m  = evaluate(cat_model, X_val,  y_val,  best_t, "val")
        tst_m  = evaluate(cat_model, X_test, y_test, best_t, "test") if len(y_test) > 0 else {}
        mlflow.log_params({**cat_params, "decision_threshold": best_t})
        mlflow.log_metrics({**val_m, **tst_m, "train_time_s": round(time.time() - t0, 1)})
        mlflow.sklearn.log_model(cat_model, artifact_path="model")
        save_shap_explainer(cat_model, X_val, "CatBoost")
        logger.success("CAT  ROC-AUC:{} Recall:{}", val_m["val/roc_auc"], val_m["val/recall"])
        results.append({"name": "catboost", "model": cat_model,
                        "roc_auc": val_m["val/roc_auc"], "best_threshold": best_t})

    # ── 7. Stacking Ensemble (top-3 base → LR meta) ───────────────────────────
    with mlflow.start_run(run_name="Stacking-Ensemble"):
        t0 = time.time()
        # Pick the 3 highest-AUC base models to stack
        top3 = sorted(results, key=lambda r: r["roc_auc"], reverse=True)[:3]
        estimators = [(r["name"], r["model"]) for r in top3]
        meta = LogisticRegression(max_iter=500, random_state=settings.model.random_state)
        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=meta,
            cv=3,
            n_jobs=-1,
            passthrough=False,
        )
        stack.fit(X_train, y_train)
        best_t = log_threshold_sweep(stack, X_val, y_val, "Stacking")
        val_m  = evaluate(stack, X_val,  y_val,  best_t, "val")
        tst_m  = evaluate(stack, X_test, y_test, best_t, "test") if len(y_test) > 0 else {}
        base_names = [r["name"] for r in top3]
        mlflow.log_params({"base_models": str(base_names), "meta_learner": "LR",
                            "decision_threshold": best_t})
        mlflow.log_metrics({**val_m, **tst_m, "train_time_s": round(time.time() - t0, 1)})
        mlflow.sklearn.log_model(stack, artifact_path="model")
        logger.success("STACK ROC-AUC:{} Recall:{} (bases={})",
                       val_m["val/roc_auc"], val_m["val/recall"], base_names)
        results.append({"name": "stacking", "model": stack,
                        "roc_auc": val_m["val/roc_auc"], "best_threshold": best_t})

    # ── Select and persist best model ─────────────────────────────────────────
    best = max(results, key=lambda r: r["roc_auc"])
    safe_name = best["name"].lower().replace(" ", "_")
    logger.success(
        "Best model: {} | ROC-AUC: {:.4f} | threshold: {}",
        best["name"], best["roc_auc"], best["best_threshold"],
    )

    # Print leaderboard
    leaderboard = sorted(results, key=lambda r: r["roc_auc"], reverse=True)
    logger.info("── Model Leaderboard ──────────────────────────────")
    for rank, r in enumerate(leaderboard, 1):
        logger.info("  {:>2}. {:20s}  ROC-AUC={:.4f}", rank, r["name"], r["roc_auc"])

    settings.model.models_dir.mkdir(parents=True, exist_ok=True)
    model_filename = f"churn_{safe_name}_prod.joblib"
    joblib.dump(best["model"], settings.model.models_dir / model_filename)
    joblib.dump(
        {
            "model_name":      best["name"],
            "model_filename":  model_filename,
            "roc_auc":         best["roc_auc"],
            "best_threshold":  best["best_threshold"],
        },
        settings.model.models_dir / "model_metadata.joblib",
    )
    logger.success("Saved → {}", settings.model.models_dir / model_filename)


if __name__ == "__main__":
    app()

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import qmc
from sklearn.linear_model import ElasticNetCV, HuberRegressor, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PARAM_BOUNDS = {
    "learning_rate": (0.01, 0.3),
    "max_depth": (3, 8),
    "num_leaves": (20, 100),
    "lambda_l1": (0.0, 1.0),
    "lambda_l2": (0.0, 1.0),
}

COL_ACTUAL = "\u771f\u5b9e\u503c"
COL_RECOMMENDED = "\u63a8\u8350\u6a21\u578b\u9884\u6d4b"
COL_ERROR = "\u8bef\u5dee"
COL_FEATURE = "\u7279\u5f81"
COL_MODEL = "\u6a21\u578b"
COL_NOTE = "\u5907\u6ce8"
COL_RUN = "\u8fd0\u884c\u6b21\u6570"
COL_SCENARIO = "\u5de5\u51b5"
COL_SAMPLE_COUNT = "\u6837\u672c\u6570"
COL_SCENARIO_LOW = "\u5de5\u51b5_\u4f4e\u6d4a\u671f"
COL_SCENARIO_FLOOD = "\u5de5\u51b5_\u6c5b\u671f"
COL_SCENARIO_STRAT = "\u5de5\u51b5_\u5206\u5c42\u671f"
NAME_LOW = "\u4f4e\u6d4a\u671f"
NAME_FLOOD = "\u6c5b\u671f"
NAME_STRAT = "\u5206\u5c42\u671f"
NAME_RECOMMENDED_MODEL = "Huber-ElasticNet\u878d\u5408(\u63a8\u8350)"
NAME_TREE_MODEL = "\u4f18\u5316LightGBM"
NAME_TREE_DEFAULT = "\u672a\u4f18\u5316LightGBM"
NAME_LINEAR = "\u7ebf\u6027\u56de\u5f52"
NAME_POWER = "\u5e42\u51fd\u6570\u56de\u5f52"
COL_BLEND_IMPORTANCE = "\u878d\u5408\u7ebf\u6027\u7cfb\u6570\u7edd\u5bf9\u503c"
COL_ENET_IMPORTANCE = "ElasticNet\u7cfb\u6570\u7edd\u5bf9\u503c"
COL_HUBER_IMPORTANCE = "Huber\u7cfb\u6570\u7edd\u5bf9\u503c"
COL_LGBM_IMPORTANCE = "LightGBM\u91cd\u8981\u6027"
COL_BLEND_PRED = "Huber-ElasticNet\u878d\u5408\u9884\u6d4b"
COL_LGBM_PRED = "\u4f18\u5316LightGBM\u9884\u6d4b"
COL_LGBM_DEFAULT_PRED = "\u672a\u4f18\u5316LightGBM\u9884\u6d4b"
COL_LINEAR_PRED = "\u7ebf\u6027\u56de\u5f52\u9884\u6d4b"
COL_POWER_PRED = "\u5e42\u51fd\u6570\u56de\u5f52\u9884\u6d4b"


@dataclass
class ModelArtifacts:
    best_params: dict[str, float]
    best_score: float
    convergence: list[float]
    final_model: LGBMRegressor
    recommended_model: dict[str, Any]
    recommendation_rule: dict[str, float]
    recommendation_weight: float
    train_metrics: pd.DataFrame
    comparison_metrics: pd.DataFrame
    stability_metrics: pd.DataFrame
    generalization_metrics: pd.DataFrame
    test_predictions: pd.DataFrame
    feature_importance: pd.DataFrame


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def decode_position(position: np.ndarray) -> dict[str, float]:
    params = {}
    for idx, (name, (lower, upper)) in enumerate(PARAM_BOUNDS.items()):
        value = lower + position[idx] * (upper - lower)
        if name in {"max_depth", "num_leaves"}:
            params[name] = int(round(value))
        else:
            params[name] = float(value)
    params["num_leaves"] = min(params["num_leaves"], 2 ** params["max_depth"] - 1)
    return params


def build_lgbm(params: dict[str, float], random_state: int = 42) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=params["learning_rate"],
        max_depth=int(params["max_depth"]),
        num_leaves=int(params["num_leaves"]),
        reg_alpha=params["lambda_l1"],
        reg_lambda=params["lambda_l2"],
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        min_child_samples=10,
        random_state=random_state,
        verbose=-1,
    )


class ISSAOptimizer:
    def __init__(self, population_size: int = 16, iterations: int = 60, discoverer_ratio: float = 0.3, random_state: int = 42) -> None:
        self.population_size = population_size
        self.iterations = iterations
        self.discoverer_ratio = discoverer_ratio
        self.random_state = np.random.default_rng(random_state)

    def initialize_population(self, dim: int) -> np.ndarray:
        sobol = qmc.Sobol(d=dim, scramble=True, seed=int(self.random_state.integers(0, 10000)))
        return np.clip(sobol.random(self.population_size), 0.0, 1.0)

    def cauchy_gaussian_mutation(self, best: np.ndarray, iteration: int) -> np.ndarray:
        scale = 1 - iteration / max(self.iterations, 1)
        cauchy_noise = self.random_state.standard_cauchy(best.shape[0]) * 0.05 * scale
        gaussian_noise = self.random_state.normal(0, 0.03 * scale, size=best.shape[0])
        return np.clip(best + cauchy_noise + gaussian_noise, 0.0, 1.0)

    def optimize(self, objective_fn) -> tuple[dict[str, float], float, list[float]]:
        dim = len(PARAM_BOUNDS)
        population = self.initialize_population(dim)
        fitness = np.array([objective_fn(decode_position(ind)) for ind in population], dtype=float)
        best_idx = int(np.argmin(fitness))
        global_best = population[best_idx].copy()
        global_score = float(fitness[best_idx])
        convergence = [global_score]
        discoverers = max(2, int(self.population_size * self.discoverer_ratio))

        for iteration in range(self.iterations):
            order = np.argsort(fitness)
            population = population[order]
            fitness = fitness[order]
            current_best = population[0].copy()
            worst = population[-1].copy()
            for i in range(self.population_size):
                alarm = self.random_state.random()
                if i < discoverers:
                    if alarm < 0.8:
                        decay = np.exp(-(i + 1) / (self.random_state.random() * self.iterations + 1e-6))
                        population[i] = population[i] * decay
                    else:
                        population[i] = population[i] + self.random_state.normal(0, 0.1, size=dim)
                else:
                    ref_a = population[int(self.random_state.integers(0, discoverers))]
                    ref_b = population[int(self.random_state.integers(0, discoverers))]
                    learn = 0.5 * (ref_a + ref_b)
                    population[i] = population[i] + self.random_state.normal(0, 0.15, size=dim) * (learn - population[i])
                    population[i] = population[i] + self.random_state.random(dim) * (current_best - worst)
                population[i] = np.clip(population[i], 0.0, 1.0)
            population[-1] = self.cauchy_gaussian_mutation(current_best, iteration)
            fitness = np.array([objective_fn(decode_position(ind)) for ind in population], dtype=float)
            best_idx = int(np.argmin(fitness))
            if float(fitness[best_idx]) < global_score:
                global_best = population[best_idx].copy()
                global_score = float(fitness[best_idx])
            convergence.append(global_score)
        return decode_position(global_best), global_score, convergence


def fit_power_regression(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    cols = [c for c in X_train.columns if (X_train[c] >= 0).all()][: min(5, X_train.shape[1])]
    if not cols:
        cols = list(X_train.columns[: min(5, X_train.shape[1])])
    safe_train = np.log1p(X_train[cols].clip(lower=0))
    safe_test = np.log1p(X_test[cols].clip(lower=0))
    model = LinearRegression()
    model.fit(safe_train, np.log(y_train.clip(lower=1e-6)))
    return np.exp(model.predict(safe_test))


def apply_recommendation_rule(base_pred: np.ndarray, raw_features: pd.DataFrame, rule: dict[str, float]) -> np.ndarray:
    pred = base_pred.copy()
    lag1 = raw_features["target_lag_1"].to_numpy()
    roll3 = raw_features["target_roll3"].to_numpy()
    high_anchor = rule["high_w"] * lag1 + (1 - rule["high_w"]) * roll3
    low_anchor = rule["low_w"] * lag1 + (1 - rule["low_w"]) * roll3
    high_mask = (lag1 >= rule["high_t"]) & (roll3 >= rule["high_t"])
    low_mask = (lag1 <= rule["low_t"]) & (roll3 <= rule["low_t"])
    pred[high_mask] = np.maximum(pred[high_mask], high_anchor[high_mask])
    pred[low_mask] = np.minimum(pred[low_mask], low_anchor[low_mask])
    return pred


def search_recommendation_rule(valid_pred: np.ndarray, raw_valid: pd.DataFrame, y_valid: pd.Series) -> dict[str, float]:
    best_rule: dict[str, float] | None = None
    best_rmse = float("inf")
    for high_t, low_t, high_w, low_w in itertools.product([8.0, 8.5, 9.0, 9.5, 10.0], [4.5, 4.8, 5.0, 5.2], [0.5, 0.6, 0.7, 0.8], [0.5, 0.6, 0.7, 0.8]):
        rule = {"high_t": high_t, "low_t": low_t, "high_w": high_w, "low_w": low_w}
        adj_pred = apply_recommendation_rule(valid_pred, raw_valid, rule)
        rmse = float(np.sqrt(mean_squared_error(y_valid, adj_pred)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_rule = rule
    assert best_rule is not None
    return best_rule


def build_recommended_prediction(
    enet_pred: np.ndarray,
    huber_pred: np.ndarray,
    raw_features: pd.DataFrame,
    weight: float,
    rule: dict[str, float],
) -> np.ndarray:
    enet_adjusted = apply_recommendation_rule(enet_pred, raw_features, rule)
    return weight * huber_pred + (1 - weight) * enet_adjusted


def summarize_generalization(predictions: pd.DataFrame) -> pd.DataFrame:
    scenarios = {
        NAME_LOW: predictions[COL_SCENARIO_LOW] == 1,
        NAME_FLOOD: predictions[COL_SCENARIO_FLOOD] == 1,
        NAME_STRAT: predictions[COL_SCENARIO_STRAT] == 1,
    }
    rows = []
    for name, mask in scenarios.items():
        subset = predictions[mask].copy()
        if len(subset) < 5:
            continue
        metric = compute_metrics(subset[COL_ACTUAL], subset[COL_RECOMMENDED])
        metric[COL_SCENARIO] = name
        metric[COL_SAMPLE_COUNT] = int(len(subset))
        rows.append(metric)
    return pd.DataFrame(rows)


def run_model_suite(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    raw_train: pd.DataFrame,
    raw_valid: pd.DataFrame,
    raw_test: pd.DataFrame,
    test_meta: pd.DataFrame,
) -> ModelArtifacts:
    def objective(params: dict[str, float]) -> float:
        model = build_lgbm(params, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        return float(np.sqrt(mean_squared_error(y_valid, pred)))

    optimizer = ISSAOptimizer(population_size=18, iterations=60, random_state=42)
    best_params, best_score, convergence = optimizer.optimize(objective)

    full_train_X = pd.concat([X_train, X_valid], axis=0)
    full_train_y = pd.concat([y_train, y_valid], axis=0)

    final_model = build_lgbm(best_params, random_state=42)
    final_model.fit(full_train_X, full_train_y)
    final_pred = final_model.predict(X_test)

    enet_train = ElasticNetCV(l1_ratio=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0], alphas=np.logspace(-4, 1, 25), max_iter=20000)
    enet_train.fit(X_train, y_train)
    recommendation_rule = search_recommendation_rule(enet_train.predict(X_valid), raw_valid, y_valid)

    huber_train = HuberRegressor(alpha=0.01, epsilon=1.2, max_iter=5000)
    huber_train.fit(X_train, y_train)
    enet_valid_pred = apply_recommendation_rule(enet_train.predict(X_valid), raw_valid, recommendation_rule)
    huber_valid_pred = huber_train.predict(X_valid)

    best_weight = 0.0
    best_blend_score = float("inf")
    for weight in np.linspace(0.0, 1.0, 101):
        blend_pred = weight * huber_valid_pred + (1 - weight) * enet_valid_pred
        score = float(np.sqrt(mean_squared_error(y_valid, blend_pred)))
        if score < best_blend_score:
            best_blend_score = score
            best_weight = float(weight)

    enet_full = ElasticNetCV(l1_ratio=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0], alphas=np.logspace(-4, 1, 25), max_iter=20000)
    enet_full.fit(full_train_X, full_train_y)
    huber_full = HuberRegressor(alpha=0.01, epsilon=1.2, max_iter=5000)
    huber_full.fit(full_train_X, full_train_y)
    recommended_pred = build_recommended_prediction(
        enet_full.predict(X_test),
        huber_full.predict(X_test),
        raw_test,
        weight=best_weight,
        rule=recommendation_rule,
    )
    recommended_model = {
        "type": "huber_elasticnet_blend",
        "weight_huber": best_weight,
        "elasticnet": enet_full,
        "huber": huber_full,
        "rule": recommendation_rule,
    }

    default_model = build_lgbm({"learning_rate": 0.1, "max_depth": 5, "num_leaves": 31, "lambda_l1": 0.0, "lambda_l2": 0.0}, random_state=42)
    default_model.fit(full_train_X, full_train_y)
    default_pred = default_model.predict(X_test)

    linear_model = LinearRegression()
    linear_model.fit(full_train_X, full_train_y)
    linear_pred = linear_model.predict(X_test)

    power_pred = fit_power_regression(full_train_X, full_train_y, X_test)

    comparison_rows = []
    model_predictions = {
        NAME_RECOMMENDED_MODEL: recommended_pred,
        NAME_TREE_MODEL: final_pred,
        NAME_TREE_DEFAULT: default_pred,
        NAME_LINEAR: linear_pred,
        NAME_POWER: power_pred,
    }
    for name, pred in model_predictions.items():
        metrics = compute_metrics(y_test, pred)
        metrics[COL_MODEL] = name
        comparison_rows.append(metrics)
    comparison_metrics = pd.DataFrame(comparison_rows).sort_values("RMSE")

    stability_rows = []
    for seed in range(20):
        model = build_lgbm(best_params, random_state=seed)
        model.fit(full_train_X, full_train_y)
        pred = model.predict(X_test)
        stability_rows.append({COL_RUN: seed + 1, **compute_metrics(y_test, pred)})
    stability_metrics = pd.DataFrame(stability_rows)

    prediction_frame = test_meta.copy()
    prediction_frame[COL_ACTUAL] = y_test.values
    prediction_frame[COL_RECOMMENDED] = recommended_pred
    prediction_frame[COL_BLEND_PRED] = recommended_pred
    prediction_frame[COL_LGBM_PRED] = final_pred
    prediction_frame[COL_LGBM_DEFAULT_PRED] = default_pred
    prediction_frame[COL_LINEAR_PRED] = linear_pred
    prediction_frame[COL_POWER_PRED] = power_pred
    prediction_frame[COL_ERROR] = prediction_frame[COL_RECOMMENDED] - prediction_frame[COL_ACTUAL]
    generalization_metrics = summarize_generalization(prediction_frame)

    feature_importance = pd.DataFrame({
        COL_FEATURE: full_train_X.columns,
        COL_BLEND_IMPORTANCE: best_weight * np.abs(huber_full.coef_) + (1 - best_weight) * np.abs(enet_full.coef_),
        COL_ENET_IMPORTANCE: np.abs(enet_full.coef_),
        COL_HUBER_IMPORTANCE: np.abs(huber_full.coef_),
        COL_LGBM_IMPORTANCE: final_model.feature_importances_,
    }).sort_values([COL_BLEND_IMPORTANCE, COL_LGBM_IMPORTANCE], ascending=False)

    train_metrics = pd.DataFrame([
        {COL_MODEL: NAME_RECOMMENDED_MODEL, **compute_metrics(y_test, recommended_pred), COL_NOTE: f"\u8fc7\u7a0b\u8bb0\u5fc6\u589e\u5f3a + \u878d\u5408\u6743\u91cd={best_weight:.2f}"},
        {COL_MODEL: NAME_TREE_MODEL, **compute_metrics(y_test, final_pred), COL_NOTE: f"ISSA\u6700\u4f18\u9a8c\u8bc1RMSE={best_score:.4f}"},
    ])

    return ModelArtifacts(
        best_params=best_params,
        best_score=best_score,
        convergence=convergence,
        final_model=final_model,
        recommended_model=recommended_model,
        recommendation_rule=recommendation_rule,
        recommendation_weight=best_weight,
        train_metrics=train_metrics,
        comparison_metrics=comparison_metrics,
        stability_metrics=stability_metrics,
        generalization_metrics=generalization_metrics,
        test_predictions=prediction_frame,
        feature_importance=feature_importance,
    )

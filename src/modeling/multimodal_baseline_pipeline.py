"""Multimodal baseline modeling pipeline with conservative leakage checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODALITY_FLAG_COLUMNS = ("has_ehr", "has_mri", "has_eeg")

LEAKAGE_SENSITIVE_MULTIMODAL_COLUMNS = {
    "ehr_likely_dre",
    "ehr_reasons",
    "ehr_missing_evidence",
    "ehr_definition_version",
    "ehr_epilepsy_evidence_status",
    "ehr_asm_evidence_status",
    "ehr_seizure_burden_status",
    "ehr_has_epilepsy_diagnosis",
    "ehr_has_recurrent_seizure_care",
    "ehr_distinct_asm_count",
    "ehr_has_two_or_more_distinct_asms",
    "ehr_second_asm_start_time",
    "ehr_post_second_asm_event_count",
    "ehr_has_persistent_seizure_burden",
}


@dataclass(slots=True)
class MultimodalBaselineModelConfig:
    """Configuration for conservative multimodal baseline model training."""

    target_col: str = "ehr_likely_dre"
    patient_id_col: str = "patient_id"
    split_col: str | None = None
    feature_cols: list[str] | None = None
    exclude_cols: list[str] | None = None
    include_modality_flags: bool = True
    allow_single_modality_rows: bool = True
    require_any_feature: bool = True
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    use_stratified_split: bool = True
    model_type: str = "logistic_regression"
    class_weight: str | None = "balanced"
    max_iter: int = 1000
    output_dir: str | None = None
    model_filename: str = "multimodal_baseline_model.joblib"
    metrics_filename: str = "multimodal_baseline_metrics.json"
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class MultimodalLeakageCheckResult:
    """Structured result of multimodal feature leakage inspection."""

    candidate_feature_cols: list[str]
    excluded_feature_cols: list[str]
    excluded_for_leakage: list[str]
    excluded_for_identifier: list[str]
    excluded_for_non_numeric: list[str]
    excluded_for_config: list[str]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class MultimodalTrainingResult:
    """Structured output for one multimodal baseline training run."""

    model_path: str | None
    metrics_path: str | None
    train_row_count: int
    val_row_count: int
    test_row_count: int
    feature_cols_used: list[str]
    leakage_check: MultimodalLeakageCheckResult
    modality_flag_cols_used: list[str]
    metrics: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""

        return _to_serializable(asdict(self))


class MultimodalBaselinePipeline:
    """Train and evaluate an auditable baseline model on fused patient tables."""

    def __init__(self, config: MultimodalBaselineModelConfig | None = None) -> None:
        self.config = config or MultimodalBaselineModelConfig()

    def run(self, df: pd.DataFrame) -> MultimodalTrainingResult:
        """Run validation, leakage checks, splitting, training, and evaluation."""

        notes = self.validate_input(df)
        if self.config.notes:
            notes.append(self.config.notes)

        modeling_df = df[df[self.config.target_col].notna()].copy()
        if modeling_df.empty:
            raise ValueError("No usable rows remain after dropping null targets.")

        leakage_result = self.run_leakage_checks(modeling_df)
        feature_cols, modality_flag_cols = self.select_feature_columns(
            modeling_df,
            leakage_result,
        )
        train_df, val_df, test_df, split_notes = self.build_splits(modeling_df)
        notes.extend(leakage_result.notes)
        notes.extend(split_notes)

        model = self.train_model(
            self._feature_frame(train_df, feature_cols),
            train_df[self.config.target_col].astype(int),
        )

        metrics = {
            "train": self.evaluate_model(
                model,
                self._feature_frame(train_df, feature_cols),
                train_df[self.config.target_col].astype(int),
                "train",
            ),
            "val": self.evaluate_model(
                model,
                self._feature_frame(val_df, feature_cols),
                val_df[self.config.target_col].astype(int),
                "val",
            ),
            "test": self.evaluate_model(
                model,
                self._feature_frame(test_df, feature_cols),
                test_df[self.config.target_col].astype(int),
                "test",
            ),
            "feature_cols_used": feature_cols,
            "modality_flag_cols_used": modality_flag_cols,
            "model_type": self.config.model_type,
        }
        model_path, metrics_path = self.save_outputs(model, metrics)

        return MultimodalTrainingResult(
            model_path=model_path,
            metrics_path=metrics_path,
            train_row_count=len(train_df),
            val_row_count=len(val_df),
            test_row_count=len(test_df),
            feature_cols_used=feature_cols,
            leakage_check=leakage_result,
            modality_flag_cols_used=modality_flag_cols,
            metrics=metrics,
            notes=list(dict.fromkeys(notes)),
        )

    def validate_input(self, df: pd.DataFrame) -> list[str]:
        """Validate required modeling columns and target availability."""

        if df.empty:
            raise ValueError("Cannot train multimodal baseline model on an empty dataframe.")
        if self.config.target_col not in df.columns:
            raise ValueError(f"Missing target column: {self.config.target_col}")
        if self.config.patient_id_col not in df.columns:
            raise ValueError(
                f"Missing patient identifier column: {self.config.patient_id_col}"
            )
        if df[self.config.target_col].isna().all():
            raise ValueError(f"Target column is all null: {self.config.target_col}")

        notes: list[str] = []
        available_flags = self._extract_modality_flag_cols(df)
        if self.config.include_modality_flags and available_flags:
            notes.append(
                "modality availability flags found: " + ", ".join(available_flags)
            )
        if not self.config.allow_single_modality_rows and available_flags:
            single_modality_rows = self._single_modality_row_count(df, available_flags)
            if single_modality_rows:
                raise ValueError(
                    "Fused dataset contains rows with fewer than two available modalities."
                )
        return notes

    def run_leakage_checks(self, df: pd.DataFrame) -> MultimodalLeakageCheckResult:
        """Exclude identifiers, target-derived fields, configured columns, and text."""

        initial_cols = (
            list(self.config.feature_cols)
            if self.config.feature_cols is not None
            else list(df.columns)
        )
        explicit_excludes = set(self.config.exclude_cols or [])
        identifier_cols = {self.config.patient_id_col}
        if self.config.split_col:
            identifier_cols.add(self.config.split_col)

        excluded_feature_cols: list[str] = []
        excluded_for_leakage: list[str] = []
        excluded_for_identifier: list[str] = []
        excluded_for_non_numeric: list[str] = []
        excluded_for_config: list[str] = []
        safe_candidate_cols: list[str] = []

        for column in initial_cols:
            if column not in df.columns:
                excluded_feature_cols.append(column)
                continue
            if column in explicit_excludes:
                excluded_for_config.append(column)
                continue
            if column in identifier_cols:
                excluded_for_identifier.append(column)
                continue
            if self._is_leakage_sensitive(column):
                excluded_for_leakage.append(column)
                continue
            if column in MODALITY_FLAG_COLUMNS and not self.config.include_modality_flags:
                excluded_for_config.append(column)
                continue
            if not self._is_numeric_or_bool(df[column]):
                excluded_for_non_numeric.append(column)
                continue
            safe_candidate_cols.append(column)

        notes: list[str] = []
        if excluded_for_leakage:
            notes.append("target-derived multimodal columns were excluded")
        if excluded_for_non_numeric:
            notes.append("non-numeric columns were excluded for this baseline")
        if excluded_for_config:
            notes.append("configured excluded columns were not used")

        return MultimodalLeakageCheckResult(
            candidate_feature_cols=list(dict.fromkeys(safe_candidate_cols)),
            excluded_feature_cols=list(dict.fromkeys(excluded_feature_cols)),
            excluded_for_leakage=list(dict.fromkeys(excluded_for_leakage)),
            excluded_for_identifier=list(dict.fromkeys(excluded_for_identifier)),
            excluded_for_non_numeric=list(dict.fromkeys(excluded_for_non_numeric)),
            excluded_for_config=list(dict.fromkeys(excluded_for_config)),
            notes=notes,
        )

    def select_feature_columns(
        self,
        df: pd.DataFrame,
        leakage_result: MultimodalLeakageCheckResult,
    ) -> tuple[list[str], list[str]]:
        """Select final safe feature columns and modality flag features."""

        safe_cols = [
            column
            for column in leakage_result.candidate_feature_cols
            if column in df.columns
        ]
        if self.config.feature_cols is not None:
            requested = set(self.config.feature_cols)
            safe_cols = [column for column in safe_cols if column in requested]

        modality_flag_cols = [
            column
            for column in self._extract_modality_flag_cols(df)
            if column in safe_cols
        ]
        if not self.config.include_modality_flags:
            safe_cols = [
                column for column in safe_cols if column not in MODALITY_FLAG_COLUMNS
            ]
            modality_flag_cols = []

        if not safe_cols and self.config.require_any_feature:
            raise ValueError("No safe numeric or boolean feature columns remain.")
        return safe_cols, modality_flag_cols

    def build_splits(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
        """Build train, validation, and test splits."""

        if self.config.split_col and self.config.split_col in df.columns:
            return self._split_from_existing_column(df)
        return self._split_randomly(df)

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """Train the configured transparent baseline classifier."""

        if self.config.model_type != "logistic_regression":
            raise ValueError(f"Unsupported baseline model_type: {self.config.model_type}")
        if X_train.empty:
            raise ValueError("Training split has no rows.")
        if not list(X_train.columns):
            raise ValueError("Training split has no feature columns.")
        if y_train.nunique(dropna=True) < 2:
            raise ValueError("Training split must contain at least two target classes.")

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight=self.config.class_weight,
                        max_iter=self.config.max_iter,
                        random_state=self.config.random_state,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train.astype(int))
        return model

    def evaluate_model(
        self,
        model: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        split_name: str,
    ) -> dict[str, Any]:
        """Evaluate one split while recording undefined metrics as notes."""

        row_count = len(X)
        if row_count == 0:
            return {
                "split": split_name,
                "row_count": 0,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "roc_auc": None,
                "average_precision": None,
                "positive_rate_true": None,
                "positive_rate_pred": None,
                "notes": ["split is empty"],
            }

        y_true = y.astype(int)
        y_pred = pd.Series(model.predict(X), index=y.index).astype(int)
        notes: list[str] = []
        metrics: dict[str, Any] = {
            "split": split_name,
            "row_count": row_count,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": None,
            "average_precision": None,
            "positive_rate_true": float(y_true.mean()),
            "positive_rate_pred": float(y_pred.mean()),
            "notes": notes,
        }

        if y_true.nunique(dropna=True) < 2:
            notes.append("roc_auc and average_precision undefined because split has one class")
            return metrics

        scores = self._positive_class_scores(model, X)
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        metrics["average_precision"] = float(average_precision_score(y_true, scores))
        return metrics

    def save_outputs(
        self,
        model: Pipeline,
        metrics: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        """Save model and metrics when an output directory is configured."""

        if self.config.output_dir is None:
            return None, None

        output_dir = Path(self.config.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / self.config.model_filename
        metrics_path = output_dir / self.config.metrics_filename

        joblib.dump(model, model_path)
        metrics_path.write_text(
            json.dumps(_to_serializable(metrics), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return str(model_path), str(metrics_path)

    def _split_from_existing_column(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
        split_values = df[self.config.split_col].astype(str).str.lower()
        train_df = df[split_values == "train"].copy()
        val_df = df[split_values.isin(["val", "valid", "validation"])].copy()
        test_df = df[split_values == "test"].copy()
        notes = ["existing split column was used"]
        if train_df.empty:
            raise ValueError("Existing split column produced an empty train split.")
        if val_df.empty:
            notes.append("existing split column produced an empty validation split")
        if test_df.empty:
            notes.append("existing split column produced an empty test split")
        return train_df, val_df, test_df, notes

    def _split_randomly(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
        notes = ["random train validation test split was created"]
        if len(df) < 5:
            notes.append("dataset was too small for separate validation and test splits")
            return df.copy(), df.iloc[0:0].copy(), df.iloc[0:0].copy(), notes

        stratify = self._stratify_or_none(df[self.config.target_col], notes)
        try:
            train_val_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=stratify,
            )
        except ValueError as exc:
            notes.append(f"test split stratification skipped: {exc}")
            train_val_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=None,
            )

        relative_val_size = self.config.val_size / (1.0 - self.config.test_size)
        stratify_train_val = self._stratify_or_none(
            train_val_df[self.config.target_col],
            notes,
        )
        try:
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=relative_val_size,
                random_state=self.config.random_state,
                stratify=stratify_train_val,
            )
        except ValueError as exc:
            notes.append(f"validation split skipped: {exc}")
            train_df = train_val_df
            val_df = train_val_df.iloc[0:0]

        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
            list(dict.fromkeys(notes)),
        )

    def _stratify_or_none(self, y: pd.Series, notes: list[str]) -> pd.Series | None:
        if not self.config.use_stratified_split:
            return None
        counts = y.value_counts(dropna=True)
        if len(counts) < 2 or counts.min() < 2:
            notes.append("stratified split skipped because class support was insufficient")
            return None
        return y

    @staticmethod
    def _feature_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        feature_df = df[feature_cols].copy()
        for column in feature_df.columns:
            if is_bool_dtype(feature_df[column]):
                feature_df[column] = feature_df[column].astype(int)
        return feature_df

    @staticmethod
    def _is_numeric_or_bool(series: pd.Series) -> bool:
        return is_numeric_dtype(series) or is_bool_dtype(series)

    def _is_leakage_sensitive(self, column: str) -> bool:
        if column == self.config.target_col:
            return True
        column_lower = column.lower()
        if column_lower in LEAKAGE_SENSITIVE_MULTIMODAL_COLUMNS:
            return True
        leakage_terms = (
            "likely_dre",
            "missing_evidence",
            "definition_version",
            "reasons",
            "reason",
        )
        return any(term in column_lower for term in leakage_terms)

    @staticmethod
    def _positive_class_scores(model: Pipeline, X: pd.DataFrame) -> list[float]:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1].tolist()
        return model.decision_function(X).tolist()

    @staticmethod
    def _extract_modality_flag_cols(df: pd.DataFrame) -> list[str]:
        return [column for column in MODALITY_FLAG_COLUMNS if column in df.columns]

    @staticmethod
    def _single_modality_row_count(
        df: pd.DataFrame,
        modality_flag_cols: list[str],
    ) -> int:
        if not modality_flag_cols:
            return 0
        flag_df = df[modality_flag_cols].fillna(False).astype(bool)
        return int((flag_df.sum(axis=1) < 2).sum())


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    return value

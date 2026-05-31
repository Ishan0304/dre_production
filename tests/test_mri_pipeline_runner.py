import json
from datetime import UTC, datetime

import pandas as pd

from core.contracts import RunContextRecord
from datasets import MRIPipelineInputConfig, MRIPipelineRunner


class FakeMRIFeatureBuilder:
    def build_feature_table(self, inventory_df, participants_df=None):
        rows = [
            {
                "subject_id": row["subject_id"],
                "t1_path": row["t1_path"],
                "feature_a": 1.0,
            }
            for _, row in inventory_df.iterrows()
            if row["has_t1"]
        ]
        feature_df = pd.DataFrame(rows)
        if participants_df is not None and not participants_df.empty:
            feature_df = feature_df.merge(
                participants_df.rename(columns={"participant_id": "subject_id"}),
                on="subject_id",
                how="left",
            )
        error_df = pd.DataFrame(columns=["subject_id", "error_type", "error_message"])
        return feature_df, error_df


def _run_context() -> RunContextRecord:
    return RunContextRecord(
        run_id="run-mri-001",
        project_name="dre_production",
        stage_name="mri_pipeline",
        timestamp_utc=datetime(2026, 5, 30, tzinfo=UTC),
    )


def test_mri_pipeline_run_writes_outputs(tmp_path) -> None:
    dataset_root = tmp_path / "openneuro"
    anat_dir = dataset_root / "sub-001" / "anat"
    anat_dir.mkdir(parents=True)
    (anat_dir / "sub-001_T1w.nii.gz").write_text("fake", encoding="utf-8")
    pd.DataFrame({"participant_id": ["sub-001"], "age": [30]}).to_csv(
        dataset_root / "participants.tsv",
        sep="\t",
        index=False,
    )
    output_dir = tmp_path / "outputs"

    result = MRIPipelineRunner(feature_builder=FakeMRIFeatureBuilder()).run(
        MRIPipelineInputConfig(
            dataset_name="mri_features",
            dataset_root=str(dataset_root),
            output_dir=str(output_dir),
        ),
        _run_context(),
    )

    assert result.subject_row_count == 1
    assert result.error_count == 0
    assert result.subject_dataset_path is not None
    assert result.dataset_profile_path is not None
    assert result.run_manifest_path is not None
    manifest = json.loads(open(result.run_manifest_path, encoding="utf-8").read())
    assert manifest["datasets"][0]["modality"] == "mri"

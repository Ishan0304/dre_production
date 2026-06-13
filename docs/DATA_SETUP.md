# Data Setup

This repository supports public demonstration runs and credentialed clinical data runs for identifying patients who likely have drug-resistant epilepsy from longitudinal patient data. The `likely_dre` label is a computable inference for research and engineering review, not a formal adjudicated ILAE diagnosis.

Raw datasets are intentionally not committed. Forked users should place or bootstrap data under `data/raw/`, validate paths, then run the package pipelines.

## Datasets

| Dataset | Modality | Access | Expected local path |
| --- | --- | --- | --- |
| MIMIC-IV Demo | EHR | Public PhysioNet demo data | `data/raw/mimic_demo/` |
| MIMIC-IV full | EHR | Credentialed PhysioNet access required | user supplied, not auto-downloaded |
| OpenNeuro ds000030 | MRI | Public OpenNeuro data | `data/raw/openneuro/ds000030/` |
| CHB-MIT | EEG | Public PhysioNet data | `data/raw/chbmit/` |

## What To Do After Cloning

1. Create an environment with Python 3.12 or newer.
2. Install the package dependencies.
3. Bootstrap public datasets where appropriate.
4. Validate local paths before running pipelines.
5. Run EHR, MRI, EEG, and multimodal pipelines from package code or thin orchestration scripts.

Example setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

## Bootstrap Public Data

The bootstrap script supports public demo data only. It does not download credentialed MIMIC-IV full data.

```powershell
python scripts\bootstrap_public_data.py --dataset mimic_demo
python scripts\bootstrap_public_data.py --dataset chbmit
python scripts\bootstrap_public_data.py --dataset openneuro
```

OpenNeuro bootstrapping requires either the OpenNeuro CLI or DataLad. On some Windows setups, the OpenNeuro CLI may authenticate successfully but stall before materializing files. In that case, use a known working OpenNeuro download method, then ensure the extracted dataset root is:

```text
data/raw/openneuro/ds000030/
```

## Validate Data Paths

Run:

```powershell
python scripts\validate_data_paths.py
```

The validator checks the standard local paths for MIMIC Demo, OpenNeuro, and CHB-MIT. It reports present and missing datasets and prints row or file counts where practical.

## Expected Dataset Structure

Lightweight manifests live under `data/manifests/`:

```text
data/manifests/mimic_demo_expected_paths.txt
data/manifests/openneuro_expected_paths.txt
data/manifests/chbmit_expected_paths.txt
```

These files describe expected local paths and representative required files. They are safe to commit because they contain no patient data or raw dataset contents.

## Running Pipelines

The example configs point to repo-local data paths:

```text
configs/ehr_example.yaml
configs/mri_example.yaml
configs/eeg_example.yaml
configs/multimodal_example.yaml
```

Until a dedicated CLI is added, run pipelines through the package runner classes with `PYTHONPATH=src`.

EHR pipeline:

```powershell
$env:PYTHONPATH = "src"
python -c "from datetime import UTC, datetime; from pathlib import Path; import yaml; from core.contracts import RunContextRecord; from datasets.ehr_pipeline_runner import EHRPipelineInputConfig, EHRPipelineRunner; from ingestion.ehr_loader import TableLoadRequest; cfg=yaml.safe_load(Path('configs/ehr_example.yaml').read_text()); ehr=cfg['ehr']; inputs=ehr['inputs']; diagnoses=TableLoadRequest(path=inputs['diagnoses']['path'], required_columns=inputs['diagnoses'].get('required_columns'), parse_date_columns=inputs['diagnoses'].get('parse_date_columns')); medications=TableLoadRequest(path=inputs['medications']['path'], required_columns=inputs['medications'].get('required_columns'), parse_date_columns=inputs['medications'].get('parse_date_columns')); run_cfg=EHRPipelineInputConfig(dataset_name=ehr['dataset_name'], diagnoses_request=diagnoses, medications_request=medications, seizure_events_request=None, output_dir=ehr['outputs']['pipeline_dir']); ctx=RunContextRecord(run_id='ehr_example_run', project_name=cfg['project']['name'], stage_name='ehr_pipeline', seed=42, timestamp_utc=datetime.now(UTC)); print(EHRPipelineRunner().run(run_cfg, ctx).to_dict())"
```

MRI pipeline:

```powershell
$env:PYTHONPATH = "src"
python -c "from datetime import UTC, datetime; from pathlib import Path; import yaml; from core.contracts import RunContextRecord; from datasets.mri_pipeline_runner import MRIPipelineInputConfig, MRIPipelineRunner; cfg=yaml.safe_load(Path('configs/mri_example.yaml').read_text()); mri=cfg['mri']; run_cfg=MRIPipelineInputConfig(dataset_name=mri['dataset_name'], dataset_root=mri['dataset_root'], output_dir=mri['output_dir']); ctx=RunContextRecord(run_id='mri_example_run', project_name=cfg['project']['name'], stage_name='mri_pipeline', seed=42, timestamp_utc=datetime.now(UTC)); print(MRIPipelineRunner().run(run_cfg, ctx).to_dict())"
```

EEG pipeline:

```powershell
$env:PYTHONPATH = "src"
python -c "from datetime import UTC, datetime; from pathlib import Path; import yaml; from core.contracts import RunContextRecord; from datasets.eeg_pipeline_runner import EEGPipelineInputConfig, EEGPipelineRunner; cfg=yaml.safe_load(Path('configs/eeg_example.yaml').read_text()); eeg=cfg['eeg']; run_cfg=EEGPipelineInputConfig(dataset_name=eeg['dataset_name'], dataset_root=eeg['dataset_root'], output_dir=eeg['output_dir']); ctx=RunContextRecord(run_id='eeg_example_run', project_name=cfg['project']['name'], stage_name='eeg_pipeline', seed=42, timestamp_utc=datetime.now(UTC)); print(EEGPipelineRunner().run(run_cfg, ctx).to_dict())"
```

Multimodal pipeline:

```powershell
$env:PYTHONPATH = "src"
python -c "from datetime import UTC, datetime; from pathlib import Path; import yaml; from core.contracts import RunContextRecord; from datasets.ehr_pipeline_runner import EHRPipelineInputConfig; from datasets.mri_pipeline_runner import MRIPipelineInputConfig; from datasets.eeg_pipeline_runner import EEGPipelineInputConfig; from datasets.multimodal_pipeline_runner import MultimodalPipelineInputConfig, MultimodalPipelineRunner; from ingestion.ehr_loader import TableLoadRequest; cfg=yaml.safe_load(Path('configs/multimodal_example.yaml').read_text()); inputs=cfg['ehr']['inputs']; ehr=EHRPipelineInputConfig(dataset_name=cfg['ehr']['dataset_name'], diagnoses_request=TableLoadRequest(path=inputs['diagnoses']['path'], required_columns=inputs['diagnoses'].get('required_columns'), parse_date_columns=inputs['diagnoses'].get('parse_date_columns')), medications_request=TableLoadRequest(path=inputs['medications']['path'], required_columns=inputs['medications'].get('required_columns'), parse_date_columns=inputs['medications'].get('parse_date_columns')), seizure_events_request=None, output_dir=cfg['ehr']['output_dir']); mri=MRIPipelineInputConfig(dataset_name=cfg['mri']['dataset_name'], dataset_root=cfg['mri']['dataset_root'], output_dir=cfg['mri']['output_dir']); eeg=EEGPipelineInputConfig(dataset_name=cfg['eeg']['dataset_name'], dataset_root=cfg['eeg']['dataset_root'], output_dir=cfg['eeg']['output_dir']); run_cfg=MultimodalPipelineInputConfig(dataset_name=cfg['multimodal']['dataset_name'], output_dir=cfg['multimodal']['output_dir'], ehr_input_config=ehr, mri_input_config=mri, eeg_input_config=eeg, run_multimodal_model=cfg['multimodal'].get('run_multimodal_model', True)); ctx=RunContextRecord(run_id='multimodal_example_run', project_name=cfg['project']['name'], stage_name='multimodal_pipeline', seed=42, timestamp_utc=datetime.now(UTC)); print(MultimodalPipelineRunner().run(run_cfg, ctx).to_dict())"
```

If the multimodal helper API changes, keep orchestration thin and call the package runner classes rather than placing logic in notebooks.

## Outputs And Examples

Full run outputs are generated under `outputs/` and are ignored by default. A small set of shareable example figures may be committed under:

```text
outputs/reports/examples/
```

Do not commit raw MIMIC, OpenNeuro, or CHB-MIT data.

import os
import pytest

def test_dvc_yaml_exists():
    assert os.path.exists("dvc.yaml"), "dvc.yaml pipeline configuration file is missing"

def test_dvc_yaml_structure():
    import yaml
    with open("dvc.yaml", "r") as f:
        config = yaml.safe_load(f)

    assert "stages" in config
    assert "train" in config["stages"]
    assert "drift_report" in config["stages"]

    train_stage = config["stages"]["train"]
    assert "cmd" in train_stage
    assert "deps" in train_stage
    assert "outs" in train_stage
    assert "data/train.csv" in train_stage["deps"]
    assert "model/model.pkl" in train_stage["outs"]

    drift_stage = config["stages"]["drift_report"]
    assert "cmd" in drift_stage
    assert "artifacts/drift_report.html" in drift_stage["outs"]

def test_dvcignore_exists():
    assert os.path.exists(".dvcignore"), ".dvcignore file is missing"

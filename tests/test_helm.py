import os
import pytest
import yaml

HELM_DIR = "helm/houseprice-estimator"

def test_helm_chart_metadata():
    chart_path = os.path.join(HELM_DIR, "Chart.yaml")
    assert os.path.exists(chart_path), "Chart.yaml is missing"

    with open(chart_path, "r") as f:
        chart = yaml.safe_load(f)

    assert chart["name"] == "houseprice-estimator"
    assert chart["apiVersion"] == "v2"
    assert "version" in chart
    assert "appVersion" in chart

def test_helm_values_structure():
    values_path = os.path.join(HELM_DIR, "values.yaml")
    assert os.path.exists(values_path), "values.yaml is missing"

    with open(values_path, "r") as f:
        values = yaml.safe_load(f)

    assert "replicaCount" in values
    assert "image" in values
    assert "repository" in values["image"]
    assert "tag" in values["image"]
    assert "service" in values
    assert values["service"]["port"] == 8000
    assert "resources" in values
    assert "livenessProbe" in values
    assert "readinessProbe" in values

def test_helm_templates_exist():
    templates_dir = os.path.join(HELM_DIR, "templates")
    assert os.path.exists(templates_dir)
    assert os.path.exists(os.path.join(templates_dir, "deployment.yaml"))
    assert os.path.exists(os.path.join(templates_dir, "service.yaml"))
    assert os.path.exists(os.path.join(templates_dir, "_helpers.tpl"))

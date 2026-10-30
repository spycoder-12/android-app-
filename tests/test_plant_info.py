import json
import pytest
from services.plant_info_service import PlantInfoError, PlantInfoService

FIELDS = ["name", "category", "summary", "symptoms", "prevention", "care", "expert_advice"]


def test_every_disease_has_all_fields():
    svc = PlantInfoService()
    items = svc.all("tomato")
    assert len(items) == 10
    for i in items:
        for f in FIELDS:
            assert i[f], f"{i['key']} missing {f}"


def test_no_dosage_or_product_prescriptions():
    text = json.dumps(PlantInfoService().all()).lower()
    for bad in (" ml ", " mg ", "per litre", "per liter", "tablespoon", "spray every", "dose"):
        assert bad not in text


def test_search():
    svc = PlantInfoService()
    assert any(i["key"] == "late_blight" for i in svc.search("blight"))
    assert [i["key"] for i in svc.search("whitefly")] == ["yellow_leaf_curl_virus"]
    assert svc.search("zzzzqq") == []
    assert len(svc.search("")) == 10 and len(svc.search("   ")) == 10


def test_get_unknown():
    svc = PlantInfoService()
    assert svc.get("tomato", "nope") is None and svc.get("potato", "healthy") is None


def test_missing_or_corrupt_data(tmp_path):
    with pytest.raises(PlantInfoError):
        PlantInfoService(str(tmp_path))
    (tmp_path / "diseases.json").write_text("{bad json")
    (tmp_path / "care_tips.json").write_text("{}")
    with pytest.raises(PlantInfoError):
        PlantInfoService(str(tmp_path))

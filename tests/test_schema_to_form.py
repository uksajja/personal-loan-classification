from src.utils.schema_to_form import build_form_fields


def test_build_form_fields_from_yaml_schema():
    schema = {
        "columns": {
            "Age": {"dtype": "int64", "greater_than_or_equal_to": 23},
            "Family": {"dtype": "int64", "isin": [1, 2, 3, 4]},
            "CCAvg": {"dtype": "float64", "greater_than_or_equal_to": 0.0},
            "Online": {"dtype": "int64", "isin": [0, 1]},
        }
    }

    fields = build_form_fields(schema)

    assert [field["name"] for field in fields] == ["Age", "Family", "CCAvg", "Online"]
    assert fields[0]["input_type"] == "number"
    assert fields[0]["default"] == 23
    assert fields[1]["options"] == [1, 2, 3, 4]
    assert fields[2]["step"] == "0.01"
    assert fields[3]["options"] == [0, 1]

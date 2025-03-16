from pathlib import Path
from idlelib.browser import file_open

import yaml
import pytest

from src.pp5.utils import YamlDict


@pytest.fixture
def temp_yaml_file(tmp_path):
    return tmp_path / "test.yaml"


def assert_yaml_dict_contents(yaml_dict, expected_data, temp_yaml_file):
    file_data = {}
    if Path(temp_yaml_file).exists():
        with open(temp_yaml_file, "r") as f:
            file_data = yaml.safe_load(f)
    assert yaml_dict == expected_data
    assert file_data == expected_data


class TestYamlDict:
    def test_yaml_dict_loads_existing_data(self, temp_yaml_file):
        data = {"key1": "value1", "key2": "value2"}
        with open(temp_yaml_file, "w") as f:
            yaml.dump(data, f)

        yaml_dict = YamlDict(temp_yaml_file)
        assert_yaml_dict_contents(yaml_dict, data, temp_yaml_file)

    def test_yaml_dict_saves_data_on_setitem(self, temp_yaml_file):
        yaml_dict = YamlDict(temp_yaml_file)
        yaml_dict["key1"] = "value1"
        assert_yaml_dict_contents(yaml_dict, {"key1": "value1"}, temp_yaml_file)

    def test_yaml_dict_saves_data_on_delitem(self, temp_yaml_file):
        yaml_dict = YamlDict(temp_yaml_file)
        yaml_dict["key1"] = "value1"
        del yaml_dict["key1"]
        assert_yaml_dict_contents(yaml_dict, {}, temp_yaml_file)

    def test_yaml_dict_initializes_empty_if_no_file(self, temp_yaml_file):
        yaml_dict = YamlDict(temp_yaml_file)
        assert_yaml_dict_contents(yaml_dict, {}, temp_yaml_file)

    def test_yaml_dict_updates_existing_data(self, temp_yaml_file):
        data = {"key1": "value1"}
        with open(temp_yaml_file, "w") as f:
            yaml.dump(data, f)

        yaml_dict = YamlDict(temp_yaml_file)
        yaml_dict["key1"] = "new_value"
        assert_yaml_dict_contents(yaml_dict, {"key1": "new_value"}, temp_yaml_file)

    def test_yaml_dict_pop(self, temp_yaml_file):
        yaml_dict = YamlDict(temp_yaml_file)
        yaml_dict["key1"] = "value1"
        value = yaml_dict.pop("key1")
        assert value == "value1"
        assert_yaml_dict_contents(yaml_dict, {}, temp_yaml_file)

    def test_yaml_dict_update(self, temp_yaml_file):
        yaml_dict = YamlDict(temp_yaml_file)
        yaml_dict.update({"key1": "value1", "key2": "value2"})
        assert_yaml_dict_contents(
            yaml_dict, {"key1": "value1", "key2": "value2"}, temp_yaml_file
        )

    def test_yaml_dict_setdefault(self, temp_yaml_file):
        yaml_dict = YamlDict(temp_yaml_file)
        default_value = yaml_dict.setdefault("key1", "default")
        assert default_value == "default"
        assert_yaml_dict_contents(yaml_dict, {"key1": "default"}, temp_yaml_file)

    def test_yaml_dict_clear(self, temp_yaml_file):
        yaml_dict = YamlDict(temp_yaml_file)
        yaml_dict["key1"] = "value1"
        yaml_dict.clear()
        assert_yaml_dict_contents(yaml_dict, {}, temp_yaml_file)

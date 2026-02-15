# This is used to test if our CLI works 
# 1. test parsing
# 2. test overriding 

import pytest
import yaml
from vibes_pipe.cli.infer import main


def test_infer_cli_overrides(tmp_path):
    # 1. SETUP: Create a temporary dummy config
    config_file = tmp_path / "config.yaml"
    initial_data = {"model": {"batch_size": 1}, "io": {"out_dir": str(tmp_path / "out")}}
    config_file.write_text(yaml.dump(initial_data))

    args = [
        "--config", str(config_file),
        "--set", "model.batch_size=64",
        "--dry-run"
    ]
    
    # This ensures the code runs to completion without crashing
    main(args)
    

def test_infer_cli_logic_overrides(tmp_path, capsys):
    # SETUP
    config_file = tmp_path / "config.yaml"
    initial_data = {"model": {"batch_size": 1}}
    config_file.write_text(yaml.dump(initial_data))

    # EXECUTE: We want to change batch_size to 64
    args = ["--config", str(config_file), "--set", "model.batch_size=64", "--dry-run"]
    main(args)

    # VERIFY: capsys captures whatever was printed to the terminal
    captured = capsys.readouterr()
    final_cfg = yaml.safe_load(captured.out)
    
    assert final_cfg["model"]["batch_size"] == 64
    assert final_cfg["run"]["config_path"] == str(config_file.resolve())
    
    
def test_infer_cli_missing_config():
    # Test that it fails if no config is provided (argparse logic)
    with pytest.raises(SystemExit):
        main([]) # No arguments should trigger a help message and exit


def test_infer_cli_bad_set_format(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("model: {}")
    
    args = ["--config", str(config_file), "--set", "bad_format_no_equals"]
    with pytest.raises(ValueError, match="--set must be key=value"):
        main(args)
        
        
def test_infer_cli_list_overrides(tmp_path, capsys):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("infer: {window_size: [1, 1, 1]}")

    # Passing a list via CLI
    args = ["--config", str(config_file), "--set", "infer.window_size=[96,96,48]", "--dry-run"]
    main(args)

    captured = capsys.readouterr()
    final_cfg = yaml.safe_load(captured.out)
    
    # Verify it's an actual list [96, 96, 48], not the string "[96,96,48]"
    assert final_cfg["infer"]["window_size"] == [96, 96, 48]
    assert isinstance(final_cfg["infer"]["window_size"], list)
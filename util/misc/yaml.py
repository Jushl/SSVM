import glob
from pathlib import Path
import re
import yaml
import inspect


# def guess_model_scale(model_path):
#     try:
#         return re.search(r"smvm?\d+([nslmx])", Path(model_path).stem).group(1)
#     except AttributeError:
#         return ""


def yaml_load(file="data.yaml", append_filename=False):
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        data = yaml.safe_load(s) or {}
        if append_filename:
            data["yaml_file"] = str(file)
        return data


def yaml_model_load(path):
    path = Path(path)
    d = yaml_load(path)

    match = re.search(r"-([^-]+)$", path.stem)
    if match:
        d["scale"] = match.group(1)
    # d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d
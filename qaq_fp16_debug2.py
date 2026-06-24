import json
from qaq.config import load_config_file
from qaq.runtime.static import run_static_runtime
from qaq.results import build_result_artifact, save_result_artifact

config = load_config_file(
    "configs/benchmarks/llama_first_milestone/hellaswag/fp16.json",
    validate_output=False,
)

result = run_static_runtime(config, example_limit=2)
artifact = build_result_artifact(config, result)

out = "runs/llama_first_milestone/hellaswag/fp16_debug2/result_artifact.json"
save_result_artifact(artifact, out)

print(json.dumps(artifact.as_dict(), indent=2, sort_keys=True))
print("saved:", out)

"""Promote a registered Food-11 model only when it clears a quality gate."""

import argparse
import math
import os

import mlflow

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_TEST_SIZES = {"processed": 3347, "mini": 1096}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--model-name", default="food11")
    parser.add_argument("--test-size", type=int)
    args = parser.parse_args()
    if args.test_size is not None and args.test_size < 1:
        parser.error("--test-size must be at least 1")
    return args


def test_size_for_run(run, explicit_size: int | None) -> int:
    if explicit_size is not None:
        return explicit_size
    if "test_size" in run.data.metrics:
        return int(run.data.metrics["test_size"])
    if "test_size" in run.data.params:
        return int(run.data.params["test_size"])
    dataset = run.data.params.get("dataset")
    if dataset in DEFAULT_TEST_SIZES:
        return DEFAULT_TEST_SIZES[dataset]
    raise ValueError("Cannot determine test size; pass --test-size explicitly")


def metric(run, name: str) -> float:
    if name not in run.data.metrics:
        raise ValueError(f"Run {run.info.run_id} does not contain metric {name!r}")
    return float(run.data.metrics[name])


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    client = mlflow.MlflowClient()

    candidate = client.get_model_version(args.model_name, args.version)
    champion = client.get_model_version_by_alias(args.model_name, "champion")
    candidate_run = client.get_run(candidate.run_id)
    champion_run = client.get_run(champion.run_id)
    candidate_accuracy = metric(candidate_run, "test_accuracy")
    champion_accuracy = metric(champion_run, "test_accuracy")
    test_size = test_size_for_run(champion_run, args.test_size)
    standard_error = math.sqrt(
        champion_accuracy * (1.0 - champion_accuracy) / test_size
    )
    required_accuracy = champion_accuracy + 2.0 * standard_error

    if candidate_accuracy >= required_accuracy:
        decision = "promoted"
        client.set_registered_model_alias(
            args.model_name, "champion", candidate.version
        )
    else:
        decision = "challenger"
        client.set_registered_model_alias(
            args.model_name, "challenger", candidate.version
        )

    tags = {
        "promotion.decision": decision,
        "promotion.candidate_accuracy": f"{candidate_accuracy:.8f}",
        "promotion.champion_accuracy": f"{champion_accuracy:.8f}",
        "promotion.required_accuracy": f"{required_accuracy:.8f}",
        "promotion.test_size": str(test_size),
        "promotion.champion_version_before": champion.version,
    }
    for key, value in tags.items():
        client.set_model_version_tag(args.model_name, candidate.version, key, value)

    print(f"decision={decision}")
    print(f"candidate_accuracy={candidate_accuracy:.6f}")
    print(f"champion_accuracy={champion_accuracy:.6f}")
    print(f"two_standard_errors={2.0 * standard_error:.6f}")
    print(f"required_accuracy={required_accuracy:.6f}")


if __name__ == "__main__":
    main()

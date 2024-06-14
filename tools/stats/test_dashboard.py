import json
import os
import re
import subprocess
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast, Dict, List, Optional, Tuple

import requests

from tools.stats.upload_stats_lib import (
    _get_request_headers,
    download_s3_artifacts,
    get_job_id,
    get_job_ids_for_paths,
    get_tests,
    unzip,
    upload_workflow_stats_to_s3,
)

REGEX_JOB_INFO = r"(.*) \/ .*test \(([^,]*), .*\)"


@lru_cache(maxsize=1000)
def get_job_name(job_id: int) -> str:
    try:
        return cast(
            str,
            requests.get(
                f"https://api.github.com/repos/pytorch/pytorch/actions/jobs/{job_id}",
                headers=_get_request_headers(),
            ).json()["name"],
        )
    except Exception as e:
        print(f"Failed to get job name for job id {job_id}: {e}")
        return "NoJobName"


@lru_cache(maxsize=1000)
def get_build_name(job_name: str) -> str:
    try:
        return re.match(REGEX_JOB_INFO, job_name).group(1)  # type: ignore[union-attr]
    except AttributeError:
        print(f"Failed to match job name: {job_name}")
        return "NoBuildEnv"


@lru_cache(maxsize=1000)
def get_test_config(job_name: str) -> str:
    try:
        return re.match(REGEX_JOB_INFO, job_name).group(2)  # type: ignore[union-attr]
    except AttributeError:
        print(f"Failed to match job name: {job_name}")
        return "NoTestConfig"


def get_td_exclusions(
    workflow_run_id: int, workflow_run_attempt: int
) -> Dict[str, Any]:
    with TemporaryDirectory() as temp_dir:
        print("Using temporary directory:", temp_dir)
        current_dir = os.getcwd()
        os.chdir(temp_dir)

        # Download and extract all the reports (both GHA and S3)
        s3_paths = download_s3_artifacts(
            "test-jsons", workflow_run_id, workflow_run_attempt
        )
        for path in s3_paths:
            unzip(path)

        grouped_tests: Dict[str, Any] = defaultdict(lambda: defaultdict(set))
        for td_exclusions, job_id in get_job_ids_for_paths(
            list(Path(".").glob("**/td_exclusions*.json")),
            workflow_run_id,
            workflow_run_attempt,
        ):
            with open(td_exclusions) as f:
                exclusions = json.load(f)
                for exclusion in exclusions["excluded"]:
                    job_id = get_job_id(td_exclusions)
                    job_name = get_job_name(job_id)
                    build_name = get_build_name(job_name)
                    test_config = get_test_config(job_name)
                    grouped_tests[build_name][test_config].add(exclusion["test_file"])

        for build_name, build in grouped_tests.items():
            for test_config, test_files in build.items():
                grouped_tests[build_name][test_config] = sorted(test_files)
        os.chdir(current_dir)
        return grouped_tests


def group_test_cases(test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    start = time.time()
    grouped_tests: Dict[str, Any] = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )
    )
    for test_case in test_cases:
        job_name = get_job_name(test_case["job_id"])
        build_name = get_build_name(job_name)
        if "bazel" in build_name:
            continue
        test_config = get_test_config(job_name)
        class_name = test_case.pop("classname", "NoClass")
        name = test_case.pop("name", "NoName")
        invoking_file = test_case.pop("invoking_file", "NoFile")
        invoking_file = invoking_file.replace(".", "/")
        test_case.pop("workflow_id")
        test_case.pop("workflow_run_attempt")
        grouped_tests[build_name][test_config][invoking_file][class_name][name].append(
            test_case
        )

    print(f"Time taken to group tests: {time.time() - start}")
    return grouped_tests


def get_reruns(grouped_tests: Dict[str, Any]) -> Dict[str, Any]:
    reruns: Dict[str, Any] = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )
    )
    for build_name, build in grouped_tests.items():
        for test_config, test_config_data in build.items():
            for invoking_file, invoking_file_data in test_config_data.items():
                for class_name, class_data in invoking_file_data.items():
                    for test_name, test_data in class_data.items():
                        if len(test_data) > 1:
                            if invoking_file in (
                                "distributed/test_distributed_spawn",
                                "onnx/test_fx_to_onnx_with_onnxruntime",
                                "distributed/algorithms/quantization/test_quantization",
                            ):
                                continue
                            reruns[build_name][test_config][invoking_file][class_name][
                                test_name
                            ] = test_data
    return reruns


def get_invoking_file_summary(grouped_tests: Dict[str, Any]) -> Dict[str, Any]:
    invoking_file_summary: Dict[str, Any] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"count": 0, "time": 0.0}))
    )
    for build_name, build in grouped_tests.items():
        for test_config, test_config_data in build.items():
            for invoking_file, invoking_file_data in test_config_data.items():
                for class_data in invoking_file_data.values():
                    for test_data in class_data.values():
                        invoking_file_summary[build_name][test_config][invoking_file][
                            "count"
                        ] += 1
                        for i in test_data:
                            invoking_file_summary[build_name][test_config][
                                invoking_file
                            ]["time"] += i["time"]

    return invoking_file_summary


def get_new_removed_tests(
    grouped: Dict[str, Any], base_grouped: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    def get_a_minus_b(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        if any(isinstance(a[key], list) for key in a):
            diff: Dict[str, Any] = {
                "count": 0,
                "nodes": [],
                "total": len(a),
            }
            for key in a:
                if key not in b:
                    diff["nodes"].append(key)
                    diff["count"] += 1
            diff["nodes"] = diff["nodes"][:10]
            return diff
        diff = {
            "count": 0,
            "nodes": {},
        }
        for key in a:
            small_diff = get_a_minus_b(a[key], b.get(key, {}))
            if small_diff["count"] == small_diff["total"]:
                small_diff["nodes"] = []
            if small_diff["count"] > 0:
                diff["nodes"][key] = small_diff
        diff["count"] = sum(diff["nodes"][key]["count"] for key in diff["nodes"])
        diff["total"] = sum(diff["nodes"][key]["total"] for key in diff["nodes"])
        return diff

    return get_a_minus_b(grouped, base_grouped), get_a_minus_b(base_grouped, grouped)


def compare(
    job_summary: Dict[str, Any], base_job_summary: Dict[str, Any], base_job_id: int
) -> Dict[str, Any]:
    start = time.time()
    new, removed = get_new_removed_tests(job_summary, base_job_summary)
    print(f"Time taken to compare tests: {time.time() - start}")
    return {
        "base_job_id": base_job_id,
        "new": new,
        "removed": removed,
    }


def get_base_id(sha: str, workflow_id: int) -> Optional[int]:
    try:
        if sha is None:
            return None
        base_sha = (
            subprocess.check_output(["git", "merge-base", "origin/main", sha])
            .decode("utf-8")
            .strip()
        )
        if base_sha == sha:
            base_sha = (
                subprocess.check_output(["git", "rev-parse", f"{sha}^"])
                .decode("utf-8")
                .strip()
            )
        return cast(
            int,
            requests.get(
                "https://hud.pytorch.org/api/corresponding_workflow_id",
                params={"sha": base_sha, "workflowId": workflow_id},  # type: ignore[arg-type]
            ).json()[0]["id"],
        )
    except Exception as e:
        print(
            f"Failed to get base id for head sha {sha} and workflow id {workflow_id}: {e}"
        )
        return None


def upload_wrapper(
    workflow_run_id: int, workflow_run_attempt: int, name: str, data: Any
) -> None:
    as_string = json.dumps(data)
    if len(as_string) > 1000000:
        data = [{"info": "Data too large to upload"}]
        print("Data too large to upload")
    upload_workflow_stats_to_s3(
        workflow_run_id,
        workflow_run_attempt,
        name,
        [data],
    )


def upload_additional_info(
    workflow_run_id: int,
    workflow_run_attempt: int,
    head_sha: str,
    test_cases: List[Dict[str, Any]],
) -> None:
    grouped_tests = group_test_cases(test_cases)
    reruns = get_reruns(grouped_tests)
    exclusions = get_td_exclusions(workflow_run_id, workflow_run_attempt)
    invoking_file_summary = get_invoking_file_summary(grouped_tests)

    base_id = get_base_id(head_sha, workflow_run_id)
    if base_id is not None:
        base_grouped_tests = group_test_cases(get_tests(base_id, 1))
        new_removed_tests = compare(grouped_tests, base_grouped_tests, base_id)
    else:
        new_removed_tests = {
            "base_job_id": None,
            "new": {"count": 0, "nodes": []},
            "removed": {"count": 0, "nodes": []},
        }

    upload_wrapper(
        workflow_run_id,
        workflow_run_attempt,
        "additional_info/reruns",
        [reruns],
    )
    upload_wrapper(
        workflow_run_id,
        workflow_run_attempt,
        "additional_info/td_exclusions",
        [exclusions],
    )
    upload_wrapper(
        workflow_run_id,
        workflow_run_attempt,
        "additional_info/invoking_file_summary",
        [invoking_file_summary],
    )
    upload_wrapper(
        workflow_run_id,
        workflow_run_attempt,
        "additional_info/new_removed_tests",
        [new_removed_tests],
    )

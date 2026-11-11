import importlib.util
import os
import shutil
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy/nautilus"
spec = importlib.util.spec_from_file_location("jepa_plan", ROOT / "src/step1/jepa_plan.py")
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)


class NautilusTests(unittest.TestCase):
    def test_step1_mounts_current_source_and_runs_checks_in_order(self):
        source = ROOT / "src/step1"
        job = yaml.safe_load((source / "job.yaml").read_text())
        self.assertEqual(job["spec"]["backoffLimit"], 0)
        self.assertGreater(job["spec"]["activeDeadlineSeconds"], 0)
        pod = job["spec"]["template"]["spec"]
        self.assertFalse(pod["automountServiceAccountToken"])
        self.assertEqual(pod["containers"][0]["command"], ["/bin/bash", "/opt/step1/run.sh"])
        for container in pod["containers"] + pod["initContainers"]:
            self.assertEqual(container["resources"]["requests"], container["resources"]["limits"])
            self.assertTrue(any(m["mountPath"] == "/opt/step1" and m["readOnly"]
                                for m in container["volumeMounts"]))
        script = (source / "run.sh").read_text()
        commands = ["jepa_smoke.py\n", "--dataset pointmaze\n", "inspect_data.py\n",
                    "jepa_plan.py --episodes 2\n", 'touch "$run_dir/SUCCESS"']
        positions = [script.index(command) for command in commands]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("set -euo pipefail", script)

    @unittest.skipUnless(shutil.which("kubectl"), "kubectl is required to render the A40 overlay")
    def test_a40_overlay_requests_one_gpu_and_inherits_eula(self):
        rendered = subprocess.run(["kubectl", "kustomize", str(ROOT / "deploy/nautilus-a40")],
                                  check=True, capture_output=True, text=True)
        pod = yaml.safe_load(rendered.stdout)["spec"]["template"]["spec"]
        for key in ("requests", "limits"):
            resources = pod["containers"][0]["resources"][key]
            self.assertEqual(resources["nvidia.com/a40"], "1")
            self.assertNotIn("nvidia.com/gpu", resources)
            self.assertEqual(resources["cpu"], "1")
        for item in pod["containers"] + pod["initContainers"]:
            self.assertIn({"name": "OMNI_KIT_ACCEPT_EULA", "value": "YES"}, item["env"])
        self.assertNotIn("tolerations", pod)

    def test_jobs_are_finite_and_preserve_results(self):
        for path in DEPLOY.glob("*-job.yaml"):
            with self.subTest(path=path.name):
                job = yaml.safe_load(path.read_text())
                # Kubernetes' YAML parser also treats bare Y/N as booleans.
                self.assertNotRegex(path.read_text(), r"(?m)^\s*value: (?:Y|N|yes|no|on|off)$")
                self.assertEqual(job["kind"], "Job")
                self.assertEqual(job["metadata"]["namespace"], "razvanlab")
                self.assertGreater(job["spec"]["activeDeadlineSeconds"], 0)
                pod = job["spec"]["template"]["spec"]
                self.assertEqual(pod["restartPolicy"], "Never")
                self.assertFalse(pod["automountServiceAccountToken"])
                self.assertNotIn("tolerations", pod)
                self.assertNotIn("hostNetwork", pod)
                self.assertTrue(any(v.get("persistentVolumeClaim", {}).get("claimName") == "mars-world-data"
                                    for v in pod["volumes"]))
                self.assertEqual(len(pod["initContainers"]), 1)
                setup = pod["initContainers"][0]
                self.assertNotIn("nvidia.com/gpu", setup["resources"]["requests"])
                self.assertEqual(setup["image"], pod["containers"][0]["image"])
                for container in pod["containers"] + pod["initContainers"]:
                    for key in ("cpu", "memory", "ephemeral-storage"):
                        self.assertIn(key, container["resources"]["requests"])
                    self.assertEqual(container["resources"]["requests"], container["resources"]["limits"])
                    for env in container.get("env", []):
                        self.assertEqual(set(env), {"name", "value"})
                        self.assertIsInstance(env["value"], str)
                    self.assertNotIn("sleep", " ".join(container.get("args", [])))
                    mounts = {m["mountPath"]: m["name"] for m in container["volumeMounts"]}
                    self.assertEqual(mounts["/runtime"], "runtime")
                    self.assertEqual(mounts["/data"], "data")
                runtime = next(v for v in pod["volumes"] if v["name"] == "runtime")
                self.assertIn("emptyDir", runtime)
                self.assertNotIn("medium", runtime["emptyDir"])

    def test_stacks_install_separately_and_data_download_is_lightweight(self):
        for name, mode in (("rover-job.yaml", "rover"), ("jepa-job.yaml", "jepa"),
                           ("jepa-plan-job.yaml", "jepa"), ("jepa-data-job.yaml", "data")):
            pod = yaml.safe_load((DEPLOY / name).read_text())["spec"]["template"]["spec"]
            self.assertEqual(pod["initContainers"][0]["args"], [mode])
        dockerfile = (ROOT / "docker/runtime.Dockerfile").read_text()
        self.assertNotIn("nvcr.io", dockerfile)
        self.assertNotIn("pip install", dockerfile)
        self.assertNotIn("COPY external", dockerfile)

    def test_image_size_gate_allows_reported_image_and_enforces_new_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            docker = Path(directory) / "docker"
            docker.write_text('#!/bin/sh\nprintf "%s\\n" "$TEST_IMAGE_SIZE"\n')
            docker.chmod(0o755)
            for size, expected in (("1094245367", 0), ("1199999999", 0), ("1200000000", 1), ("bad-size", 1)):
                env = dict(os.environ, PATH=directory + os.pathsep + os.environ["PATH"], TEST_IMAGE_SIZE=size)
                result = subprocess.run(["bash", str(ROOT / "docker/check-image-size.sh"), "test-image"],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)

    def test_isaac_requires_rtx_and_headless_execution(self):
        pod = yaml.safe_load((DEPLOY / "rover-job.yaml").read_text())["spec"]["template"]["spec"]
        terms = pod["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]
        for term in terms:
            products = next(e for e in term["matchExpressions"] if e["key"] == "nvidia.com/gpu.product")
            self.assertEqual(products["operator"], "In")
            rt_datacenter = {"NVIDIA-A10", "NVIDIA-A10G", "NVIDIA-L4", "NVIDIA-L40", "NVIDIA-L40S"}
            self.assertTrue(all("RTX" in name or name in rt_datacenter for name in products["values"]))
        container = pod["containers"][0]
        for item in pod["containers"] + pod["initContainers"]:
            environment = {entry["name"]: entry["value"] for entry in item["env"]}
            self.assertEqual(environment["OMNI_KIT_ACCEPT_EULA"], "YES")
        self.assertEqual(container["resources"]["limits"]["nvidia.com/gpu"], "1")
        args = container["args"]
        self.assertEqual(args[args.index("--viz") + 1], "none")
        self.assertGreater(int(args[args.index("--steps") + 1]), 0)
        self.assertNotIn("--headless", args)

    def test_planning_does_not_depend_on_unreleased_local_heads(self):
        source = ROOT / "external/jepa-wms/configs/evals/simu_env_planning/mz/jepa-wm/mz_L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml"
        original = yaml.safe_load(source.read_text())
        configured = plan.prepare_config(original, Path("/data/runs/test"), 2)
        self.assertEqual(original["model_kwargs"]["checkpoint"], "jepa-latest.pth.tar")
        self.assertTrue(configured["model_kwargs"]["checkpoint"].startswith("https://"))
        self.assertFalse(configured["model_kwargs"]["pretrain_kwargs"]["heads_cfg"]["pretrain_dec_path"])
        self.assertFalse(configured["planner"]["decode_each_iteration"])
        self.assertEqual(configured["meta"]["eval_episodes"], 2)
        self.assertEqual(configured["tasks_per_node"], 1)
        self.assertEqual(configured["folder"], "/data/runs/test")
        with self.assertRaises(ValueError):
            plan.prepare_config(original, Path("/data/runs/test"), 0)


if __name__ == "__main__":
    unittest.main()

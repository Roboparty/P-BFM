from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pbfm.utils.resource_paths import CONFIG_ROOT, PACKAGE_ROOT, PROJECT_ROOT, remap_resource_path, resolve_resource_path


class ResourcePathsTest(unittest.TestCase):
    def test_package_aliases_and_stale_code_resources(self):
        xml_suffix = "data/robots/g1_mjlab/g1_29dof.xml"
        expected_xml = PACKAGE_ROOT / xml_suffix
        self.assertTrue(expected_xml.is_file())
        for package in ("humanoidverse", "pbfm"):
            for prefix in (package, f"/nonexistent/pbfm-resource-test/code/{package}"):
                with self.subTest(prefix=prefix):
                    self.assertEqual(remap_resource_path(f"{prefix}/{xml_suffix}"), expected_xml)
                    self.assertEqual(remap_resource_path(f"{prefix}/config/base.yaml"), CONFIG_ROOT / "simulation/base.yaml")
        for root in ("code", "UFO", "UFO-g1depth", "PBFM"):
            self.assertEqual(
                remap_resource_path(f"/nonexistent/pbfm-resource-test/{root}/configs/robots/g1_29dof.yaml"),
                CONFIG_ROOT / "robots/g1_29dof.yaml",
            )

    def test_existing_absolute_and_external_paths_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            old_resource = Path(directory) / "code/humanoidverse/data/custom.pkl"
            old_resource.parent.mkdir(parents=True)
            old_resource.touch()
            self.assertEqual(remap_resource_path(old_resource), old_resource)
            self.assertEqual(resolve_resource_path(old_resource), old_resource)
        for external in (
            "/data/xue/bfmzero/data/lafan_29dof.pkl",
            "/nonexistent/dataset/configs/robots/g1_29dof.yaml",
            "/nonexistent/code/data/lafan.pkl",
            "/nonexistent/code/humanoidverse/data/../../external.pkl",
        ):
            self.assertEqual(remap_resource_path(external), Path(external))

    def test_config_relative_lookup_precedes_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "manifest"
            base.mkdir()
            (base / "motion.pkl").touch()
            (root / "motion.pkl").touch()
            with patch("pathlib.Path.cwd", return_value=root):
                self.assertEqual(resolve_resource_path("motion.pkl", base_dir=base), base / "motion.pkl")
                self.assertEqual(resolve_resource_path("missing.pkl", base_dir=base), root / "missing.pkl")
                self.assertEqual(resolve_resource_path("configs/robots/g1_29dof.yaml"), CONFIG_ROOT / "robots/g1_29dof.yaml")

    def test_motion_globs_and_environment_expansion(self):
        from pbfm.motion.data.paths import expand_motion_paths

        expected = PACKAGE_ROOT / "data/examples/g1_robot_state_sample.csv"
        with patch.dict(os.environ, {"PBFM_TEST_RESOURCE_PREFIX": "humanoidverse/data/examples"}):
            self.assertEqual(expand_motion_paths("$PBFM_TEST_RESOURCE_PREFIX/*.csv", suffix=".csv"), [expected])
        self.assertEqual(
            expand_motion_paths("/nonexistent/pbfm-resource-test/code/humanoidverse/data/examples/*.csv", suffix=".csv"),
            [expected],
        )
        with tempfile.TemporaryDirectory() as directory:
            external = Path(directory) / "code/humanoidverse/data/examples"
            external.mkdir(parents=True)
            external_file = external / "recorded.csv"
            external_file.touch()
            self.assertEqual(expand_motion_paths(str(external / "*.csv"), suffix=".csv"), [external_file])

    def test_original_robot_yaml_loads_outside_project_without_edits(self):
        from pbfm.robots import load_robot_spec, load_robot_training_spec
        from pbfm.motion.data.manifest import _load_manifest, _resolve_existing_path

        robot_yaml = CONFIG_ROOT / "robots/g1_29dof.yaml"
        original = robot_yaml.read_bytes()
        self.assertIn(b"xml_path: humanoidverse/data/robots/g1_mjlab/g1_29dof.xml", original)
        stale_yaml = "/nonexistent/pbfm-resource-test/code/configs/robots/g1_29dof.yaml"
        with tempfile.TemporaryDirectory() as directory, patch("pathlib.Path.cwd", return_value=Path(directory)):
            spec = load_robot_spec(stale_yaml)
            training = load_robot_training_spec("configs/robots/g1_29dof.yaml")
            self.assertEqual(spec.xml_file, PACKAGE_ROOT / "data/robots/g1_mjlab/g1_29dof.xml")
            self.assertEqual(training.robot.xml_file, spec.xml_file)
            self.assertEqual(len(spec.control_joint_names), 29)
            self.assertEqual(_resolve_existing_path(stale_yaml, Path(directory)), robot_yaml)
            fixture = PROJECT_ROOT / "tests/fixtures/example_mix.yaml"
            manifest, _, _ = _load_manifest(str(fixture))
            self.assertEqual(manifest, fixture)
        self.assertEqual(robot_yaml.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()

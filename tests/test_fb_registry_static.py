"""Registry checks that run without simulation or PyTorch dependencies."""
import ast
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FBRegistryTest(unittest.TestCase):
    def test_supported_presets_and_defaults(self):
        tree = ast.parse((ROOT / "pbfm/train.py").read_text())
        names = {"DEFAULT_AGENT", "DEFAULT_FB_UPDATE_Z_EVERY_STEP"}
        functions = {"canonical_agent_name", "_default_update_z_every_step"}
        nodes = [
            node for node in tree.body
            if (isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in names for t in node.targets
            )) or (isinstance(node, ast.FunctionDef) and node.name in functions)
        ]
        namespace = {"_FINAL_DEFAULTS": json.loads((ROOT / "pbfm/configs/legacy_large.json").read_text())}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "registry", "exec"), namespace)
        self.assertEqual(namespace["DEFAULT_AGENT"], "fb_depth")
        self.assertEqual(namespace["canonical_agent_name"]("fb_depth"), "fb_depth")
        self.assertEqual(namespace["_default_update_z_every_step"]("fb_depth"), 100)
        for name in ("fb", "fb_terrain", "unsupported"):
            with self.subTest(agent=name), self.assertRaises(ValueError):
                namespace["canonical_agent_name"](name)
            with self.subTest(agent=name), self.assertRaises(ValueError):
                namespace["_default_update_z_every_step"](name)

    def test_training_union_is_fb_only(self):
        tree = ast.parse((ROOT / "pbfm/training/config.py").read_text())
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "Agent" for t in n.targets))
        self.assertEqual({n.id for n in ast.walk(assignment.value) if isinstance(n, ast.Name)},
                         {"FBcprAuxAgentConfig"})

    def test_all_python_sources_parse(self):
        for folder in ("pbfm", "tests", "scripts"):
            for path in (ROOT / folder).rglob("*.py"):
                with self.subTest(path=str(path)):
                    ast.parse(path.read_text(), filename=str(path))


if __name__ == "__main__":
    unittest.main()

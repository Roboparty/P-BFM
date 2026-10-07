"""Export archived UFO robot and native RP1 terrain into browser-readable assets.

Run with a Python environment containing mujoco, mjlab and numpy:
  PYTHONPATH=<UFO checkout> python build_assets.py <UFO checkout> <browser config.json>
No simulation environment, CUDA workload, training or network call is created.
"""
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from mjlab.terrains import TerrainGenerator
from humanoidverse.terrains import rp1_simple as rp1


def build(source: Path, config_file: Path):
    out = Path(__file__).resolve().parent
    config = json.loads(config_file.read_text())
    robot_source = source / "humanoidverse/data/robots/g1_mjlab"
    shutil.copyfile(robot_source / "g1_29dof.xml", out / "g1_29dof.original.xml")
    shutil.copytree(robot_source / "meshes", out / "meshes", dirs_exist_ok=True)
    shutil.copyfile(source / "LICENSE", out / "LICENSE.source.txt")

    robot = ET.parse(robot_source / "g1_29dof.xml").getroot()
    robot.find("compiler").attrib.pop("meshdir")
    for mesh in robot.findall("asset/mesh"):
        mesh.set("file", "meshes/" + mesh.attrib["file"])
    for i, name in enumerate(config["jointNames"]):
        joint = robot.find(f".//joint[@name='{name}']")
        joint.set("armature", str(config["armature"][i]))
        joint.set("frictionloss", str(config["frictionLoss"][i]))
        motor = robot.find(f"actuator/motor[@joint='{name}']")
        limit = config["effortLimit"][i]
        motor.set("ctrllimited", "true")
        motor.set("ctrlrange", f"{-limit} {limit}")
    ET.indent(robot)
    ET.ElementTree(robot).write(out / "g1_29dof.xml", encoding="unicode")

    # Preserve the source generator's explicit seed=0. Environment seed=4831
    # controls environment sampling and does not override this map seed.
    generator = TerrainGenerator(rp1.make_rp1_simple_generator_cfg())
    spec = mujoco.MjSpec()
    generator.compile(spec)
    guard_tiles, _ = rp1.add_rp1_nonflat_guard_ring(spec, generator.cfg)
    rp1.add_rp1_outer_walls(spec, generator.cfg)
    # Buffer textures cannot be serialized by MjSpec. They affect appearance,
    # not collision/sensing geometry; use the original neutral gray instead.
    for geom in spec.geoms:
        geom.material = ""
    for material in list(spec.materials):
        spec.delete(material)
    for texture in list(spec.textures):
        spec.delete(texture)

    terrain = ET.fromstring(spec.to_xml())
    hfield_dir = out / "hfields"
    hfield_dir.mkdir(exist_ok=True)
    for i, (node, hfield) in enumerate(zip(terrain.findall("asset/hfield"), spec.hfields)):
        old_name = node.attrib["name"]
        name = f"terrain_hfield_{i:03d}"
        # Binary MuJoCo hfield: int32 rows, int32 cols, float32 row-major data.
        # MjSpec.userdata is in physical row order; XML elevation reverses rows.
        payload = struct.pack("<ii", hfield.nrow, hfield.ncol)
        payload += np.asarray(hfield.userdata, dtype="<f4").tobytes()
        (hfield_dir / f"{name}.bin").write_bytes(payload)
        node.attrib.clear()
        node.attrib.update(name=name, size=" ".join(map(str, hfield.size)), file=f"hfields/{name}.bin")
        for geom in terrain.findall(f".//geom[@hfield='{old_name}']"):
            geom.set("hfield", name)

    scene = ET.Element("mujoco", model="PBFM_original364M_native_rp1_map")
    ET.SubElement(scene, "include", file="g1_29dof.xml")
    ET.SubElement(scene, "option", timestep="0.005", integrator="implicitfast", solver="Newton", iterations="100", ls_iterations="50", tolerance="1e-8", cone="pyramidal", gravity="0 0 -9.81")
    # Large connected native map, not independent per-family arenas.
    scene.append(terrain.find("asset"))
    scene.append(terrain.find("worldbody"))
    ET.indent(scene)
    ET.ElementTree(scene).write(out / "scene.xml", encoding="unicode")

    families = list(generator.cfg.sub_terrains)
    spawn_regions = [dict(family=family, row=row, column=col,
                          origin=generator.terrain_origins[row, col].tolist(),
                          centerResetProfile=rp1.rp1_center_reset_profile(family)[0])
                     for row in range(generator.cfg.num_rows)
                     for col, family in enumerate(families)]
    metadata = {
        "families": families, "rows": 10, "columns": 7, "tileSize": 5,
        "coreSize": [50, 35], "outerSize": [74, 59], "generatorSeed": generator.cfg.seed,
        "environmentSeed": 4831, "guardTiles": guard_tiles,
        "defaultSpawn": {"family": "flat", "row": 4}, "spawnRegions": spawn_regions,
        "source": "Archived UFO rp1_simple.make_rp1_simple_generator_cfg + native mjlab TerrainGenerator",
        "generatorSourceSha256": hashlib.sha256((source / "humanoidverse/terrains/rp1_simple.py").read_bytes()).hexdigest(),
        "robotSourceSha256": hashlib.sha256((robot_source / "g1_29dof.xml").read_bytes()).hexdigest(),
        "robotProvenance": "Snapshot source path is absent locally; identical archived project XML verified across four related checkouts. Inertias and collision geometry retained; armature/friction and actuator limits from original364M render.log/config.",
        "dynamics": "Native MuJoCo CPU/WASM adaptation; explicit PD with DC motor torque-speed clipping, nominal gains, no training domain randomization. Not asserted equivalent to MuJoCo-Warp.",
        "terrainProvenance": "Original archived seven-family curriculum generator with default 0.10–0.20m stairs; not the historical fixed12cm video variant or a benchmark result.",
        "nativeMujocoVersion": mujoco.__version__,
    }
    (out / "terrain.json").write_text(json.dumps(metadata, indent=2) + "\n")
    files = ["scene.xml", "g1_29dof.xml"] + sorted(str(p.relative_to(out)) for folder in (out / "meshes", hfield_dir) for p in folder.iterdir() if p.is_file())
    (out / "files.json").write_text(json.dumps(files, indent=2) + "\n")

    # Validate compilation and every joint mapping against the actual checkpoint.
    model = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
    assert (model.nq, model.nv, model.nu) == (36, 35, 29)
    assert model.opt.timestep == config["physicsDt"]
    for i, name in enumerate(config["jointNames"]):
        joint = model.joint(name)
        np.testing.assert_allclose(model.dof_armature[joint.dofadr], config["armature"][i])
        np.testing.assert_allclose(model.dof_frictionloss[joint.dofadr], config["frictionLoss"][i])
    # Confirm binary hfield serialization preserves native generator surfaces.
    reference = spec.compile()
    np.testing.assert_allclose(model.hfield_data, reference.hfield_data, atol=1e-6)
    print(json.dumps({"nq": model.nq, "nv": model.nv, "nu": model.nu, "ngeom": model.ngeom, "nhfield": model.nhfield, "mapSize": metadata["coreSize"], "files": len(files)}))


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))

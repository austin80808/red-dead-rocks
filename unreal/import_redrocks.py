"""Import the Red Rocks package into the Unreal project and place it in a level.

Run headless:
    UnrealEditor-Cmd RedRocks.uproject -run=pythonscript \
        -script=unreal/import_redrocks.py -unattended -nosplash -stdout

Unreal's headless OBJ import silently decimates meshes above roughly 250
triangles (a 1,682-triangle flat plane comes back as 253), so the assets this
script produces are only usable for layout verification. Open the project in the
editor GUI and run the same import from the Content Browser for full-fidelity
geometry.

The terrain mesh is also tagged as non-Nanite: it is a single 282k-triangle
surface and Nanite buys nothing while costing build time and memory.
"""

import json
import os

import unreal

OUT_DIR = "/Users/austin/dev/redrocks-ue/out"
ASSET_DIR = "/Game/RedRocks"
LEVEL_PATH = "/Game/RedRocks/Lvl_RedRocks"

log = []


def _ensure_dir(path):
    if not unreal.EditorAssetLibrary.does_directory_exist(path):
        unreal.EditorAssetLibrary.make_directory(path)


def _import(name):
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", os.path.join(OUT_DIR, name + ".obj"))
    task.set_editor_property("destination_path", ASSET_DIR)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", True)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    return list(task.get_editor_property("imported_object_paths"))


def _disable_nanite(asset_path):
    mesh = unreal.EditorAssetLibrary.load_asset(asset_path)
    if not isinstance(mesh, unreal.StaticMesh):
        return
    try:
        settings = mesh.get_editor_property("nanite_settings")
        settings.set_editor_property("enabled", False)
        mesh.set_editor_property("nanite_settings", settings)
        unreal.EditorAssetLibrary.save_asset(asset_path)
    except Exception as exc:
        log.append("nanite toggle failed for %s: %r" % (asset_path, exc))


def main():
    with open(os.path.join(OUT_DIR, "manifest.json")) as fh:
        manifest = json.load(fh)

    _ensure_dir(ASSET_DIR)

    for name in ("RedRocks_Terrain", "Creation_Rock", "Ship_Rock"):
        paths = _import(name)
        log.append("imported %s -> %s" % (name, paths))
        for path in paths:
            _disable_nanite(path)

    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if unreal.EditorAssetLibrary.does_asset_exist(LEVEL_PATH):
        # delete_asset cannot remove the loaded persistent level, so reopen it
        # and clear its actors to keep repeated runs idempotent.
        level_subsystem.load_level(LEVEL_PATH)
        for actor in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
            if isinstance(actor, unreal.WorldSettings) or isinstance(
                actor, unreal.Brush
            ):
                continue
            unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(actor)
    else:
        level_subsystem.new_level(LEVEL_PATH)

    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

    def _place(asset_name, location):
        asset_path = "%s/%s" % (ASSET_DIR, asset_name)
        mesh = unreal.EditorAssetLibrary.load_asset(asset_path)
        if mesh is None:
            log.append("MISSING %s" % asset_path)
            return

        # spawn_actor_from_object silently no-ops in a headless commandlet
        # ("No actor was spawned"), so spawn a StaticMeshActor and assign the
        # mesh to its component instead.
        actor = actor_subsystem.spawn_actor_from_class(
            unreal.StaticMeshActor, location
        )
        if actor is None:
            log.append("SPAWN FAILED for %s" % asset_name)
            return

        actor.static_mesh_component.set_static_mesh(mesh)
        actor.set_actor_label(asset_name)
        actor.set_mobility(unreal.ComponentMobility.STATIC)
        log.append("placed %s at %s" % (asset_name, location))

    _place("RedRocks_Terrain", unreal.Vector(0.0, 0.0, 0.0))

    # The terrain mesh is authored with its lowest point at Z=0, so the bowl
    # floor sits at the height offset from that minimum.
    rock_z_cm = (
        manifest["monoliths"][0]["base_elevation_m"]
        - manifest["landscape"]["min_elevation_m"]
    ) * 100.0
    for rock in manifest["monoliths"]:
        label = rock["name"].split(" (")[0].replace(" ", "_")
        _place(
            label,
            unreal.Vector(
                rock["centroid_east"] * 100.0,
                rock["centroid_north"] * 100.0,
                rock_z_cm,
            ),
        )

    level_subsystem.save_current_level()
    log.append("saved %s" % LEVEL_PATH)


try:
    main()
except Exception as exc:
    log.append("ERROR: %r" % (exc,))

with open("/tmp/redrocks_import.log", "w") as fh:
    fh.write("\n".join(log))

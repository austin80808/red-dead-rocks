"""Issue #24: remove leftover import-experiment folders and the template level.

Runs INSIDE the Unreal editor so the asset registry fixes up references. Run it
from the project's own folder:

    UnrealEditor RedRocks.uproject -nosplash \
        -ExecCmds="py <repo>/tools/cleanup_project.py"

It is a dry run unless BOTH env vars are set:

    RR24_DELETE=1 RR24_DRY_OK=1

The dry run lists every asset in scope with its class and its referencers, and
deletes nothing.
"""

import os

import unreal

DELETE = os.environ.get("RR24_DELETE") == "1"

DIRS = [
    "/Game/Probe4",
    "/Game/Probe5",
    "/Game/Probe6",
    "/Game/Probe7",
    "/Game/RedRocksGrid",
    "/Game/RedRocksProbe",
]

# The template's grey test level. Its Blueprints/ Anims/ and Input/ stay.
LEVELS = [
    "/Game/FirstPerson/Lvl_FirstPerson",
    "/Game/FirstPerson/MI_FirstPersonColorway",
]

# World Partition spills the level's actors here; it goes with the level.
POST_DIRS = [
    "/Game/__ExternalActors__/FirstPerson/Lvl_FirstPerson",
]

LOG = []


def say(msg):
    LOG.append(msg)
    unreal.log("[rr24] %s" % msg)


def assets_under(path):
    lib = unreal.EditorAssetLibrary
    out = []
    try:
        out = list(lib.list_assets(path, recursive=True, include_folder=False))
    except Exception as exc:  # noqa: BLE001 - report, never abort the sweep
        say("list_assets failed for %s: %s" % (path, exc))
    return out


def referencers(asset_path):
    try:
        return list(
            unreal.EditorAssetLibrary.find_package_referencers_for_asset(
                asset_path, load_assets_to_confirm=False
            )
        )
    except Exception as exc:  # noqa: BLE001
        return ["<could not check: %s>" % exc]


def main():
    lib = unreal.EditorAssetLibrary
    say("mode: %s" % ("DELETE" if DELETE else "DRY RUN"))

    doomed = []

    say("--- leftover folders")
    for d in DIRS:
        exists = lib.does_directory_exist(d)
        found = assets_under(d) if exists else []
        say("%-22s exists=%s assets=%d" % (d, exists, len(found)))
        for a in found:
            refs = referencers(a)
            info = lib.find_asset_data(a)
            cls = str(info.asset_class_path.asset_name) if info.is_valid() else "?"
            say("    %-52s %-18s refs=%d %s" % (a, cls, len(refs), refs[:5]))
            doomed.append(a)

    say("--- template level")
    for a in LEVELS:
        exists = lib.does_asset_exist(a)
        refs = referencers(a) if exists else []
        say("%-46s exists=%s refs=%d %s" % (a, exists, len(refs), refs[:8]))
        if exists:
            doomed.append(a)

    say("--- world partition externals")
    for d in POST_DIRS:
        found = assets_under(d) if lib.does_directory_exist(d) else []
        say("%-58s exists=%s assets=%d" % (d, lib.does_directory_exist(d), len(found)))
        doomed.extend(found)

    say("--- summary: %d assets in scope" % len(doomed))

    if not DELETE:
        say("dry run: nothing deleted")
        unreal.SystemLibrary.quit_editor()
        return

    if os.environ.get("RR24_DRY_OK") != "1":
        say("refusing to delete: set RR24_DRY_OK=1 after reviewing the dry run")
        return

    failed = []
    # Levels go first: they hold the external-actor references.
    for a in doomed:
        if not lib.delete_asset(a):
            failed.append(a)
    for d in DIRS + POST_DIRS:
        if lib.does_directory_exist(d):
            try:
                lib.delete_directory(d)
            except Exception as exc:  # noqa: BLE001
                say("delete_directory %s: %s" % (d, exc))

    say("deleted %d assets, %d failed" % (len(doomed) - len(failed), len(failed)))
    for f in failed:
        say("    FAILED %s" % f)

    say("--- after")
    for d in DIRS:
        say("%-22s exists=%s" % (d, lib.does_directory_exist(d)))
    for a in LEVELS:
        say("%-46s exists=%s" % (a, lib.does_asset_exist(a)))

    try:
        unreal.EditorAssetLibrary.save_directory("/Game/RedRocks", only_if_is_dirty=False)
        unreal.EditorAssetLibrary.save_directory("/Game/FirstPerson", only_if_is_dirty=False)
        say("saved /Game/RedRocks and /Game/FirstPerson")
    except Exception as exc:  # noqa: BLE001
        say("save_directory: %s" % exc)

    if failed:
        say("RESULT: PARTIAL - %d assets still there" % len(failed))
    else:
        say("RESULT: OK")

    unreal.SystemLibrary.quit_editor()


main()

"""Press Play on Lvl_RedRocks and walk the player up the bowl, then quit.

This is the real test that the level is explorable: it needs the editor GUI
(PIE does not run in a commandlet), so launch the editor with it. Use
-ExecCmds, not -ExecutePythonScript: the latter quits the editor as soon as the
script returns, before a single frame of play.

    "/Users/Shared/Epic Games/UE_5.8/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor" \
        RedRocks.uproject -ExecCmds="py /Users/austin/dev/redrocks-ue/unreal/playtest.py"

The editor opens, starts Play In Editor, and on each frame:
  1. waits for the player pawn, then lets it fall and settle for 4 s;
  2. traces down from the pawn to see what it is standing on;
  3. takes a screenshot of the spawn view;
  4. walks forward (up the bowl) for 12 s, jumping every 0.8 s to clear the
     bench rows, sampling the position every second;
  5. takes a second screenshot, then one looking each way (E, N, W, S);
  6. ends Play and quits the editor.

Results go to /tmp/redrocks_playtest.log ("ok"/"FAIL" lines plus samples);
screenshots land in <project>/Saved/Screenshots/. Checks:
  - the pawn spawned and stopped falling (it did not drop through the world)
  - it is standing on the Stage
  - walking moved it at least 15 m horizontally and it climbed at least 2 m
"""

import math
import time

import unreal

LOG_PATH = "/tmp/redrocks_playtest.log"
SETTLE_S = 10.0  # auto exposure and TSR need ~10 s after the spawn cut before a shot is representative
WALK_S = 12.0
JUMP_EVERY_S = 0.8
TIMEOUT_S = 240.0
LOOK_YAWS = [0.0, 90.0, 180.0, 270.0]  # east, north, west, south

log = []
state = {"phase": "start", "t0": time.time(), "mark": None, "samples": [], "last_jump": 0.0}
handle = None


def _finish():
    global handle
    with open(LOG_PATH, "w") as fh:
        fh.write("\n".join(log) + "\n")
    for line in log:
        (unreal.log_error if line.startswith(("FAIL", "ERROR")) else unreal.log)(line)
    if handle is not None:
        unreal.unregister_slate_post_tick_callback(handle)
        handle = None
    try:
        unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).editor_request_end_play()
    except Exception:
        pass
    unreal.SystemLibrary.quit_editor()


def check(label, ok, detail=""):
    log.append("%s  %s%s" % ("ok   " if ok else "FAIL ", label, (": " + detail) if detail else ""))


def _game_world():
    try:
        return unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()
    except Exception:
        return None


def _below(world, pawn):
    loc = pawn.get_actor_location()
    hit = unreal.SystemLibrary.line_trace_single(
        world, loc, unreal.Vector(loc.x, loc.y, loc.z - 1000.0),
        unreal.TraceTypeQuery.TRACE_TYPE_QUERY1, True, [pawn],
        unreal.DrawDebugTrace.NONE, True,
    )
    t = hit.to_tuple() if hit is not None else ()
    if not t or not t[0]:
        return None, None
    actor = next((v for v in t if isinstance(v, unreal.Actor)), None)
    return (actor.get_actor_label() if actor else "?"), loc.z - t[4].z


def _shot(world, name):
    unreal.SystemLibrary.execute_console_command(world, "HighResShot 1600x900 filename=%s" % name)
    log.append("      screenshot %s" % name)


def tick(delta):
    now = time.time()
    try:
        if now - state["t0"] > TIMEOUT_S:
            log.append("ERROR: timed out in phase %s" % state["phase"])
            _finish()
            return

        phase = state["phase"]
        if phase == "start":
            unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).editor_request_begin_play()
            state["phase"] = "wait_pawn"
            return

        world = _game_world()
        pawn = unreal.GameplayStatics.get_player_pawn(world, 0) if world else None
        if phase == "wait_pawn":
            if pawn is not None:
                spawn = pawn.get_actor_location()
                log.append("      pawn %s at x=%.0f y=%.0f z=%.0f" % (pawn.get_class().get_name(), spawn.x, spawn.y, spawn.z))
                state.update(phase="settle", mark=now, spawn_z=spawn.z)
            return
        if pawn is None:
            log.append("ERROR: pawn vanished in phase %s" % phase)
            _finish()
            return

        loc = pawn.get_actor_location()
        if phase == "settle":
            if now - state["mark"] >= SETTLE_S:
                vz = pawn.get_velocity().z
                check("pawn stopped falling", abs(vz) < 5.0 and loc.z > -10000.0,
                      "z=%.0f vz=%.1f (spawned at z=%.0f)" % (loc.z, vz, state["spawn_z"]))
                label, gap = _below(world, pawn)
                check("standing on the stage", label == "Stage",
                      "on %s, %.0f cm below the pawn centre" % (label, gap or -1))
                _shot(world, "redrocks_play_spawn")
                state.update(phase="walk", mark=now, start=(loc.x, loc.y, loc.z), last_jump=now)
            return

        if phase == "walk":
            yaw = math.radians(pawn.get_actor_rotation().yaw)
            pawn.add_movement_input(unreal.Vector(math.cos(yaw), math.sin(yaw), 0.0), 1.0, True)
            if now - state["last_jump"] >= JUMP_EVERY_S:
                pawn.jump()
                state["last_jump"] = now
            if not state["samples"] or now - state["samples"][-1][0] >= 1.0:
                state["samples"].append((now, loc.x, loc.y, loc.z))
            if now - state["mark"] >= WALK_S:
                sx, sy, sz = state["start"]
                moved = math.hypot(loc.x - sx, loc.y - sy) / 100.0
                climbed = (loc.z - sz) / 100.0
                for t, x, y, z in state["samples"]:
                    log.append("      t=%4.1fs x=%.0f y=%.0f z=%.0f" % (t - state["mark"], x, y, z))
                check("walked up the bowl", moved >= 15.0 and climbed >= 2.0,
                      "moved %.1f m, climbed %.1f m" % (moved, climbed))
                label, _ = _below(world, pawn)
                log.append("      now standing on %s" % label)
                _shot(world, "redrocks_play_walked")
                state.update(phase="look", mark=now, look=0)
            return

        # Look around from where the walk ended: four compass views, one shot
        # each, 1.5 s apart so each screenshot finishes before the turn.
        if phase == "look" and now - state["mark"] >= 1.5:
            if state["look"] >= len(LOOK_YAWS):
                _finish()
                return
            yaw_deg = LOOK_YAWS[state["look"]]
            pawn.get_controller().set_control_rotation(unreal.Rotator(pitch=-5.0, yaw=yaw_deg, roll=0.0))
            state["pending_shot"] = "redrocks_play_look_%03d" % int(yaw_deg % 360)
            state.update(phase="look_shot", mark=now)
            return
        if phase == "look_shot" and now - state["mark"] >= 0.5:
            _shot(world, state["pending_shot"])
            state.update(phase="look", mark=now, look=state["look"] + 1)
    except Exception as exc:
        log.append("ERROR: %r" % (exc,))
        _finish()


handle = unreal.register_slate_post_tick_callback(tick)

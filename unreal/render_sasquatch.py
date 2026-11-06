"""Photograph and play-test the sasquatch in Lvl_RedRocks (Play In Editor).

Needs the editor GUI (grooms only render in a live viewport):
    UnrealEditor RedRocks.uproject -ExecCmds="py <repo>/unreal/render_sasquatch.py"

What it does, logged to $REDROCKS_LOG_DIR/redrocks_render_sasquatch.log:
  1. Plays, waits for the pawn to land and the shaders and exposure to settle,
     and checks he is the sasquatch (mesh, fur and ABP_Sasquatch loaded).
  2. Turns him to face the low morning sun and photographs him with RRPhoto
     (the orbit camera): front, three-quarter and side; then a close-up of
     the face.
  3. First person: looks down at his chest, arms and hands.
  4. Third person (V): walks, then sprints up the stage, a shot every 0.4 s
     (the frames for the walk and run GIFs), and checks he moved.
Screenshots: <project>/Saved/Screenshots/MacEditor/sasquatch_*.png. The
turnaround sheet is made from them by tools/make_sasquatch_sheet.py.
"""

import os
import time

import unreal

LOG_PATH = os.path.join(os.environ.get("REDROCKS_LOG_DIR", "/tmp"), "redrocks_render_sasquatch.log")
SETTLE_S = 25.0
TIMEOUT_S = 900.0
SUN_YAW = 75.0 - 90.0     # the sun's compass bearing (setup_lighting.py) as a level yaw: face into it
SHOTS = [                 # name, orbit yaw from his front, pitch, arm (cm)
    ("front", 0.0, -3.0, 430.0),
    ("three_quarter", 40.0, -4.0, 430.0),
    ("side", 90.0, -3.0, 430.0),
    ("face", 18.0, -2.0, 150.0),
]
FRAMES = 8
FRAME_DT = 0.4

log = []
state = {"phase": "start", "t0": time.time(), "mark": time.time(), "i": 0}
handle = None


def check(label, ok, detail=""):
    log.append("%s  %s%s" % ("ok   " if ok else "FAIL ", label, (": " + detail) if detail else ""))


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


def _world():
    try:
        return unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()
    except Exception:
        return None


def _shot(world, name, size="1600x900"):
    unreal.SystemLibrary.execute_console_command(world, "HighResShot %s filename=sasquatch_%s" % (size, name))
    log.append("      screenshot sasquatch_%s" % name)


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
            state.update(phase="wait_pawn")
            return
        world = _world()
        pawn = unreal.GameplayStatics.get_player_pawn(world, 0) if world else None
        if phase == "wait_pawn":
            if pawn is not None:
                unreal.SystemLibrary.execute_console_command(world, "DisableAllScreenMessages")
                pawn.toast("", 0.0)
                state.update(phase="settle", mark=now)
            return
        if pawn is None:
            log.append("ERROR: pawn vanished in phase %s" % phase)
            _finish()
            return
        dt = now - state["mark"]

        if phase == "settle" and dt >= SETTLE_S:
            check("he is the sasquatch", pawn.is_sasquatch())
            mesh = pawn.get_component_by_class(unreal.SkeletalMeshComponent)
            anim = mesh.get_anim_instance() if mesh else None
            check("ABP_Sasquatch drives him", anim is not None and "ABP_Sasquatch" in anim.get_class().get_name(),
                  anim.get_class().get_name() if anim else "no anim instance")
            grooms = pawn.get_components_by_class(unreal.GroomComponent)
            check("fur on", len(grooms) == 2 and all(g.get_editor_property("groom_asset") for g in grooms),
                  ", ".join(str(g.get_editor_property("groom_asset").get_name()) if g.get_editor_property("groom_asset")
                            else "none" for g in grooms))
            pawn.get_controller().set_control_rotation(unreal.Rotator(pitch=0.0, yaw=SUN_YAW, roll=0.0))
            state.update(phase="photo", mark=now, i=0)
            return

        if phase == "photo" and dt >= 1.0:
            if state["i"] >= len(SHOTS):
                pawn.rr_photo(0.0, 0.0, 0.0)
                pawn.get_controller().set_control_rotation(unreal.Rotator(pitch=-48.0, yaw=SUN_YAW, roll=0.0))
                state.update(phase="first_person", mark=now)
                return
            name, yaw, pitch, arm = SHOTS[state["i"]]
            pawn.rr_photo(yaw, pitch, arm)
            state.update(phase="photo_shot", mark=now, pending=name)
            return
        if phase == "photo_shot" and dt >= 2.0:
            _shot(world, state["pending"], "1200x1500" if state["pending"] != "face" else "1200x1200")
            state.update(phase="photo", mark=now, i=state["i"] + 1)
            return

        if phase == "first_person" and dt >= 2.5:
            check("first person by default", not pawn.is_third_person())
            _shot(world, "first_person_down")
            pawn.get_controller().set_control_rotation(unreal.Rotator(pitch=-15.0, yaw=SUN_YAW + 180.0, roll=0.0))
            pawn.script_move(unreal.Vector2D(0.0, 1.0), False)
            state.update(phase="first_person_walk", mark=now)
            return
        if phase == "first_person_walk" and dt >= 1.2:
            _shot(world, "first_person_walk")
            pawn.script_move(unreal.Vector2D(0.0, 0.0), False)
            pawn.script_press("Camera")
            state.update(phase="third_person", mark=now)
            return
        if phase == "third_person" and dt >= 2.0:
            check("V gives third person", pawn.is_third_person())
            _shot(world, "third_person_idle")
            loc = pawn.get_actor_location()
            state.update(phase="walk", mark=now, i=0, start=(loc.x, loc.y), last=now)
            pawn.script_move(unreal.Vector2D(0.0, 1.0), False)
            return
        if phase in ("walk", "run"):
            if now - state["last"] >= FRAME_DT:
                _shot(world, "%s_%02d" % (phase, state["i"]), "1280x720")
                state.update(i=state["i"] + 1, last=now)
            if state["i"] >= FRAMES:
                loc = pawn.get_actor_location()
                sx, sy = state["start"]
                moved = ((loc.x - sx) ** 2 + (loc.y - sy) ** 2) ** 0.5 / 100.0
                check("%s moved him" % phase, moved > 2.0, "%.1f m, %.1f m/s now" % (moved, pawn.get_speed_mps()))
                if phase == "walk":
                    pawn.script_move(unreal.Vector2D(0.0, 1.0), True)
                    state.update(phase="run", i=0, start=(loc.x, loc.y), last=now)
                else:
                    pawn.script_release()
                    state.update(phase="done", mark=now)
            return
        if phase == "done" and dt >= 1.5:
            _finish()
    except Exception as exc:
        log.append("ERROR: %r" % (exc,))
        _finish()


handle = unreal.register_slate_post_tick_callback(tick)

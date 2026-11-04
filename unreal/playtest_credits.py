"""Press Play, check the credits hint, press C to open and close the credits.

Like playtest.py this needs the editor GUI (Play In Editor does not run in a
commandlet), so launch the editor with -ExecCmds, not -ExecutePythonScript
(that one quits as soon as the script returns):

    "/Users/Shared/Epic Games/UE_5.8/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor" \
        RedRocks.uproject -nosplash \
        -ExecCmds="py /Users/austin/dev/redrocks-ue/unreal/playtest_credits.py"

On each frame (a Slate post-tick state machine, as in playtest.py) it:
  1. starts Play In Editor and waits for the pawn, then 3 s for it to settle;
  2. finds the CreditsMenu actor that setup_credits.py placed, its widget, and
     checks the widget is on screen showing page 0 (the "C  Credits" hint) and
     that IMC_Default maps the C key to IA_Credits;
  3. screenshots the hint;
  4. presses C the way a keyboard would reach the game: the Enhanced Input
     console command `Input.+key C` holds the key down through the player's
     mapping contexts, `Input.-key C` lets it go 0.25 s later;
  5. checks the widget switched to page 1 (the credits card), screenshots it;
  6. presses C again, checks it is back on page 0, screenshots, quits.

Results go to /tmp/redrocks_playtest_credits.log ("ok"/"FAIL" lines);
screenshots (with the UI, via `Shot showui`) land in
<project>/Saved/Screenshots/MacEditor/redrocks_credits_*.png.
"""

import os
import time
import traceback

import unreal

LOG_PATH = os.path.join(os.environ.get("REDROCKS_LOG_DIR", "/tmp"), "redrocks_playtest_credits.log")
MENU_BP = "/Game/RedRocks/UI/BP_CreditsMenu"
INPUT_ACTION = "/Game/RedRocks/UI/IA_Credits"
IMC_PATH = "/Game/Input/IMC_Default"
WIDGET_VAR = "CreditsWidget"
SWITCHER = "Switcher"
KEY = "C"
SETTLE_S = 3.0
HOLD_S = 0.25
AFTER_PRESS_S = 1.0
TIMEOUT_S = 240.0

log = []
state = {"phase": "start", "t0": time.time(), "mark": None}
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
    return ok


def _game_world():
    try:
        return unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()
    except Exception:
        return None


def _cmd(world, command):
    unreal.SystemLibrary.execute_console_command(world, command)


def _shot(world, name):
    # HighResShot renders the scene only; `Shot showui` keeps the UMG overlay.
    _cmd(world, "Shot showui filename=%s nosuffix" % name)
    log.append("      screenshot %s.png" % name)


def _widget(world):
    # EditorAssetLibrary.load_blueprint_class returns None during Play.
    cls = unreal.load_object(None, MENU_BP + "." + MENU_BP.rsplit("/", 1)[1] + "_C")
    actors = unreal.GameplayStatics.get_all_actors_of_class(world, cls) if cls else []
    if not actors:
        names = sorted({a.get_class().get_name() for a in
                        unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)})
        log.append("      no BP_CreditsMenu in %s; classes: %s" % (world.get_path_name(), names))
        return None, 0
    widget = None
    for name in (WIDGET_VAR, "credits_widget"):
        try:
            widget = actors[0].get_editor_property(name)
            break
        except Exception:
            continue
    return widget, len(actors)


def _page(widget):
    """Active page of the widget's Switcher (a Blueprint variable on WBP_Credits)."""
    if widget is None:
        return None
    switcher = None
    for name in (SWITCHER, SWITCHER.lower()):
        try:
            switcher = widget.get_editor_property(name)
            break
        except Exception:
            continue
    return switcher.get_active_widget_index() if switcher else None


def _mapped_keys():
    """Keys IMC_Default maps to IA_Credits (UE 5.8 keeps them under
    default_key_mappings; older versions under mappings)."""
    # EditorAssetLibrary refuses to work while Play is running; load directly.
    imc = unreal.load_asset(IMC_PATH)
    action = unreal.load_asset(INPUT_ACTION)
    rows = list(imc.get_editor_property("mappings") or [])
    try:
        rows += list(imc.get_editor_property("default_key_mappings").get_editor_property("mappings"))
    except Exception:
        pass
    keys = []
    for m in rows:
        mapped = m.get_editor_property("action")
        if mapped is not None and mapped.get_path_name() == action.get_path_name():
            # FKey's name isn't a script property; its text export is "C".
            keys.append(m.get_editor_property("key").export_text().strip('"()').replace("KeyName=", ""))
    return keys


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
                state.update(phase="settle", mark=now)
            return
        if pawn is None:
            log.append("ERROR: pawn vanished in phase %s" % phase)
            _finish()
            return

        elapsed = now - state["mark"]
        if phase == "settle" and elapsed >= SETTLE_S:
            widget, count = _widget(world)
            check("one CreditsMenu actor in play", count == 1, "%d found" % count)
            if not check("credits widget created", widget is not None):
                _finish()
                return
            state["widget"] = widget
            check("widget on screen", widget.is_in_viewport())
            check("hint showing at start (page 0)", _page(widget) == 0, "page %s" % _page(widget))
            keys = _mapped_keys()
            check("IMC_Default maps C to IA_Credits", KEY in keys, "keys %s" % keys)
            in_world = [a for a in unreal.GameplayStatics.get_all_actors_of_class(world, unreal.TextRenderActor)
                        if a.get_actor_label() == "Credits"]
            check("in-world credits still there", len(in_world) == 1, "%d found" % len(in_world))
            _shot(world, "redrocks_credits_hint")
            state.update(phase="press_open", mark=now)
            return

        # A key press: hold C for HOLD_S, release, give the UI a moment.
        if phase in ("press_open", "press_close") and elapsed >= 1.0:
            _cmd(world, "Input.+key %s" % KEY)
            state.update(phase=phase.replace("press", "release"), mark=now)
            return
        if phase in ("release_open", "release_close") and elapsed >= HOLD_S:
            _cmd(world, "Input.-key %s" % KEY)
            state.update(phase=phase.replace("release", "look"), mark=now)
            return

        if phase == "look_open" and elapsed >= AFTER_PRESS_S:
            page = _page(state["widget"])
            check("C opens the credits (page 1)", page == 1, "page %s" % page)
            _shot(world, "redrocks_credits_open")
            state.update(phase="press_close", mark=now)
            return
        if phase == "look_close" and elapsed >= AFTER_PRESS_S:
            page = _page(state["widget"])
            check("C again closes them (page 0)", page == 0, "page %s" % page)
            _shot(world, "redrocks_credits_closed")
            state.update(phase="done", mark=now)
            return
        if phase == "done" and elapsed >= 1.5:  # let the last screenshot write
            _finish()
    except Exception as exc:
        log.append("ERROR: %r" % (exc,))
        log.extend("      " + l for l in traceback.format_exc().splitlines()[-6:])
        _finish()


handle = unreal.register_slate_post_tick_callback(tick)

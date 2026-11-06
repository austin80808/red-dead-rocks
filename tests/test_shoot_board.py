"""Offline parts of unreal/shoot_board.py: shot sizes, cameras, the sun, the shot
plan, the log, and that importing it outside the editor does nothing."""

import importlib.util
import json
import math
import os
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

UNREAL = Path(__file__).resolve().parent.parent / "unreal"


class _Watched(types.ModuleType):
    """A stand-in `unreal` that records every attribute anything asks it for."""

    def __init__(self, **attrs):
        super().__init__("unreal")
        self.__dict__["asked"] = []
        self.__dict__.update(attrs)

    def __getattr__(self, name):  # only called for attributes it does not have
        self.__dict__["asked"].append(name)
        raise AttributeError(name)


def _load(name, stub=None):
    saved = sys.modules.get("unreal")
    sys.modules["unreal"] = stub if stub is not None else types.ModuleType("unreal")
    try:
        spec = importlib.util.spec_from_file_location(name + "_under_test", UNREAL / (name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if saved is None:
            del sys.modules["unreal"]
        else:
            sys.modules["unreal"] = saved


@pytest.fixture(scope="module")
def sb():
    return _load("shoot_board")


def _view(name="postcard", **over):
    v = {"name": name, "aspect": [4, 3],
         "camera": {"x": 1200.0, "y": -3400.0, "z": 2500.0, "pitch": -8.0, "yaw": -150.0, "roll": 1.5, "fov": 63.0},
         "sun": {"azimuth_deg": 130.0, "elevation_deg": 58.0},
         "landmarks": [{"name": "stage_rock_summit", "photo_xy": [600, 300]},
                       {"name": "booth_roof", "photo_xy": [700, 500]}]}
    v.update(over)
    return v


def test_importing_outside_the_editor_does_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("REDROCKS_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("REDROCKS_OUT_DIR", str(tmp_path))
    stub = _Watched()
    mod = _load("shoot_board", stub)
    assert mod.handle is None and mod.state == {"phase": "start"} and mod.log == []
    assert stub.asked == ["register_slate_post_tick_callback"]  # the guard's hasattr, nothing else
    assert list(tmp_path.iterdir()) == []


def test_importing_in_the_editor_only_registers_the_tick(tmp_path, monkeypatch):
    monkeypatch.setenv("REDROCKS_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("REDROCKS_OUT_DIR", str(tmp_path))
    registered = []
    stub = _Watched(register_slate_post_tick_callback=lambda fn: registered.append(fn) or "handle")
    mod = _load("shoot_board", stub)
    assert registered == [mod.tick] and mod.handle == "handle"
    assert stub.asked == []  # no editor call until the first tick
    assert mod.state == {"phase": "start"} and list(tmp_path.iterdir()) == []


def test_shot_size_is_1600_wide_at_the_views_aspect(sb):
    assert sb.shot_size([4, 3]) == (1600, 1200)
    assert sb.shot_size([16, 9]) == (1600, 900)
    assert sb.shot_size([896, 1138]) == (1600, 2032)  # the portrait aerial thumb
    assert sb.shot_size([4, 3], width=800) == (800, 600)
    for bad in ([4], [4, 0], None, ["a", 3]):
        with pytest.raises(ValueError):
            sb.shot_size(bad)


def test_camera_settings_are_the_views_level_frame_camera(sb):
    c = sb.camera_settings(_view())
    assert c["location"] == (1200.0, -3400.0, 2500.0)
    # Already Unreal level-frame angles: passed straight through, never negated.
    assert c["rotation"] == {"pitch": -8.0, "yaw": -150.0, "roll": 1.5}
    assert c["fov"] == 63.0
    assert c["aspect_ratio"] == 4 / 3
    no_roll = _view(camera={"x": 0, "y": 0, "z": 0, "pitch": 0, "yaw": 90, "fov": 50})
    assert sb.camera_settings(no_roll)["rotation"]["roll"] == 0.0
    with pytest.raises(ValueError, match="aerial: no fitted camera"):
        sb.camera_settings(_view("aerial", camera=None))


def test_the_aspect_ratio_is_the_shots_so_there_are_no_letterbox_bars(sb):
    c = sb.camera_settings(_view(aspect=[3, 2]))
    w, h = sb.shot_size([3, 2])
    assert c["aspect_ratio"] == w / h


@pytest.mark.parametrize("azimuth,elevation", [(130.0, 58.0), (75.0, 24.0), (250.0, 40.0), (0.0, 10.0)])
def test_the_sun_is_set_by_setup_lightings_own_maths(sb, azimuth, elevation):
    lighting = _load("setup_lighting")
    pitch, yaw = lighting.sun_rotation(azimuth, elevation)
    got = sb.sun_rotation({"azimuth_deg": azimuth, "elevation_deg": elevation})
    assert got == {"pitch": pitch, "yaw": yaw, "roll": 0.0}


def test_loading_the_sun_maths_never_runs_setup_lighting(tmp_path, monkeypatch):
    """Importing setup_lighting in the editor rebuilds and saves the light rig.
    shoot_board must load it under a bare stand-in, even inside the editor."""
    monkeypatch.setenv("REDROCKS_LOG_DIR", str(tmp_path))
    editor = _Watched(EditorAssetLibrary=object(), register_slate_post_tick_callback=lambda fn: "h")
    mod = _load("shoot_board", editor)
    saved = sys.modules.get("unreal")
    sys.modules["unreal"] = editor
    try:
        mod._LIGHTING = None
        mod.sun_rotation({"azimuth_deg": 130.0, "elevation_deg": 58.0})
        assert sys.modules["unreal"] is editor  # the real module is back afterwards
    finally:
        if saved is None:
            del sys.modules["unreal"]
        else:
            sys.modules["unreal"] = saved
    assert editor.asked == []  # setup_lighting never touched the editor API
    assert not (tmp_path / "redrocks_lighting.log").exists()


def _settings(sb, **env):
    base = {"REDROCKS_OUT_DIR": "/x/out", "REDROCKS_LOG_DIR": "/x/logs"}
    base.update(env)
    return sb.read_settings(base, [])


def test_settings_come_from_the_environment_and_views_from_the_command_line(sb):
    s = sb.read_settings({"REDROCKS_OUT_DIR": "/x/out", "REDROCKS_BOARD_SUN": "Both",
                          "REDROCKS_BOARD_ONLY": "postcard, aerial,", "REDROCKS_BOARD_CALIB": "1",
                          "REDROCKS_VIEWS": "/env/views.json"},
                         ["shoot_board.py", "--views", "/cli/views.json"])
    assert s["board_dir"] == "/x/out/board" and s["sun"] == "both" and s["calib"] is True
    assert s["only"] == ["postcard", "aerial"] and s["views"] == "/cli/views.json"
    assert s["level"] == "/Game/RedRocks/Lvl_RedRocks" and s["prefix"] == "before_"
    assert s["landmarks"] == "/x/out/board/landmarks_cm.json" and s["marker_cm"] == 60.0
    assert s["log"] == "/tmp/redrocks_board.log"
    d = sb.read_settings({"REDROCKS_LEVEL": "/Game/Other/Lvl", "REDROCKS_LOG_DIR": "/l",
                          "REDROCKS_BOARD_PREFIX": "after_", "REDROCKS_BOARD_CALIB": "0"}, [])
    assert d["sun"] == "view" and d["calib"] is False and d["level"] == "/Game/Other/Lvl"
    assert d["prefix"] == "after_" and d["log"] == "/l/redrocks_board.log"
    assert d["views"].endswith("docs/media/photo-board/views.json")
    with pytest.raises(ValueError, match="REDROCKS_BOARD_SUN"):
        sb.read_settings({"REDROCKS_BOARD_SUN": "noon"}, [])


def test_the_plan_shoots_each_view_under_its_own_sun(sb):
    shots, problems = sb.shot_plan([_view("postcard"), _view("aerial", aspect=[896, 1138])], _settings(sb))
    assert problems == []
    assert [s["file"] for s in shots] == ["before_postcard.png", "before_aerial.png"]
    assert [s["kind"] for s in shots] == ["view", "view"]
    assert shots[0]["sun"] == sb.sun_rotation(_view()["sun"])
    assert shots[1]["size"] == (1600, 2032) and shots[0]["settle_s"] >= 8.0
    assert shots[0]["camera"] == sb.camera_settings(_view())


def test_the_level_sun_keeps_the_levels_own_light(sb):
    shots, _ = sb.shot_plan([_view()], _settings(sb, REDROCKS_BOARD_SUN="level"))
    assert [(s["file"], s["kind"], s["sun"]) for s in shots] == [("before_postcard.png", "level", None)]


def test_both_suns_and_calibration_in_one_run(sb):
    marks = {"stage_rock_summit": (100.0, 200.0, 300.0), "booth_roof": (400.0, 500.0, 600.0)}
    shots, problems = sb.shot_plan([_view("postcard"), _view("aerial")],
                                   _settings(sb, REDROCKS_BOARD_SUN="both", REDROCKS_BOARD_CALIB="1"), marks)
    assert problems == []
    assert [s["file"] for s in shots] == [
        "before_postcard.png", "calib_postcard.png", "before_level_postcard.png",
        "before_aerial.png", "calib_aerial.png", "before_level_aerial.png"]
    view_sun, calib, level = shots[:3]
    assert calib["kind"] == "calib" and calib["sun"] == view_sun["sun"]  # no relight before the markers
    assert calib["markers"] == [("stage_rock_summit", (100.0, 200.0, 300.0)), ("booth_roof", (400.0, 500.0, 600.0))]
    assert level["kind"] == "level" and level["sun"] is None and level["settle_s"] >= 8.0


def test_the_plan_names_every_view_it_cannot_shoot(sb):
    views = [_view("postcard"), _view("aerial", camera=None), _view("plaza", sun=None)]
    shots, problems = sb.shot_plan(views, _settings(sb, REDROCKS_BOARD_ONLY="postcard,aerial,plaza,eiffel"))
    assert [s["file"] for s in shots] == ["before_postcard.png"]
    assert dict(problems)["aerial"].startswith("aerial: no fitted camera")
    assert "no sun" in dict(problems)["plaza"]
    assert dict(problems)["eiffel"] == "not on the board"
    only = sb.shot_plan(views, _settings(sb, REDROCKS_BOARD_ONLY="postcard"))
    assert [s["file"] for s in only[0]] == ["before_postcard.png"] and only[1] == []


def test_calibration_without_marker_positions_is_a_problem_not_a_guess(sb):
    s = _settings(sb, REDROCKS_BOARD_CALIB="1")
    shots, problems = sb.shot_plan([_view()], s, None)
    assert [x["file"] for x in shots] == ["before_postcard.png"]
    assert "landmarks_cm.json" in dict(problems)["postcard"]
    shots, problems = sb.shot_plan([_view()], s, {"stage_rock_summit": (1.0, 2.0, 3.0)})
    assert [x["file"] for x in shots] == ["before_postcard.png", "calib_postcard.png"]
    assert "booth_roof" in dict(problems)["postcard"]  # the stale file is named, not skipped silently


def test_marker_size_in_pixels(sb):
    cam = sb.camera_settings(_view(camera={"x": 0, "y": 0, "z": 0, "pitch": 0, "yaw": 0, "fov": 90}))
    assert sb.marker_size_px(cam, (10000.0, 0.0, 0.0), 1600, 60.0) == pytest.approx(4.8)  # 800 px focal
    assert sb.marker_size_px(cam, (-10000.0, 0.0, 0.0), 1600, 60.0) == 0.0  # behind
    turned = sb.camera_settings(_view(camera={"x": 0, "y": 0, "z": 0, "pitch": 0, "yaw": 90, "fov": 90}))
    assert sb.marker_size_px(turned, (0.0, 20000.0, 0.0), 1600, 60.0) == pytest.approx(2.4)


def test_a_png_counts_only_once_it_is_complete(sb, tmp_path):
    p = tmp_path / "shot.png"
    assert sb.png_complete(str(p)) is False
    Image.new("RGB", (32, 24), (120, 60, 30)).save(p)
    assert sb.png_complete(str(p)) is True
    assert sb.png_size(str(p)) == (32, 24)
    data = p.read_bytes()
    p.write_bytes(data[:-20])  # still being written
    assert sb.png_complete(str(p)) is False


def test_the_log_has_one_ok_or_fail_line_per_view(sb):
    results = {"postcard": {"files": ["before_postcard.png", "calib_postcard.png"], "errors": []},
               "aerial": {"files": [], "errors": ["aerial: no fitted camera"]},
               "plaza": {"files": ["before_plaza.png"], "errors": ["no before_level_plaza.png after 120 s"]}}
    lines = sb.summary_lines(["postcard", "aerial", "plaza", "stage_front"], results)
    assert [l.split()[0] for l in lines] == ["ok", "FAIL", "FAIL", "FAIL"]
    assert "before_postcard.png, calib_postcard.png" in lines[0]
    assert "not shot" in lines[3]


def test_views_and_marker_positions_load_from_json(sb, tmp_path):
    vf = tmp_path / "views.json"
    vf.write_text(json.dumps({"about": "x", "views": [_view()]}))
    assert sb.load_views(str(vf))[0]["name"] == "postcard"
    lf = tmp_path / "landmarks_cm.json"
    assert sb.load_landmarks(str(lf)) is None
    lf.write_text(json.dumps({"about": "x", "landmarks": {"booth_roof": [1, 2.5, 3]}}))
    assert sb.load_landmarks(str(lf)) == {"booth_roof": (1.0, 2.5, 3.0)}
    assert all(math.isfinite(c) for c in sb.load_landmarks(str(lf))["booth_roof"])


# ---- The whole run against a fake editor ---------------------------------------
# Nothing here can run Unreal, so a stand-in `unreal` with just the calls the
# script makes drives the real tick state machine on a fake clock: the order of
# shots, the sun, the hidden text, the markers and the clean-up at the end.

class _Clock:
    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now


class _Rot:
    def __init__(self, roll=0.0, pitch=0.0, yaw=0.0):
        self.roll, self.pitch, self.yaw = roll, pitch, yaw

    def key(self):
        return (round(self.pitch, 6), round(self.yaw, 6), round(self.roll, 6))


class _Vec:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class _Comp:
    def __init__(self):
        self.props = {}

    def set_editor_property(self, key, value):
        self.props[key] = value

    def set_mobility(self, value):
        self.props["mobility"] = value

    def set_static_mesh(self, mesh):
        self.props["static_mesh"] = mesh
        return True

    def set_material(self, index, mat):
        self.props["material"] = mat


class _Actor:
    clock = None  # set by _Editor.module: when an actor last moved or turned

    def __init__(self, label, comps=()):
        self.label, self.rotation, self.location, self.scale = label, _Rot(), None, None
        self.hidden_ed, self.transient, self.changed_at = False, None, 0.0
        self.comps = {c: _Comp() for c in comps}

    def _moved(self):
        self.changed_at = _Actor.clock.now

    def get_actor_label(self):
        return self.label

    def set_actor_label(self, label, mark_dirty=True):
        self.label = label

    def get_actor_rotation(self):
        return self.rotation

    def set_actor_rotation(self, rot, teleport):
        self.rotation = rot
        self._moved()
        return True

    def set_actor_location_and_rotation(self, loc, rot, sweep, teleport):
        self.location, self.rotation = loc, rot
        self._moved()

    def set_actor_location(self, loc, sweep, teleport):
        self.location = loc

    def set_actor_scale3d(self, scale):
        self.scale = scale

    def get_component_by_class(self, cls):
        return self.comps.get(cls)

    def is_temporarily_hidden_in_editor(self):
        return self.hidden_ed

    def set_is_temporarily_hidden_in_editor(self, hidden):
        self.hidden_ed = hidden


class _TextActor(_Actor):
    pass


class _Editor:
    """Records what the script does; writes each screenshot a few ticks late."""

    def __init__(self, sun=True, pie=False):
        self.actors = [_TextActor("Credits"), _TextActor("Scan_ml0w_Credit"), _Actor("Stage")]
        self.sign = _Actor("Sign", comps=("TextRenderComponent",))  # text on a non-text actor
        self.actors.append(self.sign)
        if sun:
            self.sun = _Actor("Sun")
            self.sun.rotation = _Rot(pitch=-58.0, yaw=40.0)
            self.actors.append(self.sun)
        self.pie, self.loaded, self.ended_pie = pie, [], False
        self.console, self.shots, self.pending, self.spawned, self.destroyed = [], [], [], [], []
        self.quit, self.unregistered, self.callbacks, self.piloted, self.game_view = 0, 0, [], [], False

    def module(self, clock):
        ed = self
        _Actor.clock = clock
        u = types.ModuleType("unreal")
        for name in ("LevelEditorSubsystem", "UnrealEditorSubsystem", "EditorActorSubsystem", "CameraActor",
                     "StaticMeshActor", "CameraComponent", "StaticMeshComponent", "TextRenderComponent"):
            setattr(u, name, name)
        u.TextRenderActor = _TextActor
        u.Vector, u.Rotator = _Vec, _Rot
        u.LinearColor = lambda r, g, b, a: (r, g, b, a)
        u.MaterialShadingModel = types.SimpleNamespace(MSM_UNLIT="unlit")
        u.MaterialProperty = types.SimpleNamespace(MP_EMISSIVE_COLOR="emissive")
        u.ComponentMobility = types.SimpleNamespace(MOVABLE="movable")
        u.Material = "Material"
        for name in ("MaterialExpressionConstant3Vector", "MaterialExpressionMultiply",
                     "MaterialExpressionEyeAdaptationInverse"):
            setattr(u, name, name)
        u.new_object = lambda cls, outer=None, name=None: _Comp()
        u.get_transient_package = lambda: "transient"
        u.load_object = lambda outer, path: "sphere:" + path
        u.MaterialEditingLibrary = types.SimpleNamespace(
            create_material_expression=lambda mat, cls, x, y: _Comp(),
            connect_material_expressions=lambda a, ao, b, bi: True,
            connect_material_property=lambda e, o, p: True,
            recompile_material=lambda mat: None)
        world = types.SimpleNamespace(get_path_name=lambda: "/Game/RedRocks/Lvl_RedRocks.Lvl_RedRocks")

        class LES:
            def load_level(self, path):
                assert not ed.pie, "load_level while Play In Editor runs"
                ed.loaded.append(path)
                return True

            def editor_request_end_play(self):
                ed.pie, ed.ended_pie = False, True

            def editor_get_game_view(self):
                return ed.game_view

            def editor_set_game_view(self, on):
                ed.game_view = on

            def editor_set_viewport_realtime(self, on):
                pass

            def get_exact_camera_view(self):
                return True

            def set_exact_camera_view(self, on):
                pass

            def pilot_level_actor(self, actor):
                ed.piloted.append(actor)

            def eject_pilot_level_actor(self):
                pass

        class UES:
            def get_editor_world(self):
                return world if ed.loaded else None

            def get_game_world(self):
                return "pie-world" if ed.pie else None

        class EAS:
            def get_all_level_actors(self):
                return list(ed.actors)

            def spawn_actor_from_class(self, cls, loc, rot, transient):
                comps = {"CameraActor": ("CameraComponent",), "StaticMeshActor": ("StaticMeshComponent",)}[cls]
                actor = _Actor(cls, comps)
                actor.location, actor.transient, actor.kind = loc, transient, cls
                ed.actors.append(actor)
                ed.spawned.append(actor)
                return actor

            def destroy_actor(self, actor):
                ed.actors.remove(actor)
                ed.destroyed.append(actor)
                return True

        subs = {"LevelEditorSubsystem": LES(), "UnrealEditorSubsystem": UES(), "EditorActorSubsystem": EAS()}
        u.get_editor_subsystem = lambda name: subs[name]

        def console(w, cmd):
            ed.console.append(cmd)

        def quit_editor():
            ed.quit += 1

        u.SystemLibrary = types.SimpleNamespace(execute_console_command=console,
                                                get_console_variable_int_value=lambda name: 4,
                                                quit_editor=quit_editor)

        def screenshot(w, h, path, camera):
            flags = [c for c in ed.console if c.startswith("ShowFlag.Tonemapper")]
            moved = max(camera.changed_at, ed.sun.changed_at if hasattr(ed, "sun") else 0.0)
            ed.shots.append({"file": os.path.basename(path), "since_move": clock.now - moved,
                             "camera": camera.rotation.key(),
                             "fov": camera.comps["CameraComponent"].props.get("field_of_view"),
                             "sun": ed.sun.rotation.key() if hasattr(ed, "sun") else None,
                             "text_hidden": all(a.hidden_ed for a in ed.actors if isinstance(a, _TextActor))
                             and ed.sign.hidden_ed,
                             "markers": [a for a in ed.actors if getattr(a, "kind", "") == "StaticMeshActor"],
                             "tonemapper": flags[-1] if flags else None, "size": (w, h)})
            ed.pending.append([3, path, (w, h)])
            return types.SimpleNamespace(is_task_done=lambda: True)

        u.AutomationLibrary = types.SimpleNamespace(take_high_res_screenshot=screenshot,
                                                    finish_loading_before_screenshot=lambda: None)
        u.log = lambda msg: None
        u.log_error = lambda msg: None

        def register(fn):
            ed.callbacks.append(fn)
            return "tick-handle"

        def unregister(h):
            ed.unregistered += 1

        u.register_slate_post_tick_callback = register
        u.unregister_slate_post_tick_callback = unregister
        return u

    def pump(self):
        for job in list(self.pending):
            job[0] -= 1
            if job[0] <= 0:
                Image.new("RGB", job[2], (150, 90, 60)).save(job[1])
                self.pending.remove(job)


def _run(tmp_path, monkeypatch, views, ed, **env):
    out = tmp_path / "out"
    (out / "board").mkdir(parents=True)
    vf = tmp_path / "views.json"
    vf.write_text(json.dumps({"views": views}))
    (out / "board" / "landmarks_cm.json").write_text(json.dumps({"landmarks": {
        "stage_rock_summit": [100.0, 200.0, 300.0], "booth_roof": [400.0, 500.0, 600.0]}}))
    monkeypatch.setenv("REDROCKS_OUT_DIR", str(out))
    monkeypatch.setenv("REDROCKS_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("REDROCKS_VIEWS", str(vf))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    clock = _Clock()
    mod = _load("shoot_board", ed.module(clock))
    monkeypatch.setattr(mod, "time", clock)
    for _ in range(20000):
        if ed.quit:
            break
        clock.now += 0.1  # a 10 fps editor
        ed.pump()
        mod.tick(0.1)
    return mod, out / "board", (tmp_path / "redrocks_board.log").read_text()


def test_a_whole_run_in_a_fake_editor(tmp_path, monkeypatch):
    ed = _Editor(pie=True)  # e.g. a script before this one left Play In Editor running
    mod, board, text = _run(tmp_path, monkeypatch, [_view("postcard")], ed,
                            REDROCKS_BOARD_SUN="both", REDROCKS_BOARD_CALIB="1")
    assert ed.quit == 1 and ed.unregistered == 1 and ed.callbacks == [mod.tick]
    assert ed.ended_pie and ed.loaded == ["/Game/RedRocks/Lvl_RedRocks"]
    assert [s["file"] for s in ed.shots] == ["before_postcard.png", "calib_postcard.png", "before_level_postcard.png"]
    for name in ("before_postcard.png", "calib_postcard.png", "before_level_postcard.png"):
        assert Image.open(board / name).size == (1600, 1200)
    sun = mod.sun_rotation(_view()["sun"])
    first, calib, level = ed.shots
    assert first["sun"] == (round(sun["pitch"], 6), round(sun["yaw"], 6), 0.0)
    assert calib["sun"] == first["sun"] and level["sun"] == (-58.0, 40.0, 0.0)  # the level's own
    assert ed.sun.rotation.key() == (-58.0, 40.0, 0.0)  # restored
    for shot in ed.shots:
        assert shot["camera"] == (-8.0, -150.0, 1.5) and shot["fov"] == 63.0  # straight from views.json
        assert shot["text_hidden"] and shot["size"] == (1600, 1200)
    assert [len(s["markers"]) for s in ed.shots] == [0, 2, 0]
    assert calib["tonemapper"] == "ShowFlag.Tonemapper 0" and level["tonemapper"] == "ShowFlag.Tonemapper 2"
    marker = calib["markers"][0]
    assert marker.scale.x == pytest.approx(0.6)  # 60 cm across: the engine sphere is 100 cm
    assert marker.comps["StaticMeshComponent"].props["affect_dynamic_indirect_lighting"] is False
    # Every spawned actor is transient and gone again; the text is back; nothing saved.
    assert ed.spawned and all(a.transient for a in ed.spawned) and set(ed.destroyed) == set(ed.spawned)
    assert not any(a.hidden_ed for a in ed.actors)
    assert "DisableAllScreenMessages" in ed.console and ed.game_view is False
    assert "ok    hid 3 text actor(s)" in text
    assert text.strip().splitlines()[-1].startswith("ok    postcard")
    assert "before_postcard.png, calib_postcard.png, before_level_postcard.png" in text


def test_every_shot_waits_eight_seconds_after_the_camera_or_sun_moves(tmp_path, monkeypatch):
    ed = _Editor()
    _run(tmp_path, monkeypatch, [_view("postcard"), _view("aerial")], ed,
         REDROCKS_BOARD_SUN="both", REDROCKS_BOARD_CALIB="1")
    assert [s["file"].split("_")[0] for s in ed.shots] == ["before", "calib", "before"] * 2
    for s in ed.shots:
        assert s["since_move"] >= (3.0 if s["file"].startswith("calib_") else 8.0), s["file"]


def test_a_view_that_cannot_be_shot_fails_by_name_and_the_rest_still_shoot(tmp_path, monkeypatch):
    ed = _Editor()
    _, board, text = _run(tmp_path, monkeypatch, [_view("postcard", camera=None), _view("aerial")], ed)
    assert [s["file"] for s in ed.shots] == ["before_aerial.png"]
    lines = text.strip().splitlines()
    assert lines[-2].startswith("FAIL  postcard") and "no fitted camera" in lines[-2]
    assert lines[-1].startswith("ok    aerial")


def test_no_sun_actor_is_an_error_and_still_quits_cleanly(tmp_path, monkeypatch):
    ed = _Editor(sun=False)
    _, _, text = _run(tmp_path, monkeypatch, [_view("postcard")], ed)
    assert ed.quit == 1 and ed.shots == []
    assert "ERROR" in text and "no actor labelled Sun" in text
    assert text.strip().splitlines()[-1].startswith("FAIL  postcard")


def test_a_screenshot_that_never_lands_fails_that_view_after_the_timeout(tmp_path, monkeypatch):
    ed = _Editor()
    ed.pump = lambda: None  # the PNG never arrives
    _, _, text = _run(tmp_path, monkeypatch, [_view("postcard")], ed)
    assert ed.quit == 1
    assert "FAIL  postcard" in text and "no complete before_postcard.png after 120 s" in text

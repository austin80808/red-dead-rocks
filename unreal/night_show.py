"""A night show at Red Rocks for beauty shots, in the editor world (nothing saved).

shoot_board.py calls ``apply(world_actors, cfg)`` before a view whose board entry
has ``"night": {...}`` and ``restore()`` before a day view and at the end. Every
actor it spawns is transient and destroyed by restore(); every property it
changes on the level's own actors (sun, fog, post process) is put back.

What a show looks like at Red Rocks (photos on Commons and the venue's own):
the bowl dark blue under a moonless sky, the stage roof a block of warm white
and saturated colour, movers raking the rows, narrow beams fanning up into the
haze, and Creation Rock and Ship Rock washed in deep colour from the stage.

``cfg`` keys (level cm; all optional except ``stage``):
  stage        [[x, y, z] front-north deck corner, [x, y, z] front-south deck corner]
  roof_z       underside of the roof (cm)
  back         [[x, y], [x, y]] back-north and back-south roof corners
  rocks        [[x, y, z], ...] rock faces to wash
  exposure_ev  fixed EV100 for the shot
  fog_density  height-fog density at night (volumetric fog on)
"""

import math

import unreal

log = []
_state = {}

PALETTE = [  # the colours a Red Rocks rig runs: magenta, deep blue, amber, cyan, violet, white
    (1.0, 0.08, 0.55), (0.08, 0.25, 1.0), (1.0, 0.5, 0.05),
    (0.05, 0.85, 1.0), (0.55, 0.1, 1.0), (1.0, 0.92, 0.8)]
MOVER_CD = 4.0e5
BEAM_CD = 2.0e6
WASH_CD = 6.0e4
ROCK_CD = 1.6e6
MOON_LUX = 0.25


def _sub(name):
    return unreal.get_editor_subsystem(getattr(unreal, name))


def _label(actor):
    try:
        return actor.get_actor_label()
    except Exception:
        return "?"


def _try(what, fn):
    try:
        return fn()
    except Exception as exc:  # keep shooting: one light is not worth the board
        log.append("warn  night %s: %r" % (what, exc))
        return None


def _spawn(cls, xyz, rot, label):
    a = _sub("EditorActorSubsystem").spawn_actor_from_class(
        cls, unreal.Vector(*xyz), unreal.Rotator(pitch=rot[0], yaw=rot[1], roll=0.0), True)
    if a is None:
        raise RuntimeError("could not spawn %s" % label)
    _try("label", lambda: a.set_actor_label(label, False))
    _state["spawned"].append(a)
    return a


def _lerp(a, b, t):
    return [a[i] + (b[i] - a[i]) * t for i in range(len(a))]


def _aim(src, dst):
    dx, dy, dz = dst[0] - src[0], dst[1] - src[1], dst[2] - src[2]
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    return pitch, yaw


def _spot(xyz, rot, cd, colour, inner, outer, radius, scatter, label, shadows=False):
    a = _spawn(unreal.SpotLight, xyz, rot, label)
    c = a.get_component_by_class(unreal.SpotLightComponent)
    c.set_mobility(unreal.ComponentMobility.MOVABLE)
    c.set_editor_property("intensity_units", unreal.LightUnits.CANDELAS)
    c.set_intensity(cd)
    c.set_light_color(unreal.LinearColor(r=colour[0], g=colour[1], b=colour[2], a=1.0))
    c.set_editor_property("inner_cone_angle", inner)
    c.set_editor_property("outer_cone_angle", outer)
    c.set_editor_property("attenuation_radius", radius)
    c.set_editor_property("volumetric_scattering_intensity", scatter)
    c.set_cast_shadows(shadows)
    return a


def apply(actors, cfg):
    """Turn the level to night with a show on the stage. Idempotent."""
    if _state.get("on"):
        return
    _state.clear()
    _state.update(on=True, spawned=[], restore=[])
    by_class = lambda cls: [a for a in actors if isinstance(a, cls)]

    # Sun just under the horizon: the SkyAtmosphere turns the sky deep twilight
    # blue (the planet shadows the direct light, so nothing is lit from below).
    # cfg "sun_off": true goes to full black night instead.
    for sun in by_class(unreal.DirectionalLight):
        comp = sun.get_component_by_class(unreal.DirectionalLightComponent)
        rot, lux = sun.get_actor_rotation(), comp.get_editor_property("intensity")
        _state["restore"].append(lambda sun=sun, rot=rot: sun.set_actor_rotation(rot, False))
        _state["restore"].append(lambda comp=comp, lux=lux: comp.set_intensity(lux))
        sun.set_actor_rotation(unreal.Rotator(pitch=float(cfg.get("sun_below_deg", 6.0)), yaw=rot.yaw, roll=0.0), False)
        if cfg.get("sun_off"):
            comp.set_intensity(0.0)
    _try("volumetric fog cvar", lambda: unreal.SystemLibrary.execute_console_command(None, "r.VolumetricFog 1"))
    # A faint high moon in the south-west, so the rocks keep their outline.
    moon = _spawn(unreal.DirectionalLight, (0, 0, 40000), (-38.0, 30.0), "NightMoon")
    mc = moon.get_component_by_class(unreal.DirectionalLightComponent)
    mc.set_mobility(unreal.ComponentMobility.MOVABLE)
    mc.set_intensity(MOON_LUX)
    mc.set_light_color(unreal.LinearColor(r=0.62, g=0.72, b=1.0, a=1.0))
    _try("moon not atmosphere sun", lambda: mc.set_editor_property("atmosphere_sun_light", False))

    # Haze the beams can show in.
    for fog in by_class(unreal.ExponentialHeightFog):
        fc = fog.get_component_by_class(unreal.ExponentialHeightFogComponent)
        _try("volumetric fog", lambda fc=fc: fc.set_volumetric_fog(True))
        _state["restore"].append(lambda fc=fc: _try("fog vol off", lambda: fc.set_volumetric_fog(False)))
        for prop, val in (("fog_density", cfg.get("fog_density", 0.02)),
                          ("volumetric_fog_scattering_distribution", 0.6),
                          ("volumetric_fog_extinction_scale", 0.6),
                          ("start_distance", 0.0)):
            old = _try("read fog %s" % prop, lambda prop=prop: fc.get_editor_property(prop))
            if old is not None:
                _state["restore"].append(lambda fc=fc, prop=prop, old=old: fc.set_editor_property(prop, old))
            _try("fog %s" % prop, lambda prop=prop, val=val: fc.set_editor_property(prop, val))

    # Fixed night exposure.
    ev = cfg.get("exposure_ev", 2.0)
    for ppv in by_class(unreal.PostProcessVolume):
        st = ppv.get_editor_property("settings")
        old = (st.get_editor_property("auto_exposure_min_brightness"), st.get_editor_property("auto_exposure_max_brightness"),
               st.get_editor_property("bloom_intensity"))

        def put(lo, hi, bloom, ppv=ppv):
            s2 = ppv.get_editor_property("settings")
            s2.set_editor_property("override_auto_exposure_min_brightness", True)
            s2.set_editor_property("override_auto_exposure_max_brightness", True)
            s2.set_editor_property("override_bloom_intensity", True)
            s2.set_editor_property("auto_exposure_min_brightness", lo)
            s2.set_editor_property("auto_exposure_max_brightness", hi)
            s2.set_editor_property("bloom_intensity", bloom)
            ppv.set_editor_property("settings", s2)
        _state["restore"].append(lambda old=old, put=put: put(*old))
        put(ev, ev, 1.2)

    # The rig, hung under the roof's front edge, aimed up the bowl at the crowd.
    north, south = cfg["stage"]
    roof_z = cfg.get("roof_z", north[2] + 1300.0)
    back = cfg.get("back") or [north[:2], south[:2]]
    front_mid = _lerp(north, south, 0.5)
    crowd = cfg.get("crowd", [front_mid[0] - 6000.0, front_mid[1] + 2700.0, front_mid[2] + 1500.0])
    n_movers = 10
    for i in range(n_movers):
        t = (i + 0.5) / n_movers
        p = _lerp(north, south, t)
        p = [p[0] + 120.0, p[1], roof_z - 80.0]
        # Fan across the rows, mirrored: outer movers reach the aisles.
        spread = (t - 0.5) * 2.0
        target = [crowd[0] + 1500.0 * abs(spread), crowd[1] + 4200.0 * spread * (1 if north[1] < south[1] else -1),
                  crowd[2] - 600.0 * abs(spread)]
        _spot(p, _aim(p, target), MOVER_CD, PALETTE[(i // 2) % 5], 2.0, 6.0, 15000.0, 3.0, "NightMover%d" % i)
    # Beams from the deck's back edge, up through the haze.
    n_beams = 7
    deck_back_n = _lerp(north, back[0] + [north[2]], 0.6)
    deck_back_s = _lerp(south, back[1] + [south[2]], 0.6)
    for i in range(n_beams):
        t = i / (n_beams - 1.0)
        p = _lerp(deck_back_n, deck_back_s, t)
        p[2] = north[2] + 150.0
        lean = (t - 0.5) * 50.0
        _spot(p, (68.0 - 8.0 * abs(t - 0.5), 155.0 + lean), BEAM_CD, PALETTE[(1, 0, 3, 5, 3, 0, 1)[i % 7]],
              0.8, 1.8, 60000.0, 10.0, "NightBeam%d" % i)
    # Warm white key on the deck from the roof, and a cool back wash.
    for j, (frac, colour) in enumerate(((0.3, (1.0, 0.85, 0.65)), (0.7, (1.0, 0.85, 0.65)))):
        p = _lerp(north, south, frac)
        p = [p[0] + 700.0, p[1], roof_z - 50.0]
        tgt = [p[0] - 400.0, p[1], north[2]]
        _spot(p, _aim(p, tgt), WASH_CD, colour, 30.0, 55.0, 4000.0, 1.0, "NightKey%d" % j, shadows=True)
    # Colour washes on the rock faces.
    for k, rock in enumerate(cfg.get("rocks", [])):
        src = _lerp(north if k % 2 == 0 else south, rock, 0.12)
        src[2] = north[2] + 400.0
        colour = cfg.get("rock_colours", [(0.9, 0.1, 0.6), (0.15, 0.3, 1.0), (0.6, 0.15, 1.0)])[k % 3]
        _spot(src, _aim(src, rock), ROCK_CD, colour, 14.0, 26.0, 40000.0, 1.5, "NightRock%d" % k)
    log.append("ok    night: sun off, moon %.2f lux, %d show light(s), EV100 %.1f"
               % (MOON_LUX, len(_state["spawned"]) - 1, ev))


def restore():
    if not _state.get("on"):
        return
    for fn in reversed(_state.get("restore", [])):
        _try("restore", fn)
    for a in _state.get("spawned", []):
        _try("destroy %s" % _label(a), lambda a=a: _sub("EditorActorSubsystem").destroy_actor(a))
    _state.clear()
    log.append("ok    night: restored the day")

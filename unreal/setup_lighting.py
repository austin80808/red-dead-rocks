"""Light Lvl_RedRocks: a summer late-morning sun, sky, clouds, fog and a player start.

Run headless:
    UnrealEditor-Cmd RedRocks.uproject -run=pythonscript \
        -script=unreal/setup_lighting.py -unattended -nosplash -stdout

A programmatic level has no directional light, sky or atmosphere, and renders
black until they exist. This script builds the whole outdoor rig, physically
based so it reads right at eye level and from the air with one exposure range
(configure_level.py sets the exposure and turns Lumen on):

* Sun: a real-world-bright directional light (lux), high in the south-east on
  a clear summer late morning: the light most of the photo board's Commons
  photos were taken under (short shadows, white light, sandstone its own red
  rather than golden-hour orange). The SkyAtmosphere tints and dims it through
  the air, so its colour stays white.
* SkyAtmosphere: Earth's default haze and aerial perspective: Front Range air
  is dry, and the postcard view shows downtown Denver crisp at ~25 km.
* SkyLight: real-time capture of that sky, so shade is lit by the sky's real
  colour (Lumen adds the bounce from the red ground).
* Height fog: a trace of haze low on the plains, coloured by the sky
  atmosphere; thin enough that Denver's skyline stays visible over the stage.
* Volumetric clouds: the engine's simple cloud layer, high and sparse.

Every rig actor is replaced on each run, so re-running updates its settings.
Log: /tmp/redrocks_lighting.log ("ok"/"FAIL" lines).
"""

import os
import sys
import unreal

_HERE = os.path.dirname(os.path.abspath(globals().get("__file__") or "/Users/austin/dev/redrocks-ue/unreal/x"))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import level_frame  # noqa: E402  (beside this script: terrain frame -> level frame)

LEVEL_PATH = "/Game/RedRocks/Lvl_RedRocks"
LOG_PATH = os.path.join(os.environ.get("REDROCKS_LOG_DIR", "/tmp"), "redrocks_lighting.log")

# ---- Sun ---------------------------------------------------------------------
# Red Rocks is at 39.67 N. At 10:30 solar time in late June the sun stands
# about 65 degrees up at a bearing of about 124 (south-east); a little later in
# the season or the morning gives the 130 / 58 used here, the light of most of
# the photo board's photos (short shadows under the benches, white light). The
# player spawns on the stage facing WSW up the bowl, so a morning sun is behind
# them and lights the rows and rock faces they see; an afternoon sun (WNW)
# backlit the whole view into silhouette. The light travels
# away from the sun:
# compass bearing (sun azimuth + 180), which in the level frame (X east, Y
# south, yaw clockwise from +X seen from above, level_frame.py) is
# yaw = bearing - 90.
SUN_AZIMUTH_DEG = 130.0   # compass bearing of the sun, clockwise from north (SE)
SUN_ELEVATION_DEG = 58.0  # above the horizon: late morning, high and white
# Outside the atmosphere the sun gives about 120,000 lux; the SkyAtmosphere
# takes off what the air absorbs, which at 58 degrees leaves ~95-105 klx.
SUN_LUX = 120000.0
SUN_SOURCE_ANGLE_DEG = 0.53  # the real solar disc: soft, true-width penumbrae

# ---- Sky light ---------------------------------------------------------------
SKY_LIGHT_INTENSITY = 1.0  # physical: the captured sky as-is
# Below the horizon the captured sky is black; the plains below the venue are
# red-brown dirt and dry grass, so bounce that colour up into the shade.
SKY_LOWER_HEMISPHERE = (0.14, 0.07, 0.04)

# ---- Atmosphere and fog ------------------------------------------------------
# The old golden-hour air (Mie 0.006, aerial perspective 2.5, fog 0.05) turned
# everything past the bowl grey; the photos show the plains crisp to Denver.
MIE_SCATTERING_SCALE = 0.0036  # Earth default is 0.003996: dry Front Range air
AERIAL_PERSPECTIVE_SCALE = 1.0  # physical: haze builds over tens of km, not a few
FOG_Z_CM = 0.0               # the plains east of the venue, terrain minimum
FOG_DENSITY = 0.004          # a trace: the SkyAtmosphere does the real haze
FOG_HEIGHT_FALLOFF = 0.2     # engine default: the haze sits low on the plains
FOG_START_CM = 200000.0      # clear air across the venue and the park
FOG_MAX_OPACITY = 0.85       # never a wall: Denver's towers always show through
DIRECTIONAL_INSCATTER = (0.45, 0.4, 0.33)  # a faint glow toward the sun
# The fog's colour away from the sun comes from the sky; tint it a touch toward
# the dusty grey of Front Range haze (the postcard's plains are neutral grey).
FOG_SKY_TINT = (1.12, 1.06, 1.0)

# ---- Clouds ------------------------------------------------------------------
CLOUD_MATERIAL = "/Engine/EngineSky/VolumetricClouds/m_SimpleVolumetricCloud_Inst"
CLOUD_BOTTOM_KM = 4.5
CLOUD_HEIGHT_KM = 4.0

RIG = ["Sun", "SkyAtmosphere", "SkyLight", "HeightFog", "Clouds"]

log = []


def sun_rotation(azimuth_deg, elevation_deg):
    """(pitch, yaw) of a directional light for a sun at this compass position.

    Plain arithmetic, tested offline. The light points away from the sun, so it
    pitches down by the elevation and faces the opposite bearing.
    """
    return -elevation_deg, level_frame.compass_yaw(azimuth_deg + 180.0)


def _step(label, fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        log.append("ok    %s" % label)
        return result
    except Exception as exc:
        log.append("FAIL  %s: %r" % (label, exc))
        return None


def _actors():
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


def _spawn(cls, location, label):
    actor = _actors().spawn_actor_from_class(cls, location)
    if actor is None:
        raise RuntimeError("could not spawn %s" % label)
    actor.set_actor_label(label)
    return actor


def _set(component, **props):
    for key, value in props.items():
        component.set_editor_property(key, value)


def _clear_rig():
    removed = 0
    for actor in list(_actors().get_all_level_actors()):
        if actor.get_actor_label() in RIG:
            _actors().destroy_actor(actor)
            removed += 1
    return removed


def _sun():
    pitch, yaw = sun_rotation(SUN_AZIMUTH_DEG, SUN_ELEVATION_DEG)
    sun = _spawn(unreal.DirectionalLight, unreal.Vector(0, 0, 100000), "Sun")
    sun.set_actor_rotation(unreal.Rotator(pitch=pitch, yaw=yaw, roll=0.0), False)
    comp = sun.get_component_by_class(unreal.DirectionalLightComponent)
    comp.set_mobility(unreal.ComponentMobility.MOVABLE)
    _set(comp, intensity=SUN_LUX, light_color=unreal.Color(r=255, g=255, b=255, a=255),
         use_temperature=False, atmosphere_sun_light=True, atmosphere_sun_light_index=0,
         light_source_angle=SUN_SOURCE_ANGLE_DEG, cast_shadows=True,
         cast_cloud_shadows=True, cloud_shadow_strength=0.6,
         cast_shadows_on_clouds=True)
    return "pitch %.1f yaw %.1f, %.0f lux" % (pitch, yaw, SUN_LUX)


def _atmosphere():
    sky = _spawn(unreal.SkyAtmosphere, unreal.Vector(0, 0, 0), "SkyAtmosphere")
    comp = sky.get_component_by_class(unreal.SkyAtmosphereComponent)
    _set(comp, mie_scattering_scale=MIE_SCATTERING_SCALE,
         aerial_pespective_view_distance_scale=AERIAL_PERSPECTIVE_SCALE)


def _sky_light():
    light = _spawn(unreal.SkyLight, unreal.Vector(0, 0, 50000), "SkyLight")
    comp = light.get_component_by_class(unreal.SkyLightComponent)
    comp.set_mobility(unreal.ComponentMobility.MOVABLE)
    r, g, b = SKY_LOWER_HEMISPHERE
    _set(comp, real_time_capture=True, intensity=SKY_LIGHT_INTENSITY,
         source_type=unreal.SkyLightSourceType.SLS_CAPTURED_SCENE,
         lower_hemisphere_is_black=False,
         lower_hemisphere_color=unreal.LinearColor(r=r, g=g, b=b, a=1.0))


def _fog():
    fog = _spawn(unreal.ExponentialHeightFog, unreal.Vector(0, 0, FOG_Z_CM), "HeightFog")
    comp = fog.get_component_by_class(unreal.ExponentialHeightFogComponent)
    r, g, b = DIRECTIONAL_INSCATTER
    # With a SkyAtmosphere the fog takes its colour from the sky; the flat
    # inscattering colour is zeroed so it does not paint everything grey-blue.
    _set(comp, fog_density=FOG_DENSITY, fog_height_falloff=FOG_HEIGHT_FALLOFF,
         start_distance=FOG_START_CM, fog_max_opacity=FOG_MAX_OPACITY,
         fog_inscattering_luminance=unreal.LinearColor(r=0.0, g=0.0, b=0.0, a=1.0),
         sky_atmosphere_ambient_contribution_color_scale=unreal.LinearColor(
             r=FOG_SKY_TINT[0], g=FOG_SKY_TINT[1], b=FOG_SKY_TINT[2], a=1.0),
         directional_inscattering_luminance=unreal.LinearColor(r=r, g=g, b=b, a=1.0),
         directional_inscattering_exponent=6.0,
         directional_inscattering_start_distance=FOG_START_CM,
         enable_volumetric_fog=False)


def _clouds():
    name = CLOUD_MATERIAL.rsplit("/", 1)[1]
    material = unreal.load_object(None, "%s.%s" % (CLOUD_MATERIAL, name))
    clouds = _spawn(unreal.VolumetricCloud, unreal.Vector(0, 0, 0), "Clouds")
    comp = clouds.get_component_by_class(unreal.VolumetricCloudComponent)
    if material is not None:
        comp.set_editor_property("material", material)
    if comp.get_editor_property("material") is None:
        raise RuntimeError("no cloud material at %s" % CLOUD_MATERIAL)
    _set(comp, layer_bottom_altitude=CLOUD_BOTTOM_KM, layer_height=CLOUD_HEIGHT_KM)
    return comp.get_editor_property("material").get_path_name()


def _player_start():
    # setup_gameplay.py moves it onto the stage; this only makes sure one exists.
    if any(isinstance(a, unreal.PlayerStart) for a in _actors().get_all_level_actors()):
        return "kept"
    _spawn(unreal.PlayerStart, unreal.Vector(0.0, 0.0, 18200.0), "PlayerStart")
    return "added"


def _save():
    if not unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level():
        raise RuntimeError("save_current_level returned False (read-only .umap?)")


def main():
    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not level_subsystem.load_level(LEVEL_PATH):
        log.append("ERROR: could not load %s" % LEVEL_PATH)
        return
    _step("clear old rig", _clear_rig)
    detail = _step("sun", _sun)
    if detail:
        log.append("      sun %s" % detail)
    _step("sky atmosphere", _atmosphere)
    _step("sky light (real-time capture)", _sky_light)
    _step("height fog", _fog)
    _step("volumetric clouds", _clouds)
    _step("player start", _player_start)
    _step("save %s" % LEVEL_PATH, _save)


if hasattr(unreal, "EditorAssetLibrary"):
    try:
        main()
    except Exception as exc:
        log.append("ERROR: %r" % (exc,))
    with open(LOG_PATH, "w") as fh:
        fh.write("\n".join(log) + "\n")

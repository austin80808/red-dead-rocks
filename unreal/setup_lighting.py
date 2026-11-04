"""Light Lvl_RedRocks: a golden-hour sun, sky, clouds, fog and a player start.

Run headless:
    UnrealEditor-Cmd RedRocks.uproject -run=pythonscript \
        -script=unreal/setup_lighting.py -unattended -nosplash -stdout

A programmatic level has no directional light, sky or atmosphere, and renders
black until they exist. This script builds the whole outdoor rig, physically
based so it reads right at eye level and from the air with one exposure range
(configure_level.py sets the exposure and turns Lumen on):

* Sun: a real-world-bright directional light (lux), low in the west-north-west
  for late-afternoon golden hour, when the Fountain Formation glows red. The
  SkyAtmosphere tints and dims it through the air, so its colour stays white.
* SkyAtmosphere: Earth defaults with a little more haze, and aerial perspective
  pulled in so the ridges a few km out turn blue-grey.
* SkyLight: real-time capture of that sky, so shade is lit by the sky's real
  colour (Lumen adds the bounce from the red ground).
* Height fog: thin haze that hides where the 2.4 km terrain and its skirt end,
  coloured by the sky atmosphere.
* Volumetric clouds: the engine's simple cloud layer, high and sparse.

Every rig actor is replaced on each run, so re-running updates its settings.
Log: /tmp/redrocks_lighting.log ("ok"/"FAIL" lines).
"""

import unreal

LEVEL_PATH = "/Game/RedRocks/Lvl_RedRocks"
LOG_PATH = "/tmp/redrocks_lighting.log"

# ---- Sun ---------------------------------------------------------------------
# Red Rocks is at 39.67 N. Late afternoon in summer the sun sits in the WNW, a
# little over 20 degrees up. The light travels away from it, toward the ESE:
# compass bearing (sun azimuth + 180), which in Unreal's frame (X east, Y north,
# yaw counter-clockwise from +X) is yaw = 90 - bearing.
SUN_AZIMUTH_DEG = 290.0   # compass bearing of the sun, clockwise from north
SUN_ELEVATION_DEG = 24.0  # above the horizon: low and warm, but the bowl is still lit
# Outside the atmosphere the sun gives about 120,000 lux; the SkyAtmosphere
# takes off what the air absorbs, which at 24 degrees leaves a warm ~40-60 klx.
SUN_LUX = 120000.0
SUN_SOURCE_ANGLE_DEG = 0.53  # the real solar disc: soft, true-width penumbrae

# ---- Sky light ---------------------------------------------------------------
SKY_LIGHT_INTENSITY = 1.0  # physical: the captured sky as-is
# Below the horizon the captured sky is black; the plains below the venue are
# red-brown dirt and dry grass, so bounce that colour up into the shade.
SKY_LOWER_HEMISPHERE = (0.14, 0.07, 0.04)

# ---- Atmosphere and fog ------------------------------------------------------
MIE_SCATTERING_SCALE = 0.006   # Earth default is 0.003996: a touch hazier
AERIAL_PERSPECTIVE_SCALE = 2.5  # >1 brings the blue-grey haze closer
FOG_Z_CM = 0.0               # the plains east of the venue, terrain minimum
FOG_DENSITY = 0.05
FOG_HEIGHT_FALLOFF = 0.06    # thins out over the ~600 m of relief
FOG_START_CM = 50000.0       # clear air across the venue itself
DIRECTIONAL_INSCATTER = (0.9, 0.55, 0.3)  # warm glow toward the low sun
# The fog's colour away from the sun comes from the sky, which alone paints
# the far plains ocean-blue. Warm it toward the dusty grey of Front Range haze.
FOG_SKY_TINT = (1.45, 1.22, 1.0)

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
    bearing = (azimuth_deg + 180.0) % 360.0
    yaw = (90.0 - bearing + 180.0) % 360.0 - 180.0
    return -elevation_deg, yaw


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
         start_distance=FOG_START_CM, fog_max_opacity=1.0,
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

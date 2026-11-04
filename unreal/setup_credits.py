"""Build the in-game credits overlay: press C while playing to show or hide it.

Run after setup_gameplay.py (rebuild_level.py runs it last). Headless:
    UnrealEditor-Cmd RedRocks.uproject -run=pythonscript \
        -script=/Users/austin/dev/redrocks-ue/unreal/setup_credits.py \
        -unattended -nosplash -stdout
or from the editor console:
    py "/Users/austin/dev/redrocks-ue/unreal/setup_credits.py"

Everything is generated here, nothing is hand-made in the editor, so a rebuild
recreates it. Each step is logged to /tmp/redrocks_credits.log as "ok" or
"FAIL"; an uncaught exception is "ERROR", missing source data is "ABORT".

  1. Reads the credit lines from data, so they cannot drift from the sources:
     assets/scan/ATTRIBUTION.md (the ml0w scan, CC BY 4.0), out/manifest.json's
     `attribution` list (USGS 3DEP, OpenStreetMap) and
     assets/rocks/ATTRIBUTION.md (Poly Haven, CC0). The same sources feed the
     in-world credits that setup_gameplay.py places beside the spawn.
  2. Creates the input action /Game/RedRocks/UI/IA_Credits and maps the C key
     to it in the template's /Game/Input/IMC_Default, which the FirstPerson
     player controller already adds at Play.
  3. Builds the Widget Blueprint /Game/RedRocks/UI/WBP_Credits with UMG's
     widget tree: a WidgetSwitcher named "Switcher" whose page 0 is a small
     "C  Credits" key hint in the bottom-left corner and page 1 is the credits
     card over a blurred, dimmed screen. Its function ToggleCredits flips the
     page (1 - active index).
  4. Builds the Actor Blueprint /Game/RedRocks/UI/BP_CreditsMenu. Its graph:
     BeginPlay -> Enable Input (player 0) -> Create Widget (WBP_Credits) ->
     store it in CreditsWidget -> Add to Viewport; IA_Credits Started ->
     CreditsWidget.ToggleCredits.
  5. Places one BP_CreditsMenu actor (label "CreditsMenu") in Lvl_RedRocks and
     saves. Re-running deletes and rebuilds both Blueprints and the actor.

Every save is checked: a False from save_asset or save_current_level (a
read-only Git LFS file, say) is a FAIL, not a silent no-op.
"""

import json
import os
import re

import unreal

REPO_DIR = "/Users/austin/dev/redrocks-ue"
OUT_DIR = "/Users/austin/dev/redrocks-ue/out"
SCAN_ATTRIBUTION = os.path.join(REPO_DIR, "assets", "scan", "ATTRIBUTION.md")
ROCKS_ATTRIBUTION = os.path.join(REPO_DIR, "assets", "rocks", "ATTRIBUTION.md")
LEVEL_PATH = "/Game/RedRocks/Lvl_RedRocks"
UI_DIR = "/Game/RedRocks/UI"
IA_NAME = "IA_Credits"
WBP_NAME = "WBP_Credits"
BP_NAME = "BP_CreditsMenu"
IMC_PATH = "/Game/Input/IMC_Default"
ACTOR_LABEL = "CreditsMenu"
TOGGLE_KEY = "C"
TOGGLE_FN = "ToggleCredits"
SWITCHER = "Switcher"  # page 0 = hint, page 1 = credits card
WIDGET_VAR = "CreditsWidget"
LOG_PATH = "/tmp/redrocks_credits.log"

# Links that the licences ask for but the one-line manifest entries don't
# carry. Matched by substring against the entry's text.
LINK_HINTS = {
    "OpenStreetMap": "openstreetmap.org/copyright",
}

log = []


# --------------------------------------------------------------------------
# Credit lines from data. Plain Python: tests/test_rebuild_level.py runs these
# without Unreal.
# --------------------------------------------------------------------------

def _strip_scheme(url):
    return re.sub(r"^https?://(www\.)?", "", url.strip()).rstrip("/")


def _md_field(text, name):
    m = re.search(r"\*\*%s:\*\*\s*(.+)" % re.escape(name), text)
    return m.group(1).strip() if m else None


def parse_scan_attribution(text):
    """The ml0w scan's credit from assets/scan/ATTRIBUTION.md.

    Returns a dict: role, who, licence, link. Raises ValueError when the file
    no longer has the fields a CC BY credit needs (title, author, licence,
    source), so a broken file stops the build instead of shipping no credit.
    """
    title = re.search(r"^#\s+(.+)$", text, re.M)
    author = _md_field(text, "Author")
    licence = _md_field(text, "Licence") or _md_field(text, "License")
    source = _md_field(text, "Source")
    if not (title and author and licence and source):
        raise ValueError("scan ATTRIBUTION.md lacks title/Author/Licence/Source")
    name, _, kind = title.group(1).partition(" \u2014 ")
    return {
        "role": (kind or "3D scan").strip().capitalize(),
        "who": "\u201c%s\u201d by %s" % (name.strip(), author.split()[0]),
        "licence": re.split(r"\s+[\u2014-]\s+", licence)[0].strip(),
        "link": _strip_scheme(source),
    }


def parse_rocks_attribution(text):
    """Poly Haven's credit from assets/rocks/ATTRIBUTION.md ("All from
    [Poly Haven](https://polyhaven.com), **CC0** ...")."""
    m = re.search(r"\[([^\]]+)\]\((https?://[^)]+)\),?\s*\*\*([^*]+)\*\*", text)
    if not m:
        raise ValueError("rocks ATTRIBUTION.md lacks '[source](url), **licence**'")
    return {
        "role": "Rock models",
        "who": m.group(1).strip(),
        "licence": m.group(3).strip(),
        "link": _strip_scheme(m.group(2)),
    }


def parse_manifest_line(line):
    """Split one manifest `attribution` line into role, who, licence, link.

    "Elevation: USGS 3DEP, public domain" -> Elevation / USGS 3DEP / Public
    domain. "Map data (c) OpenStreetMap contributors, ODbL 1.0" -> Map data /
    (c) OpenStreetMap contributors / ODbL 1.0, with the (c) as a real (c) sign.
    A line in neither shape is kept whole as `who`, so nothing is dropped.
    """
    text = line.strip()
    who_part, sep, licence = text.rpartition(", ")
    if not sep:
        who_part, licence = text, ""
    role, who = "", who_part
    m = re.match(r"^(?P<role>[^:]+?):\s*(?P<who>.+)$", who_part)
    c = re.match(r"^(?P<role>.+?)\s*(?:\(c\)|\u00a9)\s*(?P<who>.+)$", who_part, re.I)
    if m:
        role, who = m.group("role"), m.group("who")
    elif c:
        role, who = c.group("role"), "\u00a9 " + c.group("who")
    licence = licence.strip()
    if licence and licence[0].islower():
        licence = licence[0].upper() + licence[1:]
    link = next((url for key, url in LINK_HINTS.items() if key in text), "")
    return {"role": role.strip(), "who": who.strip(), "licence": licence, "link": link}


def credit_entries(manifest, scan_md, rocks_md):
    """All credits, in the in-world order: the scan, the manifest's list, the
    rock models."""
    entries = [parse_scan_attribution(scan_md)]
    entries.extend(parse_manifest_line(l) for l in manifest.get("attribution", []))
    entries.append(parse_rocks_attribution(rocks_md))
    return entries


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# --------------------------------------------------------------------------
# Unreal side
# --------------------------------------------------------------------------

def _step(label, fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        log.append("ok    %s" % label)
        return result
    except Exception as exc:  # keep going; the runner stops on FAIL lines
        log.append("FAIL  %s: %r" % (label, exc))
        return None


def _save(path):
    if not unreal.EditorAssetLibrary.save_asset(path, only_if_is_dirty=False):
        raise RuntimeError("could not save %s (read-only file? chmod -R u+w Content)" % path)
    return path


def _actors():
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()


def _srgb(hex_color, alpha=1.0):
    """sRGB hex -> linear LinearColor (UMG colours are linear)."""
    out = []
    for i in (0, 2, 4):
        c = int(hex_color[i:i + 2], 16) / 255.0
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return unreal.LinearColor(out[0], out[1], out[2], alpha)


# Palette: warm darks and the sandstone red of the rocks.
INK = "f5eee6"        # titles
SAND = "c9b6a4"       # body
DUST = "8f7f72"       # small print
ROCK = "e4744a"       # accent (Red Rocks sandstone)
CARD = "17110e"       # card background

FONT = "/Engine/EngineFonts/Roboto.Roboto"
# Wide enough for the Sketchfab link (the longest line) on one row.
CARD_WIDTH = 960.0


def _font(size, face="Regular", spacing=0):
    info = unreal.SlateFontInfo()
    info.set_editor_property("font_object", unreal.load_asset(FONT))
    info.set_editor_property("typeface_font_name", face)
    info.set_editor_property("size", size)
    info.set_editor_property("letter_spacing", spacing)
    return info


def _slate(color):
    return unreal.SlateColor(specified_color=color)


def _rounded(color, radius, outline=None, outline_width=0.0):
    brush = unreal.SlateBrush()
    brush.set_editor_property("draw_as", unreal.SlateBrushDrawType.ROUNDED_BOX)
    brush.set_editor_property("tint_color", _slate(color))
    o = unreal.SlateBrushOutlineSettings()
    o.set_editor_property("corner_radii", unreal.Vector4(radius, radius, radius, radius))
    o.set_editor_property("rounding_type", unreal.SlateBrushRoundingType.FIXED_RADIUS)
    if outline is not None:
        o.set_editor_property("color", _slate(outline))
        o.set_editor_property("width", outline_width)
    brush.set_editor_property("outline_settings", o)
    return brush


def _solid(color, w=0.0, h=0.0):
    brush = unreal.SlateBrush()
    brush.set_editor_property("draw_as", unreal.SlateBrushDrawType.IMAGE)
    brush.set_editor_property("tint_color", _slate(color))
    if w or h:
        size = unreal.DeprecateSlateVector2D()
        size.set_editor_property("x", w)
        size.set_editor_property("y", h)
        brush.set_editor_property("image_size", size)
    return brush


class _Tree:
    """Adds named widgets to a Widget Blueprint's tree (EditorUtilityLibrary)."""

    def __init__(self, wbp):
        self.wbp = wbp
        self.n = 0

    def add(self, cls, parent, name=None):
        if name is None:
            self.n += 1
            name = "%s_%d" % (cls.__name__, self.n)
        widget = unreal.EditorUtilityLibrary.add_source_widget(
            self.wbp, cls, name, parent if parent else "None")
        if widget is None:
            raise RuntimeError("could not add %s %s under %s" % (cls.__name__, name, parent))
        return widget

    def text(self, parent, text, size, color, face="Regular", spacing=0, name=None):
        tb = self.add(unreal.TextBlock, parent, name)
        tb.set_editor_property("text", text)
        tb.set_editor_property("font", _font(size, face, spacing))
        tb.set_editor_property("color_and_opacity", _slate(_srgb(color)))
        tb.set_editor_property("shadow_offset", unreal.Vector2D(0.0, 1.0))
        tb.set_editor_property("shadow_color_and_opacity", unreal.LinearColor(0, 0, 0, 0.6))
        return tb


def _vpad(slot, top=0.0, bottom=0.0, h=None):
    slot.set_editor_property("padding", unreal.Margin(0.0, top, 0.0, bottom))
    if h is not None:
        slot.set_editor_property("horizontal_alignment", h)


def _keycap(tree, parent, label, size=15):
    """A little rounded key, like a keyboard key: the accent fill, dark text."""
    cap = tree.add(unreal.Border, parent)
    cap.set_editor_property("background", _rounded(_srgb(ROCK), 6.0))
    cap.set_editor_property("padding", unreal.Margin(9.0, 2.0, 9.0, 3.0))
    cap.set_editor_property("horizontal_alignment", unreal.HorizontalAlignment.H_ALIGN_CENTER)
    cap.set_editor_property("vertical_alignment", unreal.VerticalAlignment.V_ALIGN_CENTER)
    t = tree.text(cap.get_name(), label, size, CARD, "Bold")
    t.set_editor_property("shadow_offset", unreal.Vector2D(0.0, 0.0))
    return cap


def _hint_page(tree, parent):
    """Page 0: "C  Credits" pill in the bottom-left corner."""
    layer = tree.add(unreal.CanvasPanel, parent, "HintLayer")
    pill = tree.add(unreal.Border, layer.get_name(), "Hint")
    pill.set_editor_property("background", _rounded(_srgb(CARD, 0.72), 10.0, _srgb(ROCK, 0.35), 1.0))
    pill.set_editor_property("padding", unreal.Margin(10.0, 8.0, 16.0, 8.0))
    slot = pill.slot
    slot.set_anchors(unreal.Anchors(minimum=unreal.Vector2D(0.0, 1.0), maximum=unreal.Vector2D(0.0, 1.0)))
    slot.set_alignment(unreal.Vector2D(0.0, 1.0))
    slot.set_auto_size(True)
    slot.set_position(unreal.Vector2D(36.0, -36.0))
    row = tree.add(unreal.HorizontalBox, pill.get_name())
    cap = _keycap(tree, row.get_name(), TOGGLE_KEY)
    cap.slot.set_editor_property("vertical_alignment", unreal.VerticalAlignment.V_ALIGN_CENTER)
    label = tree.text(row.get_name(), "Credits", 16, INK, "Bold", 60)
    label.slot.set_editor_property("padding", unreal.Margin(12.0, 0.0, 0.0, 0.0))
    label.slot.set_editor_property("vertical_alignment", unreal.VerticalAlignment.V_ALIGN_CENTER)
    return layer


def _entry(tree, parent, entry, first):
    box = tree.add(unreal.VerticalBox, parent)
    _vpad(box.slot, top=0.0 if first else 22.0)
    if entry["role"]:
        role = tree.text(box.get_name(), entry["role"].upper(), 11, ROCK, "Bold", 220)
        _vpad(role.slot, bottom=4.0)
    who = tree.text(box.get_name(), entry["who"], 22, INK, "Bold")
    who.set_editor_property("auto_wrap_text", True)
    row = tree.add(unreal.HorizontalBox, box.get_name())
    _vpad(row.slot, top=6.0)
    if entry["licence"]:
        badge = tree.add(unreal.Border, row.get_name())
        badge.set_editor_property("background", _rounded(_srgb(ROCK, 0.14), 5.0, _srgb(ROCK, 0.7), 1.0))
        badge.set_editor_property("padding", unreal.Margin(8.0, 2.0, 8.0, 3.0))
        badge.slot.set_editor_property("vertical_alignment", unreal.VerticalAlignment.V_ALIGN_CENTER)
        t = tree.text(badge.get_name(), entry["licence"], 12, INK, "Bold", 40)
        t.set_editor_property("shadow_offset", unreal.Vector2D(0.0, 0.0))
    if entry["link"]:
        link = tree.text(row.get_name(), entry["link"], 13, SAND)
        link.slot.set_editor_property("padding", unreal.Margin(12.0 if entry["licence"] else 0.0, 0.0, 0.0, 0.0))
        link.slot.set_editor_property("vertical_alignment", unreal.VerticalAlignment.V_ALIGN_CENTER)


def _credits_page(tree, parent, entries):
    """Page 1: blurred, dimmed screen with the credits card in the middle."""
    layer = tree.add(unreal.Overlay, parent, "CreditsLayer")
    blur = tree.add(unreal.BackgroundBlur, layer.get_name(), "Blur")
    blur.set_editor_property("blur_strength", 7.0)
    blur.slot.set_editor_property("horizontal_alignment", unreal.HorizontalAlignment.H_ALIGN_FILL)
    blur.slot.set_editor_property("vertical_alignment", unreal.VerticalAlignment.V_ALIGN_FILL)
    dim = tree.add(unreal.Image, layer.get_name(), "Dim")
    dim.set_editor_property("brush", _solid(unreal.LinearColor(0.0, 0.0, 0.0, 0.5)))
    dim.slot.set_editor_property("horizontal_alignment", unreal.HorizontalAlignment.H_ALIGN_FILL)
    dim.slot.set_editor_property("vertical_alignment", unreal.VerticalAlignment.V_ALIGN_FILL)

    size = tree.add(unreal.SizeBox, layer.get_name(), "CardSize")
    size.set_width_override(CARD_WIDTH)
    size.slot.set_editor_property("horizontal_alignment", unreal.HorizontalAlignment.H_ALIGN_CENTER)
    size.slot.set_editor_property("vertical_alignment", unreal.VerticalAlignment.V_ALIGN_CENTER)
    card = tree.add(unreal.Border, size.get_name(), "Card")
    card.set_editor_property("background", _rounded(_srgb(CARD, 0.94), 18.0, _srgb(ROCK, 0.28), 1.0))
    card.set_editor_property("padding", unreal.Margin(64.0, 52.0, 64.0, 44.0))
    col = tree.add(unreal.VerticalBox, card.get_name(), "CardColumn")

    accent = tree.add(unreal.Image, col.get_name(), "Accent")
    accent.set_editor_property("brush", _solid(_srgb(ROCK), 56.0, 4.0))
    _vpad(accent.slot, bottom=20.0, h=unreal.HorizontalAlignment.H_ALIGN_LEFT)
    kicker = tree.text(col.get_name(), "RED ROCKS AMPHITHEATRE  \u00b7  MORRISON, COLORADO", 12, SAND, "Bold", 260)
    _vpad(kicker.slot, bottom=6.0)
    title = tree.text(col.get_name(), "Credits", 46, INK, "Bold", -10, name="Title")
    _vpad(title.slot, bottom=4.0)
    sub = tree.text(col.get_name(), "This level is built from open data and a community scan. Thank you.", 15, DUST, "Italic")
    sub.set_editor_property("auto_wrap_text", True)
    _vpad(sub.slot, bottom=26.0)
    rule = tree.add(unreal.Image, col.get_name(), "Rule")
    rule.set_editor_property("brush", _solid(_srgb(SAND, 0.18), 0.0, 1.0))
    _vpad(rule.slot, bottom=28.0, h=unreal.HorizontalAlignment.H_ALIGN_FILL)

    for i, entry in enumerate(entries):
        _entry(tree, col.get_name(), entry, i == 0)

    rule2 = tree.add(unreal.Image, col.get_name(), "Rule2")
    rule2.set_editor_property("brush", _solid(_srgb(SAND, 0.18), 0.0, 1.0))
    _vpad(rule2.slot, top=32.0, bottom=18.0, h=unreal.HorizontalAlignment.H_ALIGN_FILL)
    foot = tree.add(unreal.HorizontalBox, col.get_name(), "Footer")
    _vpad(foot.slot, h=unreal.HorizontalAlignment.H_ALIGN_RIGHT)
    press = tree.text(foot.get_name(), "Press", 14, DUST)
    press.slot.set_editor_property("vertical_alignment", unreal.VerticalAlignment.V_ALIGN_CENTER)
    cap = _keycap(tree, foot.get_name(), TOGGLE_KEY, 13)
    cap.slot.set_editor_property("padding", unreal.Margin(8.0, 0.0, 8.0, 0.0))
    cap.slot.set_editor_property("vertical_alignment", unreal.VerticalAlignment.V_ALIGN_CENTER)
    close = tree.text(foot.get_name(), "to close", 14, DUST)
    close.slot.set_editor_property("vertical_alignment", unreal.VerticalAlignment.V_ALIGN_CENTER)
    return layer


def _pin(node, name, out):
    lib = unreal.BlueprintEditorLibrary
    pin = lib.find_output_pin(node, name) if out else lib.find_input_pin(node, name)
    if pin is None or not pin.is_valid():
        names = [str(p.get_pin_name()) for p in lib.list_all_pins(node)]
        raise RuntimeError("%s has no %s pin %r; pins: %s" % (
            node.get_name(), "output" if out else "input", name, names))
    return pin


def _link(a, a_pin, b, b_pin):
    lib = unreal.BlueprintEditorLibrary
    src = lib.find_then_pin(a) if a_pin == "then" else _pin(a, a_pin, True)
    dst = lib.find_execute_pin(b) if b_pin == "execute" else _pin(b, b_pin, False)
    if src is None or not src.is_valid() or dst is None or not dst.is_valid():
        raise RuntimeError("no exec pin linking %s -> %s" % (a.get_name(), b.get_name()))
    if not src.try_create_connection(dst):
        raise RuntimeError("could not link %s.%s -> %s.%s" % (
            a.get_name(), a_pin, b.get_name(), b_pin))


def _set_pin(node, name, value):
    if not _pin(node, name, False).set_pin_value(value):
        raise RuntimeError("could not set %s.%s = %r" % (node.get_name(), name, value))


def _need(node, what):
    if node is None:
        raise RuntimeError("could not add node: %s" % what)
    return node


def _place(node, x, y):
    unreal.BlueprintEditorLibrary.set_node_pos(node, unreal.IntPoint(x, y))
    return node


def _compile(bp, path):
    if not unreal.BlueprintEditorLibrary.compile_blueprint(bp):
        bad = []
        for graph in unreal.BlueprintEditorLibrary.list_graphs(bp) if hasattr(
                unreal.BlueprintEditorLibrary, "list_graphs") else []:
            try:
                ed = unreal.BlueprintGraphEditor.get_graph_editor(graph)
                bad += ["%s: %s" % (n.get_name(), n.get_editor_property("error_msg"))
                        for n in ed.list_nodes_with_errors()]
            except Exception:
                pass
        raise RuntimeError("%s did not compile: %s" % (path, bad or "see Output Log"))


def _delete_old():
    eal = unreal.EditorAssetLibrary
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    removed = 0
    for actor in _actors():
        if actor.get_actor_label() == ACTOR_LABEL:
            subsystem.destroy_actor(actor)
            removed += 1
    for name in (BP_NAME, WBP_NAME):
        path = "%s/%s" % (UI_DIR, name)
        if eal.does_asset_exist(path) and not eal.delete_asset(path):
            raise RuntimeError("could not delete old %s" % path)
    return "%d old actor(s) removed" % removed


def _input_action():
    path = "%s/%s" % (UI_DIR, IA_NAME)
    eal = unreal.EditorAssetLibrary
    if not eal.does_directory_exist(UI_DIR):
        eal.make_directory(UI_DIR)
    ia = eal.load_asset(path) if eal.does_asset_exist(path) else None
    if ia is None:
        ia = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            IA_NAME, UI_DIR, unreal.InputAction, unreal.InputAction_Factory())
    ia.set_editor_property("value_type", unreal.InputActionValueType.BOOLEAN)
    ia.set_editor_property("action_description", "Show or hide the credits")
    _save(path)
    return ia


def _map_key(ia):
    imc = unreal.EditorAssetLibrary.load_asset(IMC_PATH)
    if imc is None:
        raise RuntimeError("no input mapping context at %s" % IMC_PATH)
    imc.unmap_all_keys_from_action(ia)
    key = unreal.Key()
    key.set_editor_property("key_name", TOGGLE_KEY)
    imc.map_key(ia, key)
    _save(IMC_PATH)
    return "%s -> %s in %s" % (TOGGLE_KEY, IA_NAME, IMC_PATH)


def _widget(entries):
    path = "%s/%s" % (UI_DIR, WBP_NAME)
    wbp = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        WBP_NAME, UI_DIR, unreal.WidgetBlueprint, unreal.WidgetBlueprintFactory())
    if wbp is None:
        raise RuntimeError("could not create %s" % path)
    tree = _Tree(wbp)
    root = tree.add(unreal.CanvasPanel, None, "Root")
    root.set_editor_property("visibility", unreal.SlateVisibility.SELF_HIT_TEST_INVISIBLE)
    switcher = tree.add(unreal.WidgetSwitcher, "Root", SWITCHER)
    slot = switcher.slot
    slot.set_anchors(unreal.Anchors(minimum=unreal.Vector2D(0.0, 0.0), maximum=unreal.Vector2D(1.0, 1.0)))
    slot.set_offsets(unreal.Margin(0.0, 0.0, 0.0, 0.0))
    switcher.set_editor_property("visibility", unreal.SlateVisibility.HIT_TEST_INVISIBLE)
    for page in (_hint_page(tree, SWITCHER), _credits_page(tree, SWITCHER, entries)):
        page.slot.set_editor_property("horizontal_alignment", unreal.HorizontalAlignment.H_ALIGN_FILL)
        page.slot.set_editor_property("vertical_alignment", unreal.VerticalAlignment.V_ALIGN_FILL)
    switcher.set_editor_property("active_widget_index", 0)
    _compile(wbp, path)  # makes the Switcher variable the graph needs

    # ToggleCredits: Switcher.SetActiveWidgetIndex(1 - Switcher.GetActiveWidgetIndex())
    ed = unreal.BlueprintGraphEditor.create_and_edit_function_graph(wbp, TOGGLE_FN)
    get_sw = _place(_need(ed.add_get_member_variable_node(SWITCHER), "get Switcher"), 0, 200)
    get_idx = _place(_need(ed.add_call_function_node(
        "/Script/UMG.WidgetSwitcher.GetActiveWidgetIndex"), "GetActiveWidgetIndex"), 220, 260)
    sub = _place(_need(ed.add_call_function_node(
        "/Script/Engine.KismetMathLibrary.Subtract_IntInt"), "Subtract_IntInt"), 480, 260)
    set_idx = _place(_need(ed.add_call_function_node(
        "/Script/UMG.WidgetSwitcher.SetActiveWidgetIndex"), "SetActiveWidgetIndex"), 700, 0)
    _link(get_sw, SWITCHER, get_idx, "self")
    _link(get_sw, SWITCHER, set_idx, "self")
    _link(get_idx, "ReturnValue", sub, "B")  # types the promotable operator as int
    _set_pin(sub, "A", "1")
    _link(sub, "ReturnValue", set_idx, "Index")
    entry = ed.find_graph_entry_pin()
    if not entry.try_create_connection(unreal.BlueprintEditorLibrary.find_execute_pin(set_idx)):
        raise RuntimeError("could not link ToggleCredits entry")
    _compile(wbp, path)
    _save(path)
    return wbp


def _function_path(wbp_class, fn):
    return "%s:%s" % (wbp_class.get_path_name(), fn)


def _menu_actor_blueprint(wbp, ia):
    path = "%s/%s" % (UI_DIR, BP_NAME)
    lib = unreal.BlueprintEditorLibrary
    bp = lib.create_blueprint_asset_with_parent(path, unreal.Actor)
    if bp is None:
        raise RuntimeError("could not create %s" % path)
    wbp_class = wbp.generated_class()
    if not lib.add_member_variable(bp, WIDGET_VAR, lib.get_object_reference_type(wbp_class)):
        raise RuntimeError("could not add %s variable" % WIDGET_VAR)
    _compile(bp, path)

    ed = unreal.BlueprintGraphEditor.get_graph_editor(lib.find_event_graph(bp))
    begin = _need(lib.add_event_override(bp, "ReceiveBeginPlay", unreal.IntPoint(0, 0)), "BeginPlay")
    pc = _place(_need(ed.add_call_function_node(
        "/Script/Engine.GameplayStatics.GetPlayerController"), "GetPlayerController"), 120, 220)
    enable = _place(_need(ed.add_call_function_node("/Script/Engine.Actor.EnableInput"), "EnableInput"), 300, 0)
    create = _place(_need(ed.create_node_from_name(
        "UserInterface|CreateWidget", unreal.Vector2D(600, 0), []), "Create Widget"), 600, 0)
    _set_pin(create, "Class", wbp_class.get_path_name())
    store = _place(_need(ed.add_set_member_variable_node(WIDGET_VAR), "set CreditsWidget"), 920, 0)
    get_w = _place(_need(ed.add_get_member_variable_node(WIDGET_VAR), "get CreditsWidget"), 1120, 160)
    show = _place(_need(ed.add_call_function_node("/Script/UMG.UserWidget.AddToViewport"), "AddToViewport"), 1300, 0)
    _set_pin(show, "ZOrder", "50")

    _link(begin, "then", enable, "execute")
    _link(pc, "ReturnValue", enable, "PlayerController")
    _link(enable, "then", create, "execute")
    _link(pc, "ReturnValue", create, "OwningPlayer")
    _link(create, "then", store, "execute")
    _link(create, "ReturnValue", store, WIDGET_VAR)
    _link(store, "then", show, "execute")
    _link(get_w, WIDGET_VAR, show, "self")

    on_key = _place(_need(ed.create_node_from_name(
        "Input|EnhancedActionEvents|%s" % IA_NAME, unreal.Vector2D(0, 400), []), "IA_Credits event"), 0, 420)
    get_w2 = _place(_need(ed.add_get_member_variable_node(WIDGET_VAR), "get CreditsWidget"), 300, 560)
    toggle = None
    for fn_path in (_function_path(wbp_class, TOGGLE_FN),
                    "%s.%s" % (wbp_class.get_path_name(), TOGGLE_FN)):
        toggle = ed.add_call_function_node(fn_path)
        if toggle is not None:
            break
    _place(_need(toggle, "call %s" % TOGGLE_FN), 560, 420)
    _link(on_key, "Started", toggle, "execute")
    _link(get_w2, WIDGET_VAR, toggle, "self")

    _compile(bp, path)
    _save(path)
    return bp


def _place_actor(bp):
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actor = subsystem.spawn_actor_from_class(bp.generated_class(), unreal.Vector(0.0, 0.0, 0.0))
    if actor is None:
        raise RuntimeError("could not spawn %s" % BP_NAME)
    actor.set_actor_label(ACTOR_LABEL)
    actor.tags = [unreal.Name("credits")]
    return actor.get_actor_label()


def _save_level():
    if not unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level():
        raise RuntimeError(
            "could not save %s; is Content/RedRocks/Lvl_RedRocks.umap read-only "
            "(git lfs lock it, or close the editor holding it)?" % LEVEL_PATH)


def main():
    for path in (SCAN_ATTRIBUTION, ROCKS_ATTRIBUTION, os.path.join(OUT_DIR, "manifest.json")):
        if not os.path.exists(path):
            log.append("ABORT: %s not found" % path)
            return
    with open(os.path.join(OUT_DIR, "manifest.json")) as fh:
        manifest = json.load(fh)
    entries = credit_entries(manifest, _read(SCAN_ATTRIBUTION), _read(ROCKS_ATTRIBUTION))
    for e in entries:
        log.append("      credit: %s | %s | %s | %s" % (e["role"], e["who"], e["licence"], e["link"]))

    if not unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).load_level(LEVEL_PATH):
        raise RuntimeError("could not load %s" % LEVEL_PATH)
    if _step("remove old credits menu", _delete_old) is None:
        return
    ia = _step("input action %s" % IA_NAME, _input_action)
    if ia is None:
        return
    _step("map %s in IMC_Default" % TOGGLE_KEY, _map_key, ia)
    wbp = _step("widget %s" % WBP_NAME, _widget, entries)
    if wbp is None:
        return
    bp = _step("actor blueprint %s" % BP_NAME, _menu_actor_blueprint, wbp, ia)
    if bp is None:
        return
    _step("place %s in level" % ACTOR_LABEL, _place_actor, bp)
    _step("save %s" % LEVEL_PATH, _save_level)


if hasattr(unreal, "EditorAssetLibrary"):
    try:
        main()
    except Exception as exc:
        log.append("ERROR: %r" % (exc,))
    with open(LOG_PATH, "w") as fh:
        fh.write("\n".join(log) + "\n")
    for line in log:
        (unreal.log_error if line.startswith(("FAIL", "ERROR", "ABORT")) else unreal.log)(line)

"""List level actors with their bounds (for finding what makes a shape in a shot)."""
import unreal
out = []
for a in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
    try:
        o, e = a.get_actor_bounds(False)
        out.append("%-40s %-28s c=(%7.0f %7.0f %7.0f) e=(%7.0f %7.0f %7.0f) hid=%s" % (
            a.get_actor_label()[:40], a.get_class().get_name()[:28], o.x, o.y, o.z, e.x, e.y, e.z,
            a.is_hidden_ed()))
    except Exception as exc:
        out.append("%s %r" % (a.get_actor_label(), exc))
open("/workspace/work/redrocks/logs/actors.txt", "w").write("\n".join(sorted(out)))

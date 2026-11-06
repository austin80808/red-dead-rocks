# The sasquatch player character: design

Date: 2026-11-06. Status: draft for Austin's review (on the build PR). Built ahead of
approval because the task said to; nothing has been compiled or rendered yet.

## What Austin asked for

"You can go ahead and design the most realistic sasquatch you can cuz thats the
character." The player in the Red Rocks game becomes a sasquatch. It has to read
as a real animal, not a man in a suit: heavy, hairy, ape-like face, long arms.
You play it in first person (as today) or switch to third person to see him.

Not in this task: the mushroom easter egg (eat mushrooms, the world gets
trippier), climbing, wingsuit and riding animations (see Follow-ups).

## Success

1. A side-by-side render (front, side, three-quarter) in
   `docs/media/sasquatch/` that Austin looks at and says "that's a sasquatch".
2. In game, V switches between first and third person. In first person you see
   his hairy arms and hands when you look down or move; in third person he
   idles, walks and runs with retargeted mannequin animations, with no foot
   sliding you notice at walk speed.
3. Everything is rebuilt from scripts in this repo, from free content only.

## Anatomy and proportions

Built after the Patterson-Gimlin film subject and the common field-report
picture, pushed toward a great ape where reports are vague.

| Part | Target |
|---|---|
| Height | 2.55 m to the top of the crest (2.4-2.7 m allowed); 1.42x the 1.80 m mannequin |
| Build | about 300 kg: shoulders 1.05 m across, deep barrel chest, no waist, heavy thighs and buttocks |
| Neck | none visible: the trapezius runs from the back of the head to the shoulders; head carried forward |
| Arms | long: fingertips reach just above the knee (intermembral index about 0.85, human 0.70); upper arm and forearm 1.18x the mannequin's after the overall scale |
| Hands | 1.3x a scaled human hand, thick fingers, dark leathery palms and finger pads |
| Feet | 40 cm long, 18 cm wide, flat (no arch), five toes in line, leathery soles |
| Head | conical: a sagittal crest peaking at the back of the crown, sloping forehead, heavy brow ridge, deep-set eyes, broad flat nose with flared nostrils, a short muzzle, wide thin lips, receding chin, small ears under the hair |
| Eyes | dark brown iris and dark sclera (an ape's, no white showing), wet |
| Skin | dark grey-brown, leathery and wrinkled; bare on the face's centre (eyes, nose, mouth), palms and soles |
| Hair | dark brown-auburn at the root, warming to reddish brown, tipping to lighter grey-brown; 10-18 cm over the shoulders, back and forearms (hanging off the arms as in the film), 6-10 cm on the body and legs, 4-6 cm on the head, 1-3 cm on the face and backs of the hands; clumped and a little matted |
| Gait | compliant: knees bent through the stride, upper body pitched forward, long arm swing |

## How realism is reached

**Chosen: a procedural sculpt of the UE5 mannequin plus strand fur, all from
code.** Everything comes from the mannequin that ships with the project and a
generator in this repo, so it is free, rebuildable and in git.

1. `unreal/export_mannequin.py` exports `SKM_Manny_Simple` as glTF (to `out/`,
   never committed: it is Epic template content).
2. `redrocks sasquatch` (`src/redrocks/sasquatch/`) reshapes it:
   - **Proportions** by changing the bind pose: the whole skeleton scaled 1.42x,
     then arms, hands, feet and spine lengthened or shortened bone by bone; the
     skin follows by linear blend skinning, so weights stay valid.
   - **Mass** by moving vertices along their normals with smooth regional fields
     (trapezius, chest, belly, buttocks, thighs, calves, forearms).
   - **Head, hands and feet** subdivided for detail, then the head shrink-wrapped
     onto a sculpted signed-distance head (crest, brow, sockets, nose, muzzle,
     lips, ears) and the feet given toes and a flat sole. Eyes are separate
     spheres on the head bone.
   - **Fur:** about 250,000 strands grown from the final surface: area-weighted
     roots, a density and length map per region (bare face centre, palms and
     soles), combed down the body and toward the elbows on the arms (as on apes),
     lifted off the skin, drooping under gravity, clumped and frizzed, coloured
     root to tip. Written as a small binary `.rrgroom` file.
   - Writes `out/sasquatch/sasquatch.glb` (skinned mesh), the groom files and a
     manifest of the measured proportions.
3. `unreal/setup_sasquatch.py` imports them into `/Game/Sasquatch`: the skeletal
   mesh, the grooms (through a small `.rrgroom` translator in a new
   `RedRocksEditor` module) bound to the mesh, and the materials (Substrate skin
   with subsurface scattering and procedural wrinkles, hair shading for the fur,
   wet eyes).

Rejected: **MetaHuman** scaled and re-proportioned. Its faces are human, and a
hairy human is exactly what this must not look like; its assets also can't go in
a public repo. **A hand-sculpted mesh** (Blender or ZBrush) would look best, but
there is no DCC on this Mac and it can't be rebuilt from the repo; the generated
mesh can be swapped for one later without touching the rig or the game code.
**Hair cards or shells** instead of strands: cheaper, but they read as fake up
close. Strands render on Metal SM6, which the project targets; cards stay a
fallback if they prove too slow.

## Rig and animation

- The mesh keeps the mannequin's bone names and hierarchy, with the new bind
  pose, and gets its own skeleton `SK_Sasquatch`, so the mannequin stays
  untouched.
- IK Rigs for the mannequin and the sasquatch (auto-generated retarget chains)
  and an IK Retargeter between them. Its retarget pose bends the knees and
  pitches the spine forward for the compliant gait; feet use IK goals so
  strides scale with his legs.
- The retargeter batch-exports the template's unarmed idle, walk, jog, jump,
  fall and land animations and the idle-walk-run blend space to
  `/Game/Sasquatch/Anims`.
- In game the mesh plays the blend space directly (single-node animation,
  driven by speed from C++), and the fall loop in the air. No Animation
  Blueprint, so nothing needs hand-wiring in the editor.

## In the game

- `ARRExplorer` replaces the blocky figure with the sasquatch skeletal mesh and
  two grooms (head and body), under the same `Body` component, so the wingsuit
  and ride poses still tilt him.
- The capsule grows to his size (radius 42 cm, half-height 128 cm) and the eye
  to 2.28 m. Step height stays 52 cm, so the rows and aisle stairs still work.
  The climb probes scale with the capsule.
- **First person:** the camera sits at his eyes. The head bone and the head
  groom are hidden from your own view; the body, arms and hands stay, so looking
  down shows his chest, arms and hands, and they swing into view as he walks.
- **Third person:** V (already the chase-camera key) pulls the camera back about
  4.2 m behind and above his shoulder. Rides, wingsuit and parachute keep their
  own chase distances.

## Verification

- Tests on the generator: height in range, arm reach to the knee, normalised
  skin weights, a valid glTF, strand counts, lengths and colours per region,
  bare face, palms and soles.
- `unreal/render_sasquatch.py` renders front, side and three-quarter views in a
  plain daylight studio and on the stage at Red Rocks; `tools/` composes the
  side-by-side into `docs/media/sasquatch/`.
- A play-test in the editor: V toggles the view, idle, walk and run play, and
  the seating walk still climbs the aisle.

## Follow-ups

- Climb, mantle, wingsuit, parachute and ride animations (today he holds the
  locomotion pose while those move him).
- Groom physics (hair swaying) and LODs tuned on the real frame budget.
- The mushroom easter egg.

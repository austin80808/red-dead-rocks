# Controllers and the streaming spike: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove Pixel Streaming 2 works from the M4 to Austin's iPhone (the spike), then make every action
playable on Xbox, PlayStation and Switch pads, with hints that show the pad in use (rollout PR 1).

**Architecture:**
- **Input:** one C++ table builds every Enhanced Input action and the mapping context at runtime.
- **Device:** a Slate input processor tracks which device was last used.
- **Hints:** a token renderer turns `{Jump}`-style text into key caps or pad glyphs that the canvas HUD draws.
- **Pipeline side:** spot tips and the credits widget script are tokenized to match.

**Tech Stack:**
- Unreal Engine 5.8 C++ (module `RedRocks`), Enhanced Input, Slate `IInputProcessor`.
- Unreal automation tests, run headless.
- Python editor scripts and play-tests (`unreal/*.py`), pytest.
- Pixel Streaming 2, spike only.

**Spec:** `docs/superpowers/specs/2026-11-04-controller-and-phone-play-design.md`. Part 1 and "The spike" are
built here. Parts 2 and 3 get a second plan once the spike's findings are in.

## Global Constraints

- **Engine:** UE 5.8 at `/Users/Shared/Epic Games/UE_5.8`, on Mac only. Build with
  `Engine/Build/BatchFiles/Mac/Build.sh RedRocksEditor Mac Development -Project=<uproject> -WaitMutex`.
- **The Unreal lock:** every step that runs `UnrealEditor`, `UnrealEditor-Cmd`, `Build.sh`/UBT or shader compiles
  first runs `~/.local/bin/unreal-lock wait "Red Rocks" "<why>"`, then `unreal-lock release "Red Rocks"` once Unreal
  has quit. `tools/ue.sh` (Task 2) does this for you; use it.
- **Copies you may touch:**
  - Never build or test in `~/Documents/Unreal Projects/RedRocks` (Austin's open project).
  - Never touch `RedRocks-seating` (the seating session) or `RedRocks-agent`, which has 54 uncommitted changes that
    aren't ours.
  - Work in `~/Documents/Unreal Projects/RedRocks-controllers` (made in Task 2) and `RedRocks-spike` (Task 1).
  - Pipeline work goes in the worktree `~/dev/redrocks-ue-controller-phone`. Never switch branches in
    `~/dev/redrocks-ue`.
- **Keyboard and mouse stay exactly as they are**, including mouse-look feel.
- **No new third-party dependencies in PR 1.** Engine modules only.
- **Strings:** every key-naming string uses the tokens `{Move} {Look} {Run} {Jump} {Interact} {Drop} {Camera}
  {Respawn} {Spots} {Night} {Snow} {Help} {Credits}`. The new wording is copied verbatim from the spec's
  "Strings that change" table.
- **Stick look:** 220°/s yaw, 150°/s pitch, 0.2 radial dead zone, response exponent 2.0. The cvars are
  `rr.PadLookYaw`, `rr.PadLookPitch`, `rr.PadLookExponent` and `rr.PadInvertY` (default 0).
- **Device names on Mac:** `Xbox Wireless Controller` → Xbox; `PS4 Wireless Controller` and `PS5 Wireless Controller`
  → PlayStation; anything else → Position.
- **`RRExplorer.cpp` edits stay out of the constructor's movement block** (`MaxStepHeight` and the lines around it),
  because the terraced-seating branch may change it.
- **Merging:** merge only through `merge-gate <PR>`. Commits use Conventional Commits.

## Review Focus

1. **Keyboard and pad held together.** The player moves at full speed, not double, and runs if Shift is held or the
   run toggle is on. *Test in Task 4 (`RedRocks.Input.MoveCombines`) and Task 6 (play-test `pad_and_keys`).*
2. **A pad disconnects with the stick held.** The player stops and run turns off. *Test in Task 6 (play-test
   `pad_disconnect`).*
3. **A frame hitch while looking with the stick.** The view doesn't jump; look clamps the frame time at 0.1 s.
   *Test in Task 4 (`RedRocks.Input.StickLookClampsHitch`).*
4. **A level with fewer than 9 spots, or none.** The D-pad wraps over the spots that exist, and does nothing when
   there are none. *Test in Task 4 (`RedRocks.Input.CycleSpot`).*
5. **A pad connected at startup but not touched.** Hints show the keyboard until the pad is actually used. *Test in
   Task 5 (`RedRocks.Device.DefaultsToKeyboard`).*

---

### Task 1: Spike: Pixel Streaming 2 from the M4 to the iPhone (throwaway)

**Files:**
- Create: `~/Documents/Unreal Projects/RedRocks-spike/` (APFS clone, never committed)
- Create: `Source/RedRocks/RRSpikeEcho.h/.cpp` in the spike copy only
- Modify: the spec, adding an appendix "Spike results (2026-11-..)"
- Create: `<shared notes>/pixel-streaming-spike.md`, in `/Users/austin/Library/Application Support/Claude/side-session-notes/local_5c4e625b-ade3-47ed-a762-495e1a5fdf83/`

**Interfaces:**
- Produces, for plan 2:
  - the exact launch flags and Wilbur command
  - the fps cap (30 or 60) and any scalability steps
  - whether Epic's encoder path works on the M4, and if not, the fix
  - whether `InputController=Host` means the first peer
  - whether WebRTC takes the Tailscale route
  - whether the reflection binding to `UPixelStreaming2Input` works

- [ ] **Step 1: Ask Austin for the two downloads and wait for a yes:** Homebrew `node@22` (keg-only), and a
  `git clone --branch UE5.8` of `https://github.com/EpicGamesExt/PixelStreamingInfrastructure` into
  `~/.cache/redrocks-remote/ps-infra`. Record the commit hash.
- [ ] **Step 2: Make the spike copy.** Run `cp -cR "RedRocks" "RedRocks-spike"` from `~/Documents/Unreal Projects`
  (copy-on-write, instant). Add `{"Name": "PixelStreaming2", "Enabled": true}` to its `RedRocks.uproject`.
- [ ] **Step 3: Write `RRSpikeEcho`.** It's a `UActorComponent` that, in `BeginPlay`:
  - finds `UClass` `/Script/PixelStreaming2.PixelStreaming2Input` with `FindObject<UClass>` and adds an instance to
    its owner;
  - binds its `OnInputEvent` multicast delegate through `FMulticastScriptDelegate` to a `UFUNCTION()
    void OnInput(const FString& Descriptor)`, which logs `RRSpike in: <descriptor>`;
  - calls `SendPixelStreaming2Response` through `ProcessEvent` with `{"echo": <descriptor>}`.

  Spawn it from `ARRExplorer::BeginPlay` in the spike copy only.
- [ ] **Step 4: Build** with the Unreal lock: `unreal-lock wait "Red Rocks" "spike build"`, then `Build.sh` for the
  spike copy. Expected: `Result: Succeeded`.
- [ ] **Step 5: Start Wilbur** on Node 22, following the branch's `SignallingWebServer/README.md`, with the stock
  frontend: player port 80 and streamer port 8888, on localhost. Record the exact command.
- [ ] **Step 6: Launch the game.** Hold the lock and run `UnrealEditor RedRocks.uproject /Game/RedRocks/Lvl_RedRocks
  -game -windowed -ResX=1280 -ResY=720` plus the PS2 connection flag for `ws://127.0.0.1:8888`, H.264,
  `-PixelStreaming2.InputController=Host` (or the spelling the plugin settings use) and `-ExecCmds="t.MaxFPS 30"`.
  Record the flags that worked.
- [ ] **Step 7: Check in Safari on the Mac** at `http://127.0.0.1`.
  - **Pass:** live video, and the mouse turns the view.
  - **If it asserts in `FVideoProducerBackBuffer::OnBackBufferReady`, or the video is dark:** read
    `pixelstreaming2-native-mac`'s source on GitHub (don't clone it, don't run it). Write the same fix into the spike
    copy: copy the back buffer into a shader-readable texture before handing it to the encoder. Rebuild and retry.
    If it still fails, stop and report to Austin.
- [ ] **Step 8: Try the iPhone.**
  - The CLI is `/Applications/Tailscale.app/Contents/MacOS/Tailscale`. If serve says HTTPS isn't enabled for the
    tailnet, ask Austin to turn on HTTPS certificates in the Tailscale admin console.
  - Run `Tailscale serve status` and save its output. Then `Tailscale serve --bg --https=443 http://127.0.0.1:80`.
  - Ask Austin to open `https://austins-macbook-air.tailb31dd2.ts.net` on his iPhone.
  - Check the selected ICE candidate pair in the frontend's stats panel. It should be a 100.x address.
  - If a pad is paired to the phone, check that the stock frontend's input shows it.
  - Afterwards, restore the saved serve config.
- [ ] **Step 9: Round trip.** In the page's console, run `emitUIInteraction("ping")`.
  - **Pass:** the game log shows `RRSpike in: ping`, and the page console shows `{"echo":"ping"}`.
  - With two browsers connected, input from the second is ignored, which confirms Host semantics.
- [ ] **Step 10: Measure hot.** Send console commands `RRGo 2` and `CsvProfile Start` through `-ExecCmds`, leave it
  streaming to the phone for 20 minutes, then `CsvProfile Stop`. Compute mean fps and the 5th percentile, per
  minute, from the CSV's `FrameTime` column.
  - **Under 30 fps at the 5th percentile:** retry with `r.ScreenPercentage 75`, then add `sg.FoliageQuality 2`.
  - **60 sustained:** note it.
- [ ] **Step 11: Release and record.** Release the lock. Write "Spike results" into the spec as an appendix. Cover
  the flags, the Wilbur command, the PSInfra commit, the encoder result and fix, the Host check, the ICE route, the
  fps table and the reflection binding result. Copy it to the shared notes file.
- [ ] **Step 12: Commit and open PR 0.** Commit the spec appendix (`docs: Pixel Streaming spike results`), push
  `docs/controller-phone-spec`, and open the docs PR in red-dead-rocks containing the spec, this plan and the
  appendix. Merge with `merge-gate` (🟢 docs). Delete `RedRocks-spike` once Austin has seen the results.

### Task 2: Work copy, `tools/ue.sh` and a first automation test

**Files:**
- Create: `~/Documents/Unreal Projects/RedRocks-controllers/` (APFS clone of `RedRocks`, branch `feat/controllers`)
- Create: `tools/ue.sh` (red-dead-rocks)
- Create: `Source/RedRocks/Tests/RRSmokeTest.cpp` (red-dead-rocks-unreal)
- Modify: `Source/RedRocks/RedRocks.Build.cs`, adding `"ApplicationCore"` to `PrivateDependencyModuleNames`

**Interfaces:**
- Produces:
  - `tools/ue.sh build <project-dir>`
  - `tools/ue.sh cpp-tests <project-dir> [filter=RedRocks.]`: exit 0 when every test passes, and prints the
    failures otherwise
  - `tools/ue.sh playtest <project-dir> [SCENARIOS=...]`: runs `unreal/playtest_explore.py` from this worktree and
    tails its log
  - `tools/ue.sh rebuild <project-dir>`: runs `unreal/rebuild_level.py`
- Each subcommand runs `unreal-lock wait "Red Rocks" "<subcommand> <project>"` with a `trap` that releases on exit.
  While an `UnrealEditor` process for a *different* `.uproject` is running, it waits, polling every 60 s, and says
  so. `cpp-tests` runs `UnrealEditor-Cmd <uproject> -ExecCmds="Automation RunTests <filter>;Quit" -unattended
  -nopause -nosplash -nullrhi -NoSound -ReportExportPath=/tmp/redrocks_cpp_tests` and reads `index.json` there.

- [ ] **Step 1: Make the work copy.** Run `cp -cR RedRocks RedRocks-controllers`, then `git -C RedRocks-controllers
  switch -c feat/controllers` (on top of `origin/main`, after `git fetch`). Confirm `git status` is clean.
- [ ] **Step 2: Write the failing smoke test** `RedRocks.Smoke.ModuleLoads`:

```cpp
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRRSmokeTest, "RedRocks.Smoke.ModuleLoads",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FRRSmokeTest::RunTest(const FString&)
{
	TestNotNull(TEXT("RRExplorer class"), ARRExplorer::StaticClass());
	TestTrue(TEXT("fails until ue.sh reads reports"), false);
	return true;
}
```
- [ ] **Step 3: Write `tools/ue.sh`** as described in Interfaces. Run `tools/ue.sh build <copy>`, then
  `tools/ue.sh cpp-tests <copy>`. Expected: build `Succeeded`, then cpp-tests exits 1 and prints
  `RedRocks.Smoke.ModuleLoads ... fails until ue.sh reads reports`.
- [ ] **Step 4: Remove the `false` assertion** and rerun `tools/ue.sh cpp-tests <copy>`. Expected: exit 0,
  `1 passed`.
- [ ] **Step 5: Commit** in both repos: `test: headless automation tests and the ue.sh wrapper`.

### Task 3: Tokens and hint rendering (`RRHints`)

**Files:**
- Create: `Source/RedRocks/RRInputTypes.h`
- Create: `Source/RedRocks/RRHints.h/.cpp`
- Test: `Source/RedRocks/Tests/RRHintsTest.cpp`

**Interfaces:**
- Produces:
  - `UENUM(BlueprintType) enum class ERRInputDevice : uint8 { KeyboardMouse, Xbox, PlayStation, Position, Touch };`
  - `USTRUCT(BlueprintType) struct FRRContextLabels { FString Primary; FString Secondary; };` (both
    `UPROPERTY(BlueprintReadOnly)`)
  - `enum class ERRChip : uint8 { None, KeyCap, Disc, PsCross, PsCircle, PsSquare, PsTriangle, Diamond, Stick, DPad, Pill };`
  - `struct FRRHintRun { FString Text; ERRChip Chip = ERRChip::None; FString Label; FLinearColor Color; int32 Slot = -1; };`
    `Slot` is the diamond position (0 bottom, 1 right, 2 left, 3 top) or the D-pad arm (0 down, 1 right, 2 left,
    3 up, 4 left+right).
  - `namespace RRHints { TArray<FRRHintRun> Parse(const FString& Text, ERRInputDevice Device, const FRRContextLabels& Labels); FString ToPlain(const FString& Text, ERRInputDevice Device, const FRRContextLabels& Labels); const TArray<FName>& KnownTokens(); }`
- `ToPlain` writes each chip as `[Label]` and leaves plain text as it is. An unknown token stays literal
  (`{Bogus}`) and logs a warning once per token.

**Labels** (a table in `RRHints.cpp`). Touch uses the context label in capitals for Jump, Interact and Drop, with
the fallback after "or" when the label is empty:

| Token | KeyboardMouse | Xbox | PlayStation | Position | Touch |
|---|---|---|---|---|---|
| Move | WASD | L stick | L stick | L stick | left stick |
| Look | MOUSE | R stick | R stick | R stick | drag |
| Run | SHIFT | L3 | L3 | L3 | RUN |
| Jump | SPACE | A | Cross | Bottom | Primary or JUMP |
| Interact | E | X | Square | Left | Secondary or USE |
| Drop | Q | B | Circle | Right | Secondary or LET GO |
| Camera | V | Y | Triangle | Top | Menu: Camera |
| Respawn | R | RB | R1 | R | Menu: Respawn |
| Spots | 1-9 | D-pad left/right | D-pad left/right | D-pad left/right | Menu: Spots |
| Night | T | D-pad up | D-pad up | D-pad up | Menu: Night |
| Snow | N | D-pad down | D-pad down | D-pad down | Menu: Snow |
| Help | H | View | Create | - | Menu: Help |
| Credits | C | Menu | Options | + | Menu: Credits |

Touch writes "Menu:" rather than ☰ because the HUD's engine font may not have that glyph. The spec's "☰ Snow" means
this.

**Chip colours:**
- **Xbox discs:** A `(0.30, 0.69, 0.31)`, B `(0.90, 0.22, 0.21)`, X `(0.13, 0.59, 0.95)`, Y `(1.00, 0.76, 0.03)`.
- **PlayStation:** dark discs `(0.12, 0.12, 0.14)` with the shape in Cross `(0.49, 0.70, 0.91)`, Circle
  `(0.94, 0.40, 0.40)`, Square `(0.85, 0.55, 0.80)` and Triangle `(0.25, 0.75, 0.65)`.
- **Everything else:** dark pills or key caps.

- [ ] **Step 1: Write the failing tests** `RedRocks.Hints.ToPlain` and `RedRocks.Hints.Runs`:

```cpp
const FRRContextLabels None;
const FRRContextLabels Climb{TEXT("Leap off"), TEXT("Let go")};
TestEqual(TEXT("kb"), RRHints::ToPlain(TEXT("{Jump} leap off   {Drop} let go"), ERRInputDevice::KeyboardMouse, None), TEXT("[SPACE] leap off   [Q] let go"));
TestEqual(TEXT("xbox"), RRHints::ToPlain(TEXT("{Jump} leap off"), ERRInputDevice::Xbox, None), TEXT("[A] leap off"));
TestEqual(TEXT("ps"), RRHints::ToPlain(TEXT("{Drop} let go"), ERRInputDevice::PlayStation, None), TEXT("[Circle] let go"));
TestEqual(TEXT("switch"), RRHints::ToPlain(TEXT("{Jump}"), ERRInputDevice::Position, None), TEXT("[Bottom]"));
TestEqual(TEXT("touch ctx"), RRHints::ToPlain(TEXT("{Jump} leap off"), ERRInputDevice::Touch, Climb), TEXT("[LEAP OFF] leap off"));
TestEqual(TEXT("touch empty"), RRHints::ToPlain(TEXT("{Jump}"), ERRInputDevice::Touch, None), TEXT("[JUMP]"));
TestEqual(TEXT("touch menu"), RRHints::ToPlain(TEXT("{Snow} makes it snow."), ERRInputDevice::Touch, None), TEXT("[Menu: Snow] makes it snow."));
TestEqual(TEXT("unknown"), RRHints::ToPlain(TEXT("{Bogus} x"), ERRInputDevice::Xbox, None), TEXT("{Bogus} x"));
TestEqual(TEXT("no tokens"), RRHints::ToPlain(TEXT("Landed it."), ERRInputDevice::Xbox, None), TEXT("Landed it."));
TestEqual(TEXT("13 tokens"), RRHints::KnownTokens().Num(), 13);
// Runs: Xbox {Jump} is a green disc; PlayStation {Interact} a square; Position {Drop} a diamond at slot 1; Xbox {Spots} a D-pad with slot 4.
```
  `RedRocks.Hints.Runs` asserts those four runs' `Chip`, `Color` and `Slot`, plus the text runs on either side of a
  token.
- [ ] **Step 2: Run** `tools/ue.sh cpp-tests <copy> RedRocks.Hints`. Expected: build fails, because `RRHints.h`
  doesn't exist yet.
- [ ] **Step 3: Implement `RRInputTypes.h` and `RRHints.h/.cpp`** to the interfaces and the label table.
- [ ] **Step 4: Run** `tools/ue.sh cpp-tests <copy> RedRocks.Hints`. Expected: 2 passed.
- [ ] **Step 5: Commit:** `feat(input): hint tokens rendered per device`.

### Task 4: The action table, look curve, run toggle and spot cycling (`RRInput`)

**Files:**
- Create: `Source/RedRocks/RRInput.h/.cpp`
- Create: `tests/data/input_table.json` (red-dead-rocks)
- Test: `Source/RedRocks/Tests/RRInputTest.cpp`; `tests/test_input_table.py` (red-dead-rocks)

**Interfaces:**
- Consumes: `ERRInputDevice` (Task 3).
- Produces:
  - `struct FRRBinding { FName Action; EInputActionValueType Type; TArray<FKey> Keys; TArray<FKey> Pad; };`
  - `namespace RRInput`:
    - `const TArray<FRRBinding>& Table();`
    - `UInputMappingContext* BuildContext(UObject* Outer, TMap<FName, TObjectPtr<UInputAction>>& OutActions);`
    - `FString DescribeJson(const UInputMappingContext* Context);`
    - `FVector2D StickLookDegrees(FVector2D Stick, float DeltaSeconds);` returns (yaw°, pitch°), positive pitch is
      look up
    - `int32 CycleSpot(int32 Current, int32 Delta, int32 Count);`
  - `class FRRRunToggle { public: void Press(); bool Update(FVector2D Move, bool bAirMode, float Dt); bool IsOn() const; void Reset(); };`
- **Action names:** `Move` (Axis2D), `Look` (mouse, Axis2D), `LookStick` (Axis2D), `Run` (held), `RunToggle`,
  `Jump`, `Interact`, `Drop`, `Camera`, `Respawn`, `SpotNext`, `SpotPrev`, `Spot1`…`Spot9`, `Night`, `Snow`, `Help`,
  `Credits`.
- **Keys:**
  - The table in spec Part 1, with `SpotN` on the number keys `One`…`Nine`.
  - `Run` on `LeftShift` and `RightShift`; `RunToggle` on `Gamepad_LeftThumbstick`.
  - `Help` on `H` and `Gamepad_Special_Left`; `Credits` on `C` and `Gamepad_Special_Right`.
  - `Move` on `W A S D` (template modifiers: W swizzle YXZ; S swizzle YXZ plus negate; A negate) and
    `Gamepad_Left2D`, with a radial `UInputModifierDeadZone` (lower threshold 0.25).
  - `LookStick` on `Gamepad_Right2D`, raw (the dead zone lives in `StickLookDegrees`).
  - `Look` on `Mouse2D`.
- **`Credits` binds the existing asset** `/Game/RedRocks/UI/IA_Credits` (loaded with `LoadObject`) instead of a new
  action. If it's missing, log a warning and skip the row.
- **Look curve:**
  - The cvars `rr.PadLookYaw` (220), `rr.PadLookPitch` (150), `rr.PadLookExponent` (2.0) and `rr.PadInvertY` (0)
    are declared in `RRInput.cpp`.
  - Clamp `DeltaSeconds` to 0.1.
  - With m = |stick|: if m < 0.2, return zero. Otherwise scale = ((m − 0.2) / 0.8)^exponent, direction =
    stick / m.
  - Return (dir.X × scale × yaw × dt, dir.Y × scale × pitch × dt × (invertY ? −1 : 1)).
- **`FRRRunToggle`:**
  - `Press` flips it on or off.
  - `Update` turns it off once `Move` has been under 0.2 for 0.5 s in a row, or as soon as `bAirMode` is true.
  - It returns `IsOn()`.

- [ ] **Step 1: Read the template's mouse-look settings.** In the work copy, run an editor Python snippet with
  `tools/ue.sh` and the lock that prints every mapping's key and modifier classes and properties from
  `/Game/Input/IMC_MouseLook` and `/Game/Input/IMC_Default`, plus `IA_MouseLook`'s and `IA_Move`'s own modifiers.
  Copy the `Mouse2D` and WASD modifiers into the table exactly as printed.
- [ ] **Step 2: Write `tests/data/input_table.json` and its failing pytest.** The JSON is a list of
  `{"action", "keys": [...], "pad": [...]}` objects in the order above, with FKey names as UE spells them
  (`SpaceBar`, `Gamepad_FaceButton_Bottom`, `Gamepad_DPad_Right`, …). `tests/test_input_table.py` asserts:
  - every spec token has a row;
  - `Jump` has `SpaceBar` and `Gamepad_FaceButton_Bottom`, `Drop` has `Q` and `Gamepad_FaceButton_Right`, and
    `Interact` has `E` and `Gamepad_FaceButton_Left`;
  - no pad key appears twice, and no keyboard key appears twice;
  - the Home button (`Gamepad_Special_*` beyond Left/Right) never appears.

  Run `.venv/bin/python -m pytest tests/test_input_table.py -q`. Expected: FAIL (file missing). Then write the
  JSON. Expected: PASS.
- [ ] **Step 3: Write the failing C++ tests:**

```cpp
// RedRocks.Input.StickLook
FVector2D Full = RRInput::StickLookDegrees(FVector2D(1, 0), 1.0f);  // dt clamps to 0.1
TestEqual(TEXT("full yaw"), Full.X, 22.f, 0.01f);
TestEqual(TEXT("dead zone"), RRInput::StickLookDegrees(FVector2D(0.15f, 0), 0.016f), FVector2D::ZeroVector);
TestEqual(TEXT("curve"), RRInput::StickLookDegrees(FVector2D(0.6f, 0), 0.1f).X, 5.5f, 0.01f); // 220*0.25*0.1
TestEqual(TEXT("pitch up"), RRInput::StickLookDegrees(FVector2D(0, 1), 0.1f).Y, 15.f, 0.01f);
// RedRocks.Input.StickLookClampsHitch
TestEqual(TEXT("hitch"), RRInput::StickLookDegrees(FVector2D(1, 0), 0.5f).X, 22.f, 0.01f);
// RedRocks.Input.CycleSpot
TestEqual(TEXT("next"), RRInput::CycleSpot(3, +1, 9), 4);   TestEqual(TEXT("wrap up"), RRInput::CycleSpot(9, +1, 9), 1);
TestEqual(TEXT("wrap down"), RRInput::CycleSpot(1, -1, 9), 9); TestEqual(TEXT("unset next"), RRInput::CycleSpot(0, +1, 9), 1);
TestEqual(TEXT("unset prev"), RRInput::CycleSpot(0, -1, 5), 5); TestEqual(TEXT("none"), RRInput::CycleSpot(2, +1, 0), 0);
// RedRocks.Input.RunToggle
FRRRunToggle T; T.Press(); TestTrue(TEXT("on"), T.Update(FVector2D(0, 1), false, 0.1f));
TestTrue(TEXT("brief stop"), T.Update(FVector2D::ZeroVector, false, 0.3f));
TestFalse(TEXT("stopped"), T.Update(FVector2D::ZeroVector, false, 0.3f));
T.Press(); TestFalse(TEXT("air"), T.Update(FVector2D(0, 1), true, 0.016f));
// RedRocks.Input.Context: BuildContext maps Jump to SpaceBar and Gamepad_FaceButton_Bottom, Credits to IA_Credits, and every Table() row has its keys.
// RedRocks.Input.MoveCombines: the Move action's accumulation behaviour is TakeHighestAbsoluteValue.
```
- [ ] **Step 4: Run** `tools/ue.sh cpp-tests <copy> RedRocks.Input`. Expected: build fails, because `RRInput.h`
  doesn't exist yet.
- [ ] **Step 5: Implement `RRInput.h/.cpp`** to the interfaces. `DescribeJson` writes
  `[{"action": ..., "keys": [...]}]`, with keyboard and pad keys together in one sorted list.
- [ ] **Step 6: Run** `tools/ue.sh cpp-tests <copy> RedRocks.`. Expected: all passed.
- [ ] **Step 7: Commit** in both repos: `feat(input): action table, stick look curve, run toggle`.

### Task 5: Which device is in use (`RRInputDevice`)

**Files:**
- Create: `Source/RedRocks/RRInputDevice.h/.cpp`
- Test: `Source/RedRocks/Tests/RRInputDeviceTest.cpp`

**Interfaces:**
- Consumes: `ERRInputDevice` (Task 3).
- Produces:
  - `class FRRInputDeviceTracker`, with these members:
    - `static ERRInputDevice FamilyFromDeviceName(FName Name);`
    - `void OnKey(const FKey& Key, FName DeviceName);`
    - `void OnAnalog(const FKey& Key, float Value, FName DeviceName);`
    - `void OnMouseMove(FVector2D Delta);`
    - `void SetRemote(ERRInputDevice Device);`
    - `ERRInputDevice Get() const;`
  - `class FRRInputDeviceProcessor : public IInputProcessor`, which forwards `HandleKeyDownEvent`,
    `HandleAnalogInputEvent` and `HandleMouseMoveEvent` to the tracker. The device name comes from
    `FInputDeviceScope::GetCurrent()`, and it never consumes an event.
  - `UCLASS() class URRInputDeviceSubsystem : public UGameInstanceSubsystem`, with these members:
    - `static URRInputDeviceSubsystem* Get(const UObject* WorldContext);`
    - `ERRInputDevice GetDevice() const;`
    - `void SetRemoteDevice(ERRInputDevice Device);`
    - `FRRInputDeviceTracker& Tracker();`

    It registers the processor with `FSlateApplication` in `Initialize`, only if it's initialized, and unregisters
    it in `Deinitialize`.
- **Rules:**
  - The tracker starts at `KeyboardMouse`.
  - A gamepad key, or an analog value with |v| ≥ 0.25, sets the family.
  - A non-gamepad key, or a mouse move over 3 px, sets `KeyboardMouse`.
  - `SetRemote` holds until the next local event.

- [ ] **Step 1: Write the failing tests:**

```cpp
// RedRocks.Device.Family
TestEqual(TEXT("xbox"), FRRInputDeviceTracker::FamilyFromDeviceName(TEXT("Xbox Wireless Controller")), ERRInputDevice::Xbox);
TestEqual(TEXT("ps4"), FRRInputDeviceTracker::FamilyFromDeviceName(TEXT("PS4 Wireless Controller")), ERRInputDevice::PlayStation);
TestEqual(TEXT("ps5"), FRRInputDeviceTracker::FamilyFromDeviceName(TEXT("PS5 Wireless Controller")), ERRInputDevice::PlayStation);
TestEqual(TEXT("switch"), FRRInputDeviceTracker::FamilyFromDeviceName(TEXT("ExtendedGamepad")), ERRInputDevice::Position);
TestEqual(TEXT("none"), FRRInputDeviceTracker::FamilyFromDeviceName(NAME_None), ERRInputDevice::Position);
// RedRocks.Device.DefaultsToKeyboard
FRRInputDeviceTracker T; TestEqual(TEXT("start"), T.Get(), ERRInputDevice::KeyboardMouse);
// RedRocks.Device.Switching
T.OnAnalog(EKeys::Gamepad_LeftX, 0.1f, TEXT("PS5 Wireless Controller")); TestEqual(TEXT("drift ignored"), T.Get(), ERRInputDevice::KeyboardMouse);
T.OnAnalog(EKeys::Gamepad_LeftX, 0.5f, TEXT("PS5 Wireless Controller")); TestEqual(TEXT("stick"), T.Get(), ERRInputDevice::PlayStation);
T.OnMouseMove(FVector2D(2, 0)); TestEqual(TEXT("jiggle ignored"), T.Get(), ERRInputDevice::PlayStation);
T.OnKey(EKeys::W, NAME_None); TestEqual(TEXT("key"), T.Get(), ERRInputDevice::KeyboardMouse);
T.OnKey(EKeys::Gamepad_FaceButton_Bottom, TEXT("Xbox Wireless Controller")); TestEqual(TEXT("button"), T.Get(), ERRInputDevice::Xbox);
T.SetRemote(ERRInputDevice::Touch); TestEqual(TEXT("remote"), T.Get(), ERRInputDevice::Touch);
T.OnMouseMove(FVector2D(10, 0)); TestEqual(TEXT("local wins"), T.Get(), ERRInputDevice::KeyboardMouse);
```
- [ ] **Step 2: Run** `tools/ue.sh cpp-tests <copy> RedRocks.Device`. Expected: build fails, because the header
  doesn't exist yet.
- [ ] **Step 3: Implement `RRInputDevice.h/.cpp`** to the interfaces.
- [ ] **Step 4: Run** `tools/ue.sh cpp-tests <copy> RedRocks.`. Expected: all passed.
- [ ] **Step 5: Commit:** `feat(input): track the device in use`.

### Task 6: Wire it into the game, with pad play-tests

**Files:**
- Create: `Source/RedRocks/RRPlayerController.h/.cpp`
- Create: `Source/RedRocks/RRInputLibrary.h/.cpp`
- Modify: `Source/RedRocks/RRGameMode.cpp` (the controller class, dropping the `FClassFinder`)
- Modify: `Source/RedRocks/RRExplorer.h`: the input members (lines 96–101), the new members and getters below,
  and `PollKeys` removed
- Modify: `Source/RedRocks/RRExplorer.cpp`: the input finders (126–137), `PawnClientRestart`,
  `SetupPlayerInputComponent`, `Tick`'s `PollKeys` call, `PollKeys` itself (delete), and `HandleAction`
- Test: `unreal/playtest_explore.py` (red-dead-rocks), with new scenarios `pad`, `pad_and_keys` and
  `pad_disconnect`

**Interfaces:**
- Consumes: `RRInput::BuildContext`, `RRInput::StickLookDegrees`, `RRInput::CycleSpot`, `FRRRunToggle` (Task 4);
  `URRInputDeviceSubsystem` (Task 5); `FRRContextLabels` (Task 3).
- Produces on `ARRExplorer`, all `UFUNCTION(BlueprintPure/Callable)` except where marked:
  - `FName GetLastAction() const`
  - `int32 GetLastSpot() const`
  - `bool IsCreditsOpen() const`
  - `FRRContextLabels GetContextLabels() const`
  - `void AddLookDegrees(float YawDeg, float PitchDeg)`: adds to the control rotation directly, clamping pitch to
    the camera manager's `ViewPitchMin/Max`, so the legacy input scales don't apply
  - `const UInputMappingContext* GetInputContext() const` (C++ only)
  - the existing `GetMoveInput()` and `IsSprinting()` become `UFUNCTION(BlueprintPure)` so play-tests can read them
- Produces on `URRInputLibrary`, all static `UFUNCTION(BlueprintCallable)`:
  - `FString DescribeInput(const UObject* WorldContext)`
  - `void SimulateKey(const UObject* WorldContext, FName KeyName, bool bPressed, const FString& DeviceName)`
  - `void SimulateAxis(const UObject* WorldContext, FName KeyName, float Value, const FString& DeviceName)`
  - `void SimulatePadDisconnect(const UObject* WorldContext)`
  - `FString RenderHint(const UObject* WorldContext, const FString& Text)`: `ToPlain` for the current device and labels
  - `ERRInputDevice GetInputDevice(const UObject* WorldContext)`
- `SimulateKey` and `SimulateAxis` feed `PC->InputKey(FInputKeyParams(...))` with `bIsGamepad` set from the key, and
  report the same key and device name to the tracker.
- `SimulatePadDisconnect` broadcasts
  `IPlatformInputDeviceMapper::Get().GetOnInputDeviceConnectionChange()` with `Disconnected` for the default device.
- **Context labels:**
  - Primary: "Wingsuit" when falling more than `WingsuitMinHeight` up in Walk, otherwise "Jump"; Climb "Leap off";
    Wingsuit "Chute"; Ride "Hop"; Mantle and Parachute "".
  - Secondary: Walk near a ride "Ride"; Ride "Get off"; Climb "Let go"; Parachute "Cut away"; otherwise "".
- **Behaviour:**
  - `bSprint = bRunHeld || RunToggle.IsOn()` when not scripted.
  - `HandleAction` records `LastAction` and handles `SpotNext`/`SpotPrev` (`RRGo(CycleSpot(LastSpot, ±1,
    Spots.Num()))`), `Spot1`–`Spot9` and `RunToggle`.
  - The `Credits` action flips `bCreditsOpen` and does nothing else; `BP_CreditsMenu` still opens the card.
  - On a device-disconnect event, zero `MoveInput` and reset the run toggle.

- [ ] **Step 1: Look at the template controller.** Hold the lock and open `BP_FirstPersonPlayerController` in the
  work copy with an editor Python snippet that lists its graph nodes and its class defaults. Confirm it only adds
  mapping contexts and the mobile touch UI. If it does anything else, stop and note it in the PR.
- [ ] **Step 2: Write the failing play-test scenarios** in `playtest_explore.py`, added to `ALL` after `bowl`:
  - **`pad`:** `rr_go(1)`, then for each row of `tests/data/input_table.json` whose pad key is a button and whose
    action isn't `Credits` or `RunToggle`: `simulate_key(row pad key, True, "PS5 Wireless Controller")`; next tick,
    `simulate_key(..., False, ...)`; then `check("pad %s" % action, pawn.get_last_action() == action)`.
    Toggle actions are pressed twice to leave the world as it was.
    - Then `check("device ps", get_input_device() == PlayStation)` and
      `check("pad hint", "[Cross]" in render_hint("{Jump} jump"))`.
    - Then `rr_go(1)`, D-pad right → `get_last_spot() == 2`, D-pad left twice → `get_last_spot() == count`.
    - Then `check("controller", pc class name == "RRPlayerController")` and
      `check("no IMC_Default", not subsystem.has_mapping_context(IMC_Default))`.
    - Then `check("table", describe_input() rows == input_table.json rows as sets)`.
  - **`pad_and_keys`:** hold `W` (keyboard) and `Gamepad_LeftY` 1.0 together. Check that horizontal speed stays
    within 5% of walking speed after 1 s. Press L3 with Shift up and check `is_sprinting()`. Release the stick for
    0.7 s and check it's not sprinting.
  - **`pad_disconnect`:** with `Gamepad_LeftY` 1.0 and L3 on, call `simulate_pad_disconnect()`. Next tick, check
    `get_move_input() == (0, 0)` and not sprinting.

  Every `RRInputLibrary` call takes the pawn as its world context.

  Run `tools/ue.sh playtest <copy> SCENARIOS=pad,pad_and_keys,pad_disconnect` with the old C++. Expected: `ERROR`
  or `FAIL`, because `simulate_key` doesn't exist yet.
- [ ] **Step 3: Implement `ARRPlayerController`** (empty subclass), the `RRGameMode` change, `URRInputLibrary` and
  the `RRExplorer` changes, to the interfaces.
- [ ] **Step 4: Build and run** `tools/ue.sh cpp-tests <copy>`, then `tools/ue.sh playtest <copy>` (all scenarios,
  including the existing climb, wingsuit, cart, sled, tubing, night and morrison). Expected: no `FAIL` or `ERROR`
  lines in `/tmp/redrocks_playtest_explore.log`.
- [ ] **Step 5: Commit** in both repos: `feat(input): play every action on a gamepad`.

### Task 7: Pad glyphs in the HUD, token strings and the credits pill

**Files:**
- Modify: `Source/RedRocks/RRHUD.h/.cpp`: draw `FRRHintRun`s, the help panel with tokens, the credits pill
  "{Credits} Credits" at bottom-left, and the highlighted current spot in GO TO
- Modify: `Source/RedRocks/RRExplorer.cpp`: the toast and hint literals in `BeginPlay`, `TryStartClimb`,
  `StartWingsuit`, `StartParachute`, `Crash`, `RRNight` and `UpdateHint`
- Modify: `Source/RedRocks/RRRide.cpp`: the board message and `GetRideHint` literals
- Test: `unreal/playtest_explore.py`, with new scenario `hints`

**Interfaces:**
- Consumes: `RRHints::Parse` (Task 3), `URRInputDeviceSubsystem::GetDevice` (Task 5), `GetContextLabels` and
  `GetLastSpot` (Task 6).
- Produces: `void ARRHUD::DrawHintText(const FString& S, float X, float Y, const FLinearColor& Color, float Scale,
  bool bCenter)`, which replaces `Text` for hints, toasts, help and the pill.
- **Chip drawing:**
  - The chip height is the line height × 1.05.
  - Discs, PlayStation shapes and the diamond's dots use `FCanvasNGonItem` (24 sides). Cross, triangle and D-pad
    arms are drawn with `FCanvasLineItem`. Key caps and pills use `FCanvasTileItem` with the label centred.
  - Leave 0.3 em of space after each chip.

- [ ] **Step 1: Write the failing `hints` scenario:**
  - For the climb, wingsuit, parachute and cart states (reached the same way as the existing scenarios), and for
    each device set by `simulate_key` (keyboard `W`; pad A with each of the three device names):
    `text = render_hint(pawn.get_hint())`. Then `check("no raw tokens", "{" not in text)`, and
    `check("no bare keys", not re.search(r"\b(SPACE|WASD|SHIFT|MOUSE)\b", text))` when the device is a pad.
  - Also render the latest `get_toast()` after climb start and after a crash, the same way.
  - Take one `_shot` per device in the climb state, named `redrocks_hints_<device>.png`.
- [ ] **Step 2: Run** `tools/ue.sh playtest <copy> SCENARIOS=hints`. Expected: FAIL on "no bare keys", because
  the strings still name keys.
- [ ] **Step 3: Replace every string** in the spec's "Strings that change" table, verbatim. Implement
  `DrawHintText` and the chips, the help panel, the pill, and the GO TO highlight.
- [ ] **Step 4: Run** `tools/ue.sh playtest <copy> SCENARIOS=hints,climb,wingsuit`. Expected: all `ok`. Open the
  four `redrocks_hints_*.png` files: Xbox shows a green A disc and a blue X; PlayStation shows a blue cross and a
  pink square; Switch shows a diamond with the bottom dot filled; keyboard shows SPACE and Q caps.
- [ ] **Step 5: Commit** in both repos: `feat(hud): pad glyphs and device-aware hints`.

### Task 8: Pipeline: tokenized tips, the credits widget, a rebuild and a full play-test

**Files:**
- Modify: `src/redrocks/explore.py:321-372`, the spot tips
- Modify: `unreal/setup_credits.py`:
  - `_hint_page` builds an empty page 0
  - `_map_key` and its `_step` call in `main()` are removed
  - The footer at lines 374–381 becomes "Press" + cap "C" + "or the pad's Menu button to close"
- Modify: `tests/test_setup_credits.py`, `tests/test_explore.py`
- Create: `tests/test_hint_tokens.py`
- Modify: `README.md`, adding a "Play with a controller" section with the pad table from the spec and the glyph
  screenshots copied into `docs/media/controllers/`

**Interfaces:**
- Consumes: the token list `RRHints::KnownTokens()` (Task 3), mirrored as `KNOWN_TOKENS` in
  `tests/test_hint_tokens.py`; the rows of `tests/data/input_table.json` (Task 4).

- [ ] **Step 1: Write the failing pytests.**
  - **`tests/test_hint_tokens.py`:**
    - `test_tips_use_known_tokens`: for every `Spot` that `explore` builds in the existing `test_explore.py`
      fixture, every `{X}` is in `KNOWN_TOKENS`.
    - `test_tips_name_no_keys`: no tip matches
      `r"\b(Press [A-Z]|Space|Shift|E to|N for)\b"`.
    - `test_unreal_strings_name_no_keys`: skipped unless
      `$REDROCKS_UNREAL_SOURCE`/`RRExplorer.cpp` exists. It finds every `TEXT("...")` literal in `RRExplorer.cpp`,
      `RRRide.cpp` and `RRHUD.cpp` that contains a space, and checks none matches `r"\b(SPACE|WASD|SHIFT|MOUSE|A/D)\b"`
      or `r"\b[EQRNTHVC]  "`.
  - **In `test_setup_credits.py`:** `test_no_key_mapping` (the source has no `unmap_all_keys_from_action`), and
    `test_footer_mentions_pad` ("Menu button" appears in the footer text).

  Run `.venv/bin/python -m pytest tests -q`. Expected: the new tests FAIL.
- [ ] **Step 2: Rewrite the tips** to the spec's table and make the `setup_credits.py` changes. Run
  `.venv/bin/python -m pytest tests -q` with `REDROCKS_UNREAL_SOURCE=~/Documents/Unreal Projects/RedRocks-controllers/Source/RedRocks`.
  Expected: all pass.
- [ ] **Step 3: Rebuild the level in the work copy.** Run `redrocks build`, then `tools/ue.sh rebuild <copy>`.
  Expected: `/tmp/redrocks_rebuild.log` has no `FAIL`. Then `tools/ue.sh playtest <copy>` with every scenario.
  Expected: no `FAIL` or `ERROR`, and the carts toast on arrival reads with a pad glyph when the last device was a
  pad.
- [ ] **Step 4: Commit:** `feat(explore): token spot tips; credits pill moves to the HUD` in red-dead-rocks, and
  `Rebuild: token spot tips, credits pill in the HUD` (the level and the `WBP_Credits` and `IA_Credits` assets) in
  the work copy.

### Task 9: Ship PR 1

**Files:** none new.

- [ ] **Step 1: Check against the seating branch.** Run `git -C RedRocks-controllers fetch origin`. If
  `feat/terraced-seating` has merged into `origin/main`, rebase `feat/controllers` onto it and rerun
  `tools/ue.sh build`, `cpp-tests` and `playtest`. Expected: all green.
- [ ] **Step 2: Open the two PRs.**
  - red-dead-rocks-unreal: `feat/controllers`, the C++ and the rebuild commit, titled
    "Play with any controller: pad bindings and device-aware hints".
  - red-dead-rocks: the tips, credits script, tests, `tools/ue.sh`, play-tests and README, titled the same.

  Each body has the pad table, the four glyph screenshots, and the test commands and results. Cross-link the two.
- [ ] **Step 3: Merge** each with `merge-gate <PR> --tested "<the cpp-tests, playtest and pytest commands and
  their pass counts>"`, the pipeline PR first. If it holds (🔴), tell Austin in one line with the link.
- [ ] **Step 4: Austin's check on his Mac.** After he pulls and builds `RedRocksEditor`, he plays one session each
  with the Xbox, PlayStation and Switch pads. Each item to check:
  - every row of the pad table works;
  - the hints show that pad's buttons;
  - the mouse feels unchanged.

  Record what he reports in PR 1's thread.

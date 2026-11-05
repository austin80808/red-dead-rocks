# Play Red Rocks with any controller and from a phone

Date: 2026-11-04. Status: design approved in chat section by section; this written spec awaits Austin's review.
Repos: [red-dead-rocks](https://github.com/austin80808/red-dead-rocks) (pipeline, scripts, web page) and
[red-dead-rocks-unreal](https://github.com/austin80808/red-dead-rocks-unreal) (C++ module, project config).

## What we're building and why

Austin wants to play his Red Rocks world on the Mac with a game controller, and on his iPhone at home, away
from home, and by sending friends a link. Today it is keyboard and mouse only, and only on the Mac.

**Done when**

1. **Controllers, on the Mac.** With an Xbox, PlayStation (DualShock 4 or DualSense) or Switch Pro pad, every
   action the keyboard has today works without touching the keyboard. Hints, toasts, the help panel and the
   credits pill show that pad's buttons, and switch back to keys as soon as the keyboard or mouse is used.
   Keyboard and mouse play exactly as before. The headless play-tests pass.
2. **Phone, for Austin.** From his iPhone in Safari, on home Wi-Fi and over cellular through Tailscale, the game
   streams from the Mac at a steady 30 fps or better for 20 minutes. Every action can be done by touch, and a pad
   paired to the phone also works and shows its own buttons.
3. **Phone, for friends.** A friend whose email Austin has allowed opens `https://redrocks.voodude.space`, signs
   in with an emailed PIN and plays. Anyone who joins while someone is playing watches instead of controlling.

## Decisions taken in chat (2026-11-04)

| Question | Decision |
|---|---|
| Phone approach | Stream from the Mac with Pixel Streaming 2 first. A native iPhone app is a possible later project of its own. |
| Where the phone plays | At home, away from home, and friends' phones. |
| Controllers to support | Xbox, PlayStation, Switch Pro / Joy-Con. No generic USB pads, so no SDL3 plugin. |
| How friends get in | A public link behind Cloudflare Access (email PIN), video through Cloudflare's TURN relay. |
| Friends' address | `redrocks.voodude.space` |
| Touch controls | Two context buttons whose labels follow the moment, plus a ☰ menu. |

## What we found (facts the design rests on)

- **Host Mac:** MacBook Air, Apple M4, 24 GB, fanless. Long sessions will throttle, so the stream is capped (see
  [Streaming](#part-2-streaming-to-the-phone)). The other Mac on the tailnet (`shitstation`) is an M2 with 8 GB:
  no better.
- **Phone:** an iPhone. It and the Mac are both on Austin's tailnet already.
- **Frame rate today is unmeasured.** Earlier runs were throttled background editors; issue #25 still asks for real
  `stat unit` numbers. The spike measures it.
- **How Unreal reads pads on Mac:** Apple's GameController framework
  (`Engine/Source/Runtime/ApplicationCore/Private/Apple/AppleControllerInterface.cpp`). It sees Xbox One/Series,
  DualShock 4, DualSense, Switch Pro, Joy-Con pairs and MFi/clip-on pads. It reports a device name per event:
  `Xbox Wireless Controller`, `PS4 Wireless Controller`, `PS5 Wireless Controller`, or `ExtendedGamepad` for
  everything else, Switch included. Buttons are reported by position, so a Switch pad's bottom button (labelled B)
  arrives as `Gamepad_FaceButton_Bottom`. Generic USB HID pads are not seen. The legacy HID path is only used on
  macOS older than 10.14.
- **Input today:** the template's `IMC_Default` maps `Gamepad_Left2D` to move, `Gamepad_Right2D` to look and
  `Gamepad_FaceButton_Bottom` to jump. `IMC_Default` also maps C to `IA_Credits`, which `BP_CreditsMenu` uses.
  Every other key is polled by hand in `ARRExplorer::PollKeys` (`Source/RedRocks/RRExplorer.cpp`).
  `RRGameMode` uses the template's `BP_FirstPersonPlayerController`.
- **Pixel Streaming 2 ships with UE 5.8 for Mac.** The plugin's `.uplugin` allows Mac, Mac binaries are installed,
  and the engine has a VideoToolbox encoder (`Plugins/Experimental/AVCodecs/VTCodecs`).
  [Epic's reference](https://dev.epicgames.com/documentation/en-us/unreal-engine/unreal-engine-pixel-streaming-reference)
  lists VideoToolbox Macs as supported, tested on macOS 13.5.1.
- **Risk:** one third-party project,
  [pixelstreaming2-native-mac](https://github.com/aaryansachdeva/pixelstreaming2-native-mac) (MIT, UE 5.7), says
  Epic's Mac path "crashes on M3 and M4 Apple Silicon with an `RHI.cpp:1764` SRV assertion" or shows dark frames.
  That claim is unverified, so the spike tests it first. Its own backend is H.264 VideoToolbox and skips gamepad input.
- **Pixel Streaming 2 pieces we use:** `UPixelStreaming2Input` (an actor component with an `OnInputEvent` delegate and
  `SendPixelStreaming2Response`), and the cvar `PixelStreaming2.InputController` with values `Any` or `Host`.
  The component's header is under the plugin's `Internal/` folder.
- **Epic's web servers and frontend:**
  [PixelStreamingInfrastructure](https://github.com/EpicGamesExt/PixelStreamingInfrastructure).
  - It has a `UE5.8` branch marked "Current" and supports only Node `v22.14.0`. This Mac has Node 26 from Homebrew.
  - The frontend can send custom messages with `emitUIInteraction`. Its gamepad and touch forwarding can be switched
    off with the `GamepadInput` and `TouchInput` flags.
- **A native iPhone build is a big job.** Lumen doesn't run on iOS
  ([Epic](https://dev.epicgames.com/documentation/unreal-engine/using-lumen-global-illumination-on-mobile-in-unreal-engine)),
  and Nanite on iOS [asserts](https://forums.unrealengine.com/t/ios-nanite/2687586). The world would need a separate
  mobile content pass.
- **Ruled out for phones:**
  - Moonlight/Sunshine: a Mac host [can't pass gamepad input](https://www.mintlify.com/LizardByte/Sunshine/installation/macos),
    and the fork that can needs Apple's file-integrity protection (AMFI) switched off.
  - Steam Link: possible, but its touch controls are clumsy and it needs Steam set up around a non-Steam game.
  - Cloud streaming services: they need a Windows or Linux build, which this Mac can't produce.
- **Cost:** Tailscale, Cloudflare Tunnel and Access (up to 50 users) are free.
  [Cloudflare TURN](https://developers.cloudflare.com/realtime/sfu/platform/pricing/) gives 1,000 GB a month free,
  then $0.05/GB. A 720p stream at about 8 Mbps is about 3.6 GB an hour, so the free tier covers about 275 hours.

## Part 1: Controllers and on-screen hints

### Code layout (red-dead-rocks-unreal, `Source/RedRocks`)

| File | Job |
|---|---|
| `RRInput.h/.cpp` (new) | Builds every `UInputAction` and one `UInputMappingContext` in C++ at startup, from one table of action → keyboard keys → pad buttons. It binds them to `ARRExplorer`. |
| `RRInputDevice.h/.cpp` (new) | A game-instance subsystem owning an `IInputProcessor`. It tracks the last device used: keyboard/mouse, pad (with family), or remote touch. |
| `RRHints.h/.cpp` (new) | Turns `{Token}` text into draw runs for the current device. It also has a plain-text form for tests. |
| `RRPlayerController.h/.cpp` (new) | A minimal C++ player controller, so the template's `BP_FirstPersonPlayerController` can't add `IMC_Default` or its own mobile touch UI. |
| `RRExplorer.h/.cpp` | `PollKeys` and the template input-asset lookups (constructor lines 126–137) are removed. Input setup is handed to `RRInput`. Sprint, spot and toggle actions call the existing `HandleAction`. Hint and toast strings switch to tokens. |
| `RRRide.cpp` | Hint and toast strings switch to tokens. |
| `RRHUD.h/.cpp` | Draws tokens as glyphs, draws the credits pill (moved out of the widget), and scales up for phones. |
| `RRGameMode.cpp` | Uses `ARRPlayerController`. |

`RRExplorer` edits stay small and away from the constructor's movement block, where the terraced-seating branch
may raise `MaxStepHeight`. If both branches touch the file, the second to merge rebases.

The first task of the controllers PR checks that `BP_FirstPersonPlayerController` does nothing else we need. Mouse
capture comes from `DefaultInput.ini`, not from the controller. The template's `IA_Move`, `IA_Look`,
`IA_MouseLook`, `IA_Jump`, `IMC_Default` and `IMC_MouseLook` assets stop being referenced. They stay on disk until
the cleanup issue (#24). `IA_Credits` stays in use.

### The action table

Keyboard and mouse are unchanged. Pad names are Xbox / PlayStation; a Switch pad uses the same positions.

| Action (token) | Keyboard / mouse | Pad | What it does |
|---|---|---|---|
| Move `{Move}` | WASD | Left stick | walk, climb, bank/dive/flare, steer rides |
| Look `{Look}` | Mouse | Right stick | look; steers the wingsuit |
| Run `{Run}` | Shift (hold) | L3 (click, toggles) | run; climb faster |
| Jump `{Jump}` | Space | A / ✕ | jump, leap off, wingsuit, chute, hop |
| Interact `{Interact}` | E | X / □ | ride / get off |
| Drop `{Drop}` | Q | B / ○ | let go / cut away |
| Camera `{Camera}` | V | Y / △ | chase camera |
| Respawn `{Respawn}` | R | RB / R1 | back to the last spot |
| Spots `{Spots}` | 1–9 | D-pad ← / → | previous / next spot, wrapping 9 → 1 |
| Night `{Night}` | T | D-pad ↑ | night and the concert |
| Snow `{Snow}` | N | D-pad ↓ | snow |
| Help `{Help}` | H | View / Create / − | show or hide the help panel |
| Credits `{Credits}` | C | Menu / Options / + | show or hide the credits (mapped to the existing `IA_Credits`) |

Behaviour details:

- **Look on a stick turns at a rate, not by distance.** Full tilt is 220°/s yaw and 150°/s pitch, after a 0.2 radial
  dead zone and a response exponent of 2.0, times the frame time. The console variables `rr.PadLookYaw`,
  `rr.PadLookPitch`, `rr.PadLookExponent` and `rr.PadInvertY` (default off) tune it without a rebuild. Mouse look
  matches today's feel. The controllers PR reads the template assets' modifiers first and copies them.
- **Run on a pad is a toggle.** L3 turns it on. It turns off after the left stick has been centred for 0.5 s, or
  when the mode becomes Wingsuit or Parachute. Shift still works by holding.
- **Spots on a pad cycle.** D-pad → goes to the next spot after `LastSpot`, ← to the previous one. The help panel's
  GO TO list highlights the current spot. The number keys still jump straight to a spot.
- **Not mapped:** the Home/PS/Xbox button (macOS reserves it), and LB, LT, RT and R3 (left free).

### Which device hints show

`RRInputDevice` watches Slate input:

- **Pad:** any pad button, or a stick past its dead zone, sets the device to a pad. The family comes from
  `FInputDeviceScope::GetCurrent()->InputDeviceName`: `Xbox Wireless Controller` → Xbox, `PS4/PS5 Wireless
  Controller` → PlayStation, anything else → Position (Switch and unknown pads).
- **Keyboard/mouse:** any key, or mouse movement over 3 px, sets keyboard/mouse.
- **Remote:** the streaming bridge sets Touch, or a pad family, from the page's `device` message (Part 3).

### Tokens and glyphs

Every player-facing string that names a key uses `{Action}` tokens. `RRHints` renders each token for the current
device as a small chip drawn on the canvas, with no new textures:

| Device | `{Jump}` looks like | Composite tokens |
|---|---|---|
| Keyboard | key cap "SPACE" | `{Move}` "WASD", `{Look}` "MOUSE", `{Run}` "SHIFT", `{Spots}` "1–9" |
| Xbox | "A" on a green disc (B red, X blue, Y yellow, bumpers as pills) | `{Move}` "L", `{Look}` "R" stick rings, `{Run}` "L3", `{Spots}` D-pad ←→ |
| PlayStation | ✕ drawn in lines on a dark disc (○ □ △ the same way) | as Xbox, with L1/R1 for bumpers |
| Position | four-dot diamond, bottom dot filled | as Xbox |
| Touch | chip with the on-screen button's current label (e.g. "LEAP OFF") | `{Move}` "left stick", `{Look}` "drag", `{Run}` "RUN", menu items as "☰ Snow" |

Strings that change (old → new):

| Where | Old | New |
|---|---|---|
| Welcome toast | "H shows the controls." | "{Help} shows the controls." |
| Climb toast | "Space leaps off, Q lets go." | "{Jump} leaps off, {Drop} lets go." |
| Wingsuit toast | "Mouse steers, W dives, S flares, Space pulls the chute." | "{Look} steers, {Move} dives and flares, {Jump} pulls the chute." |
| Parachute toast | "A/D turn, W/S speed, Q cuts away." | "{Move} turns and trims, {Drop} cuts away." |
| Crash toast | "R to try again." | "{Respawn} to try again." |
| Night toast | "The band is on: press 1 for the stage." | "The band is on: {Spots} to spot 1, the stage." |
| Walk hints | "SPACE  wingsuit", "E  %s" | "{Jump}  wingsuit", "{Interact}  %s" |
| Climb hint | "WASD climb   SHIFT faster   SPACE leap off   Q let go" | "{Move} climb   {Run} faster   {Jump} leap off   {Drop} let go" |
| Wingsuit hint | "MOUSE steer   A/D bank   W dive   S flare   SPACE parachute" | "{Look} steer   {Move} bank, dive, flare   {Jump} parachute" |
| Parachute hint | "A/D turn   W faster   S slower   Q cut away" | "{Move} turn and speed   {Drop} cut away" |
| Ride hint (RRRide) | "W push   S brake   A/D steer   SPACE hop   E get off" | "{Move} push, brake, steer   {Jump} hop   {Interact} get off" |
| Tube toast (RRRide) | "W paddles, A/D steer, the creek does the rest." | "{Move} paddles and steers, the creek does the rest." |
| Sled toast (RRRide) | "Press N to make it snow." | "{Snow} makes it snow." |
| Help panel (RRHUD) | the key list | the same list with tokens; "GO TO" shows `{Spots}` |
| Spot tips (`src/redrocks/explore.py`) | "Shift climbs faster", "press Space for the wingsuit", "E to climb in", "Press N for snow, then E", "E to grab an inner tube" | "{Run} climbs faster", "{Jump} for the wingsuit", "{Interact} to climb in", "{Snow} for snow, then {Interact}", "{Interact} to grab an inner tube" |

Spot tips reach the game through `setup_explore.py` on the next level rebuild, run in the agent copy and shipped as
a Rebuild PR in red-dead-rocks-unreal.

**Credits.** `setup_credits.py` stops building the "C Credits" pill (page 0 of `WBP_Credits` becomes empty) and no
longer maps a key into `IMC_Default`. `RRInput` maps C and the pad's Menu button to `IA_Credits`. `RRHUD` draws the
pill as "{Credits} Credits" in the bottom-left corner, as before. The credits card's footer becomes "C or the pad's
Menu button closes". `RRInput` also tracks whether the credits are open, toggling on each `IA_Credits` press, for
the phone page.

## Part 2: Streaming to the phone

### What runs

`remote-play start` (in red-dead-rocks, `tools/remote-play`) does this, in order:

1. Runs `unreal-lock acquire "Red Rocks" "remote play"`, then `caffeinate -dimsu` for as long as the game runs.
   With the lid closed, a MacBook Air only stays awake on power with an external display; the script says so.
2. Generates a 12-hour TURN credential from Cloudflare's API. The API key lives in the 1Password item
   `Red Rocks TURN`, which PR 3 creates; PR 3 also records its `op://` path in `.env.example`.
3. Starts Epic's signalling and web server (Wilbur) from PixelStreamingInfrastructure `UE5.8`, at a commit pinned
   in `remote/PS_INFRA_COMMIT`:
   - It runs on Node 22 and is checked out and built by `remote/setup.sh` into `~/.cache/redrocks-remote/`.
   - It serves our page from `remote/web/`.
   - Player port 8080 and streamer port 8888, both on 127.0.0.1 only.
4. Launches the game from Austin's project:
   - Command: `UnrealEditor RedRocks.uproject /Game/RedRocks/Lvl_RedRocks -game -windowed -ResX=1280 -ResY=720`.
   - It adds the Pixel Streaming flags (connection URL `ws://127.0.0.1:8888`, H.264, `InputController=Host`) and
     `t.MaxFPS 30`.
   - The spike confirms exact flag spellings and the 30 vs 60 cap, and the script records them.
   - The script refuses to start if `RedRocksEditor` is older than the C++ sources and says to build first. No
     packaged build is needed, so issue #25 doesn't block this.
5. Runs `tailscale serve` for HTTPS on the Mac's `.ts.net` name, proxying to `127.0.0.1:8080`. This is tailnet-only
   serve, not Funnel. Safari only gives web pages the Gamepad API over HTTPS.
6. Runs `cloudflared tunnel run redrocks-mac`, mapping `redrocks.voodude.space` to `http://127.0.0.1:8080`.
7. Prints both links.

`remote-play stop` stops everything in reverse and releases the lock. If the game exits on its own, the script stops
everything, releases the lock and prints the log path. `remote-play status` shows what's running.

### Networking

- **Austin (tailnet):** the page loads from `https://<mac>.ts.net`. WebRTC connects directly over the Tailscale
  interface, at home and away, with no relay.
- **Friends (public):** the page loads from `https://redrocks.voodude.space` through the Tunnel, behind a Cloudflare
  Access self-hosted app:
  - Login is email one-time PIN, with a 24-hour session.
  - The policy is an allowlist: Austin's email plus the friends he names.
  - The WebSocket signalling goes through the same hostname, so it's behind Access too.
- **Video for friends:** it can't go through the Tunnel (WebRTC uses UDP), so it goes through Cloudflare TURN:
  - The page sets `iceTransportPolicy: "relay"` whenever it isn't on a `.ts.net` host, so friends never see the
    Mac's IP.
  - The TURN credential reaches the page through Wilbur's peer-connection options.
- **Off switch:** when remote play is off, the public link shows Cloudflare's offline page after sign-in.

### Who controls

`PixelStreaming2.InputController=Host` gives control to the host peer, the first to connect (the spike confirms
this). Others watch. The bridge sends `controller: false` to them, and the page hides its controls and shows
"Watching: someone else is playing". Browser console commands stay off: no `-AllowPixelStreamingCommands`.

### The bridge in the game (red-dead-rocks-unreal)

- **Plugin:** `RedRocks.uproject` enables `PixelStreaming2`.
- **`RRRemote.h/.cpp` (new):** an actor component on the explorer. At `BeginPlay` it finds the
  `UPixelStreaming2Input` class by name (`/Script/PixelStreaming2.PixelStreaming2Input`), adds one to the explorer
  and binds `OnInputEvent` through reflection. Our module therefore doesn't include the plugin's `Internal/`
  headers, and still builds with the plugin off.
- **Input:** it parses the page's messages (Part 3) and feeds them into the same paths as local input:
  - `MoveInput`, the same rate-based look as the stick, `HandleAction`, `RRGo`, and the run toggle.
  - It sets `RRInputDevice` to Touch or the reported pad family.
- **Output:** it sends `state` through `SendPixelStreaming2Response` on connect and whenever a field changes, at most
  10 times a second.
- **Phone HUD:** while the device is remote Touch, `RRHUD` draws 1.5× larger. It hides the help panel and the
  credits pill (the ☰ menu has both) and keeps the status, the hint line and toasts.

### Quality and frame rate

The stream starts at 1280×720 H.264, capped at 30 fps; Pixel Streaming adapts the bitrate. The spike records
`stat unit` once a minute for 20 minutes at the top of the bowl, the busiest view. That shows whether the fanless
Air holds 30 (or 60) once it's hot. If it can't hold 30, the spike also tries `r.ScreenPercentage 75`, then
`sg.FoliageQuality 2`, and records which is needed. Those settings apply only to remote-play launches.

## Part 3: The touch page and the messages

### The page (red-dead-rocks, `remote/web/`)

- **Code:** plain ES modules with no build step, plus the Epic frontend library bundle that `remote/setup.sh` builds.
  The logic is in `protocol.js` and `sticks.js` (pure, tested); `overlay.js` handles drawing and touch; `app.js`
  connects to the stream.
- **Start screen:** "Tap to play". iOS needs a tap before video with sound can play. In portrait it shows
  "Turn your phone sideways".
- **Home Screen:** a web-app manifest lets it be added to the Home Screen for true full screen. Safari on iPhone
  has no fullscreen API.
- **Left half:** a floating stick that appears where the thumb lands. It sends `move` at 30 Hz while held and a zero
  `move` on release. A RUN chip sits beside it and toggles like L3.
- **Right half:** dragging sends `look` deltas, scaled by a sensitivity slider in the menu.
- **Two context buttons, bottom right:**
  - Primary (the Jump action) and secondary (Interact or Drop; the modes where each applies never overlap).
  - Labels come from `state`, and a button is hidden when its label is empty.
  - Possible labels: Jump, Wingsuit, Chute, Leap off, Hop / Ride, Get off, Let go, Cut away.
- **☰ menu, top left:** spots 1–9 by name, Snow, Night, Camera, Respawn, Help and Credits, plus look sensitivity.
  Tapping outside closes it. While `credits` is true, a "Close credits" button shows instead of the game buttons.
- **Status chip, top right:** round-trip time and fps from WebRTC stats. A reconnect button appears if the stream drops.
- **A pad paired to the phone:** the page reads it through the Gamepad API, maps the standard layout onto the same
  actions, and sends `sticks` plus `press`/`release`. The family is guessed from the pad's id string (`Xbox`,
  `DualSense`/`DualShock`, otherwise Position) and sent in `device`.
- **No double input:** the Epic library's own `GamepadInput`, `TouchInput` and `FakeMouseWithTouches` are off.
  Keyboard and mouse forwarding stays on, so a friend on a laptop can play with keys.
- **Demo mode:** `?demo=1` runs the page with scripted `state` messages and no stream, for screenshots.

### Messages

Each message is one JSON object with `"v": 1` and a type `t`. Unknown types are ignored, and logged once each.

| Direction | `t` | Fields | Meaning |
|---|---|---|---|
| page → game | `move` | `x`, `y` in −1..1 | touch stick (x right, y forward) |
| page → game | `look` | `dx`, `dy` in degrees | touch drag since the last message |
| page → game | `sticks` | `lx`, `ly`, `rx`, `ry` in −1..1 | phone-paired pad; the game applies the stick look rate |
| page → game | `press` / `release` | `a`: Jump, Interact, Drop, Run, Camera, Respawn, Snow, Night, Help, Credits, SpotNext, SpotPrev | buttons |
| page → game | `go` | `n` 1–9 | menu spot |
| page → game | `device` | `kind`: touch or pad; `family`: xbox, ps, position | for glyphs and the phone HUD |
| game → page | `state` | `mode`, `primary`, `secondary`, `run`, `credits`, `controller`, `spots` (titles) | button labels and what to show |

### When things go wrong

- **Stale input:** if `move` or `sticks` stops for 500 ms while non-zero, the bridge zeros it. When the peer
  disconnects, it zeros all remote input and turns run off.
- **Bad messages:** a malformed or unknown message is ignored, never crashes, and is logged once per type.
- **Plugin missing:** if the plugin is off or the class isn't found, `RRRemote` logs one line and does nothing.
  Local play is unchanged.
- **Script failures:** `remote-play` checks each step and stops with a clear message: the lock is held by Hanging
  Lake, Node 22 is missing, the editor binary is stale, the tunnel or TURN key is missing, or a port is busy. When
  the lock is held, it names the "Hanging Lake work" session to message.

## The spike (first, before anything is built)

The spike is throwaway. It runs in the `RedRocks-agent` copy, holding the Unreal lock, and keeps nothing except a
findings note.

1. Enable `PixelStreaming2` in the agent copy (not committed) and build `RedRocksEditor`.
2. Fetch PixelStreamingInfrastructure `UE5.8` and run Wilbur with the stock frontend on Node 22.
3. Launch `-game` with streaming on. Check in Safari on the Mac, then on Austin's iPhone over `tailscale serve`.
   Austin opens one link on his phone.
4. Measure fps, latency and bitrate from the stats, plus `stat unit` each minute for 20 minutes at the top of the bowl.
5. Send one `emitUIInteraction` from the page and see it logged by a test `UPixelStreaming2Input`. Send one
   response back and see it in the page console.

**Passes when** the video reaches the iPhone over Tailscale at 30 fps or better for 10 minutes without a crash, and
the message round trip works.

**If Epic's Mac path crashes or streams dark frames:** read the MIT third-party project's source, write its fix ourselves in our own module (copy the back buffer into a texture the encoder can read, before encoding), and repeat. We don't download or run its code: it has one commit, three stars and no review. The bridge doesn't
depend on Pixel Streaming's gamepad forwarding, so either backend works. If both fail, stop and bring it back to
Austin.

Findings go into this spec as an appendix, and into the shared notes folder for the session group.

Two downloads need Austin's yes when the spike starts: Homebrew `node@22` (keg-only, so it won't replace Node 26),
and Epic's PixelStreamingInfrastructure repo.

## Testing

- **Headless play-tests** (`unreal/playtest_explore.py`, in the agent copy, holding the lock):
  - `RRInputDescribe()` returns the live action → keys table, and the test compares it with
    `tests/data/input_table.json`.
  - `RRRemoteInject(json)` pushes page messages into the bridge without streaming:
    - `move` makes the player walk.
    - `press Jump` while falling more than 5 m up starts the wingsuit.
    - Stale `move` is zeroed after 500 ms.
    - `go 3` warps to spot 3.
  - `RRHintsRender(text, device)` returns the plain form of tokens for each device. Every in-game string renders
    with no leftover `{` and no bare key names.
- **pytest:** spot tips in `src/redrocks/explore.py` use only known tokens and no bare key names ("Press E",
  "Space", "Shift").
- **`node --test remote/web/test/`:** message encoding and decoding, stick dead zone and curve, label → button
  mapping, the 500 ms stale rule on the page side, and the Gamepad API mapping.
- **Screenshots** of `?demo=1` at iPhone sizes (landscape 852×393 and 932×430, plus portrait) go through the cloud
  Visual QA routine. red-dead-rocks has to be added to that routine first.
- **Austin's iPhone checklist, last:**
  - Home Wi-Fi: every action, by touch and with a pad.
  - Cellular via Tailscale.
  - The friends' link with an allowlisted email.
  - A second viewer watching.

## Rollout

Each step is its own PR, merged through `merge-gate`.

| # | What | Repos | merge-gate |
|---|---|---|---|
| 0 | Spike findings (appendix to this spec) | red-dead-rocks | 🟢 docs |
| 1 | Controllers: `RRInput`, `RRInputDevice`, `RRHints`, `RRPlayerController`, HUD glyphs, credits pill, token strings; token tips, `setup_credits.py`, play-test checks; then a Rebuild PR for the new tips | both | 🟡 |
| 2 | Phone over Tailscale: plugin on, `RRRemote`, `remote/` page, `remote/setup.sh`, `tools/remote-play`, README "Play on your phone" | both | 🔴 held (pulls in Epic's packages) |
| 3 | Friends' link: tunnel `redrocks-mac`, DNS, Access app and allowlist, TURN key in 1Password (`op://` path only), Notion registry row for `redrocks.voodude.space` | red-dead-rocks plus Cloudflare | 🔴 held; Austin confirms before it goes live |

PR 1 is useful on its own: controllers work on the Mac the day it merges.

**Plans.** Two implementation plans. The first covers the spike and PR 1. The second covers PRs 2 and 3, and is
written after the spike, because the spike settles the launch flags, the frame-rate cap and which Pixel Streaming
backend to use.

## Out of scope

- A native iPhone or Android build. A later project of its own if wanted.
- Generic USB pads (SDL3) and on-screen key remapping.
- More than one player at a time, or multiplayer of any kind.
- Cloud-hosted streaming.
- Android phones. Chrome on Android should work, but it isn't on the checklist.

## Risks

| Risk | Plan |
|---|---|
| Epic's Pixel Streaming Mac path is broken on M4 | The spike tests it first; then we write the third-party fix ourselves; then stop and ask. |
| The fanless Air throttles | 720p at 30 fps; the spike measures 20 minutes hot; remote-only scalability steps. |
| iOS Safari quirks (autoplay, the Gamepad API) | "Tap to play"; HTTPS everywhere; both checked in the spike and on the checklist. |
| WebRTC doesn't pick the Tailscale route | Checked in the spike; if it fails, use TURN for tailnet too. |
| `InputController=Host` means something other than "first peer" | Checked in the spike; otherwise the bridge ignores input from all but the first peer it heard from. |
| Clash with the terraced-seating branch in `RRExplorer.cpp` | Small edits away from the movement block; rebase on whichever merges second. |

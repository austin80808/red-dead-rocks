#!/bin/bash
# Red Rocks on the Runpod 4090 (pod ue-redrocks-4090), as user ue.
#   rr_pod.sh build              compile the RedRocks editor module for Linux
#   rr_pod.sh shots PREFIX       beauty shots (docs/media/beauty/views.json) -> $OUT/board/PREFIX<view>.png
#   rr_pod.sh rebuild            rebuild Lvl_RedRocks from $OUT (wipes the level)
#   rr_pod.sh py SCRIPT          run one unreal/ script headless (-run=pythonscript)
# Env: RR_ONLY (comma list of views), RR_WIDTH (default 1920).
set -u
P=/workspace/projects/RedRocks/RedRocks.uproject; UE=/workspace/UE_5.8/Engine
REPO=/workspace/redrocks-ue; OUT=$REPO/out; W=/workspace/work/redrocks
export HOME=/home/ue XDG_RUNTIME_DIR=/run/user/1001 VK_DRIVER_FILES=/workspace/vk/nvidia_egl_icd.json
export REDROCKS_OUT_DIR=$OUT REDROCKS_REPO=$REPO REDROCKS_UNREAL_DIR=$REPO/unreal REDROCKS_LOG_DIR=$W/logs
mkdir -p $W/logs; L=$W/logs/rr_$1.log
say() { echo "$(date -u +%T) $*" | tee -a $L; }
case "$1" in
  build) say "build"; $UE/Build/BatchFiles/Linux/Build.sh RedRocksEditor Linux Development -project="$P" -waitmutex -NoUBA >> $L 2>&1; say "build rc=$?";;
  rebuild|py)
    S=${2:-$REPO/unreal/rebuild_level.py}; [ "$1" = rebuild ] && S=$REPO/unreal/rebuild_level.py
    say "headless $S"
    timeout 5400 $UE/Binaries/Linux/UnrealEditor-Cmd "$P" -run=pythonscript -script="$S" -unattended -nosplash -NoSound -stdout >> $L 2>&1
    say "rc=$?"; cat $W/logs/redrocks_rebuild.log 2>/dev/null | tail -25 | tee -a $L;;
  shots)
    say "shots ${2:-before_}"
    REDROCKS_VIEWS=$REPO/docs/media/beauty/views.json REDROCKS_BOARD_PREFIX=${2:-before_} REDROCKS_BOARD_SUN=view \
    REDROCKS_BOARD_ONLY=${RR_ONLY:-} REDROCKS_BOARD_WIDTH=${RR_WIDTH:-1920} \
    timeout 3600 $UE/Binaries/Linux/UnrealEditor "$P" -RenderOffscreen -nosplash -unattended -NoSound -ResX=1920 -ResY=1080 \
      "-ini:EditorPerProjectUserSettings:[/Script/UnrealEd.EditorPerformanceSettings]:bThrottleCPUWhenNotForeground=False" \
      -ExecCmds="py $REPO/unreal/shoot_board.py" >> $L 2>&1
    say "editor rc=$?"; tail -12 $W/logs/redrocks_board.log | tee -a $L;;
esac
# Persistent editor (unreal/pod_server.py): rr_pod.sh serve, then drop jobs with
#   rr_pod.sh job NAME SCRIPT [KEY=VAL ...]
case "$1" in
  serve)
    mkdir -p $W/queue; say "serve"
    $UE/Binaries/Linux/UnrealEditor "$P" -RenderOffscreen -nosplash -unattended -NoSound -ResX=1920 -ResY=1080 \
      "-ini:EditorPerProjectUserSettings:[/Script/UnrealEd.EditorPerformanceSettings]:bThrottleCPUWhenNotForeground=False" \
      -ExecCmds="py $REPO/unreal/pod_server.py" >> $L 2>&1
    say "editor rc=$?";;
  job)
    N=$2; S=$3; shift 3; E=""
    for kv in "$@"; do E="$E\"${kv%%=*}\": \"${kv#*=}\","; done
    echo "{\"script\": \"$S\", \"env\": {${E%,}}}" > $W/queue/.$N.tmp && mv $W/queue/.$N.tmp $W/queue/$N.json
    echo "queued $N";;
esac

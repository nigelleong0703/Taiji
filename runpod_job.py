"""Run one of this repo's pod jobs on RunPod: create the pod (retrying while none is free), watch its log, save the
logs, delete the pod. Every *_round.sh / speed_probe.sh job reads its code from CODE_B64 and serves logs on port 8888.

python runpod_job.py --script mix_round.sh --log mix.log --done MIX_DONE --code mix.py s1.py train.py check_init.py \
    requirements.txt --env NAME=v4s "MIX=nnetnav=12000 agenttrek=10000" --hf-token --kind cpu
python runpod_job.py ... --kind gpu --gpus "NVIDIA RTX A6000,NVIDIA L40S" --region us --dry-run

Secrets come from files, never the command line: ~/.config/runpod/key (RunPod API key), ~/.cache/huggingface/token
(--hf-token adds HF_TOKEN), --env-file (e.g. the agent's .env, sent as ENV_B64 for teacher_round.sh).
"""

import argparse
import base64
import io
import json
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
US = ["US-CA-2", "US-WA-1", "US-TX-3", "US-TX-4", "US-IL-1", "US-GA-2", "US-KS-2", "US-NC-1", "US-DE-1", "US-MD-1",
      "US-GA-1", "US-KS-3"]
CHEAP_GPUS = "NVIDIA RTX A4000,NVIDIA GeForce RTX 3090,NVIDIA RTX A5000,NVIDIA L4,NVIDIA GeForce RTX 4090"


def api(method, path, key, body=None):
    request = urllib.request.Request(f"https://rest.runpod.io/v1/{path}", json.dumps(body).encode() if body else None,
                                     {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                      "User-Agent": "taiji"}, method=method)
    return json.loads(urllib.request.urlopen(request, timeout=60).read() or b"{}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--script", required=True, help="the job, run as the pod's start command")
    p.add_argument("--log", required=True, help="the job's main log under /workspace/s1/logs")
    p.add_argument("--done", required=True, help="the log line that ends the job (a line ending in _FAILED also does)")
    p.add_argument("--code", nargs="*", default=[], help="repo files packed into CODE_B64")
    p.add_argument("--env", nargs="*", default=[], help="KEY=VALUE pairs")
    p.add_argument("--env-file", help="sent as ENV_B64 (teacher_round.sh writes it to /root/agent.env)")
    p.add_argument("--hf-token", action="store_true", help="add HF_TOKEN from ~/.cache/huggingface/token")
    p.add_argument("--kind", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--gpus", default=CHEAP_GPUS, help="comma-separated RunPod GPU type ids, tried in order")
    p.add_argument("--region", choices=["any", "us"], default="any")
    p.add_argument("--volume", help="network volume id to mount at /workspace (pins the pod to its datacenter)")
    p.add_argument("--datacenter", help="with --volume: the volume's datacenter, e.g. AP-JP-1")
    p.add_argument("--disk", type=int, default=40, help="GB for /workspace (container disk on CPU pods; max 40 there)")
    p.add_argument("--name", default="taiji-job")
    p.add_argument("--out", default="runpod_logs", help="folder for the saved logs")
    p.add_argument("--extra-logs", nargs="*", default=[], help="more log names to save, e.g. probe_base.log")
    p.add_argument("--retries", type=int, default=30, help="attempts, 2 minutes apart, while no machine is free")
    p.add_argument("--max-minutes", type=int, default=600)
    p.add_argument("--dry-run", action="store_true", help="print the first pod request and stop")
    args = p.parse_args()

    key = (Path.home() / ".config/runpod/key").read_text().strip()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name in args.code:
            tar.add(HERE / name, arcname=name)
    env = dict(item.split("=", 1) for item in args.env)
    env["CODE_B64"] = base64.b64encode(buffer.getvalue()).decode()
    if args.hf_token:
        env["HF_TOKEN"] = (Path.home() / ".cache/huggingface/token").read_text().strip()
    if args.env_file:
        env["ENV_B64"] = base64.b64encode(Path(args.env_file).expanduser().read_bytes()).decode()
    common = {"name": args.name, "imageName": "runpod/base:1.3.2-ubuntu2204", "ports": ["8888/http"], "env": env,
              "dockerStartCmd": ["bash", "-c", (HERE / args.script).read_text()]}
    if args.volume:
        common.update(networkVolumeId=args.volume, volumeMountPath="/workspace", dataCenterIds=[args.datacenter])
    if args.kind == "cpu":
        options = [{"computeType": "CPU", "cpuFlavorIds": [f], "vcpuCount": v, "containerDiskInGb": min(args.disk, 40)}
                   for f in ("cpu5c", "cpu3c", "cpu5g", "cpu3g", "cpu5m", "cpu3m") for v in (8, 16, 4)]
    else:
        regions = [US, None] if args.region == "us" and not args.volume else [None]
        options = [{"cloudType": cloud, "gpuCount": 1, "gpuTypeIds": [gpu], "containerDiskInGb": 30,
                    **({} if args.volume else {"volumeInGb": args.disk, "volumeMountPath": "/workspace"}),
                    **({"dataCenterIds": dcs} if dcs else {})}
                   for dcs in regions for cloud in ("SECURE", "COMMUNITY") for gpu in args.gpus.split(",")]
    if args.dry_run:
        hidden = ("TOKEN", "KEY", "_B64")  # secrets and packed payloads: never printed, not even partly
        shown = {**common, **options[0],
                 "env": {k: f"<{len(v)} chars>" if any(h in k for h in hidden) else v for k, v in env.items()}}
        shown["dockerStartCmd"] = ["bash", "-c", f"<{args.script}>"]
        print(json.dumps(shown, indent=2), f"\n({len(options)} machine options)")
        return

    pod = None
    for attempt in range(args.retries):
        for option in options:
            try:
                created = api("POST", "pods", key, {**common, **option})
                pod = created["id"]
                print(time.strftime("%H:%M"), "created", pod, option.get("cpuFlavorIds") or option.get("gpuTypeIds"),
                      f"${created.get('costPerHr')}/h", flush=True)
                break
            except urllib.error.HTTPError:
                continue
        if pod:
            break
        print(time.strftime("%H:%M"), "no machine free, retrying in 2 minutes", flush=True)
        time.sleep(120)
    if not pod:
        raise SystemExit("gave up: no machine")

    def fetch(name):
        request = urllib.request.Request(f"https://{pod}-8888.proxy.runpod.net/{name}", headers={"User-Agent": "taiji"})
        try:
            return urllib.request.urlopen(request, timeout=20).read().decode(errors="replace")
        except Exception:  # noqa: BLE001 - the pod may not serve logs yet
            return ""

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        while time.time() - started < args.max_minutes * 60:
            log = fetch(args.log)
            if args.done in log or any(line.strip().endswith("_FAILED") for line in log.splitlines()):
                break
            time.sleep(60)
        else:
            print("timed out", flush=True)
    finally:
        for name in [args.log, *args.extra_logs]:
            (out / name).write_text(fetch(name))
        api("DELETE", f"pods/{pod}", key)
        print(f"deleted {pod}; logs in {out}/ ({round((time.time() - started) / 60)} min)", flush=True)
    tail = [line for line in (out / args.log).read_text().splitlines() if not line.startswith("+ echo")][-15:]
    print("\n".join(line[:300] for line in tail))


if __name__ == "__main__":
    main()

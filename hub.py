"""Bulk downloads from the Hugging Face Hub.

The Hub rate-limits its per-file download endpoint. A tree listing (1,000 entries per call) plus one Git LFS batch call
(up to 500 files) return direct CDN links, so thousands of small LFS files cost a handful of Hub requests instead of
one each; the CDN downloads themselves are not rate-limited.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

HUB = "https://huggingface.co"
_lock = threading.Lock()
_clients = {}


def token():
    path = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "token"
    return os.environ.get("HF_TOKEN") or (path.read_text().strip() if path.exists() else None)


def client():
    with _lock:  # one pooled client per process (worker processes each make their own)
        if os.getpid() not in _clients:
            auth = {"Authorization": f"Bearer {token()}"} if token() else {}
            _clients[os.getpid()] = httpx.Client(timeout=120, follow_redirects=True, headers=auth,
                                                 limits=httpx.Limits(max_connections=32))
        return _clients[os.getpid()]


def _hub(method, url, **kwargs):
    """A Hub API call that waits out 429s and retries dropped connections."""
    for attempt in range(10):
        try:
            response = client().request(method, url, **kwargs)
            if response.status_code != 429 and response.status_code < 500:
                response.raise_for_status()
                return response
        except httpx.TransportError:
            pass
        time.sleep(min(120, 5 * 2 ** attempt))
    raise RuntimeError(f"Hub request kept failing: {url}")


def tree(repo, path):
    """Every file under `path` in a dataset repo: [{"path", "size", "lfs": {"oid", "size"} or absent}, ...]."""
    url, files = f"{HUB}/api/datasets/{repo}/tree/main/{path}", []
    params = {"recursive": "true"}
    while url:
        response = _hub("GET", url, params=params)
        files += [f for f in response.json() if f["type"] == "file"]
        url, params = response.links.get("next", {}).get("url"), None
    return files


def _fetch(url, headers, path):
    for attempt in range(6):
        try:
            response = client().get(url, headers=headers or {})
            if response.status_code == 200:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(response.content)
                return str(path)
        except httpx.TransportError:
            pass
        time.sleep(1 + attempt)
    return None


def download(repo, entries, cache, threads=16):
    """Tree entries -> {repo path: local file, or None if it failed}. LFS files via batch links, others via resolve."""
    cache, out, todo = Path(cache), {}, []
    for e in entries:
        local = cache / e["path"]
        if local.exists():
            out[e["path"]] = str(local)
        else:
            todo.append(e)
    jobs = [(f"{HUB}/datasets/{repo}/resolve/main/{e['path']}", None, e) for e in todo if not e.get("lfs")]
    lfs = [e for e in todo if e.get("lfs")]
    for i in range(0, len(lfs), 500):
        chunk = lfs[i:i + 500]
        body = {"operation": "download", "transfers": ["basic"],
                "objects": [{"oid": e["lfs"]["oid"], "size": e["lfs"]["size"]} for e in chunk]}
        response = _hub("POST", f"{HUB}/datasets/{repo}.git/info/lfs/objects/batch", json=body,
                        headers={"Accept": "application/vnd.git-lfs+json",
                                 "Content-Type": "application/vnd.git-lfs+json"})
        links = {o["oid"]: o.get("actions", {}).get("download") for o in response.json()["objects"]}
        for e in chunk:
            link = links.get(e["lfs"]["oid"])
            if link:
                jobs.append((link["href"], link.get("header"), e))
            else:
                out[e["path"]] = None
    with ThreadPoolExecutor(threads) as pool:
        results = pool.map(lambda job: _fetch(job[0], job[1], cache / job[2]["path"]), jobs)
        out.update({job[2]["path"]: local for job, local in zip(jobs, results, strict=True)})
    return out

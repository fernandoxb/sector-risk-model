"""Chrome headless controlado pelo DevTools Protocol (CDP): capturas de ecrã e PDF.

Usado por scripts/capturas.py e scripts/nota_pdf.py. Precisa do Google Chrome e de `websockets`.
"""
import asyncio
import json
import subprocess
import tempfile
import time
import urllib.request
from contextlib import asynccontextmanager

import websockets

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 9333


@asynccontextmanager
async def chrome(w: int = 1440, h: int = 1250, scale: float = 2):
    """Abre o Chrome headless e devolve `cmd(method, **params)` para a página em branco."""
    prof = tempfile.mkdtemp(prefix="cdp-")
    proc = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                             f"--remote-debugging-port={PORT}", f"--user-data-dir={prof}",
                             f"--window-size={w},{h}", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        tabs = None
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                break
            except Exception:  # noqa: BLE001  o Chrome ainda está a arrancar
                time.sleep(0.2)
        ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
        async with websockets.connect(ws_url, max_size=100_000_000) as ws:
            n = 0

            async def cmd(method, **params):
                nonlocal n
                n += 1
                await ws.send(json.dumps({"id": n, "method": method, "params": params}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == n:
                        if "error" in msg:
                            raise RuntimeError(f"{method}: {msg['error']}")
                        return msg.get("result", {})

            await cmd("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=scale, mobile=False)
            yield cmd
    finally:
        proc.terminate()


async def wait_for(cmd, js: str, timeout: float = 90) -> bool:
    """Espera até a expressão JavaScript `js` ser verdadeira."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = await cmd("Runtime.evaluate", expression=js, returnByValue=True)
        if r.get("result", {}).get("value"):
            return True
        await asyncio.sleep(1)
    return False

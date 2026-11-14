"""Capturas de ecrã para o README (docs/img/), com o Chrome headless.

Uso:  .venv/bin/python -m scripts.capturas URL SAIDA.png TEXTO_A_ESPERAR [largura altura] [rolar_até_título] [separador]
Espera (até 90 s) que TEXTO_A_ESPERAR apareça e que o Streamlit deixe de mostrar spinners. Com `separador`,
clica primeiro no separador (st.tabs) cujo nome contém esse texto. A variável de ambiente CAPTURA_JS corre
JavaScript na página antes da captura (ex.: abrir um setor no mapa).
Ex.:  .venv/bin/python -m scripts.capturas http://localhost:8501 docs/img/cockpit-backtest.png \\
          "Backtesting do VaR" 1440 1100 "Backtesting do VaR"
"""
import asyncio
import base64
import json
import os
import sys
import time

from scripts.cdp import chrome, wait_for


def comprimir(caminho: str, cores: int = 256) -> None:
    """PNG com uma paleta de `cores` cores: cerca de 4× mais leve, sem diferença visível num ecrã escuro."""
    from PIL import Image
    im = Image.open(caminho).convert("RGB")
    im.quantize(colors=cores, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(caminho, optimize=True)


async def main(url, out, wait_text, w=1440, h=1250, scroll_text=None, tab=None):
    async with chrome(w, h) as cmd:
        t0 = time.time()
        await cmd("Page.navigate", url=url)
        ok = await wait_for(cmd, f"document.body && document.body.innerText.includes({json.dumps(wait_text)}) && "
                                 "!document.body.innerText.includes('A descarregar') && "
                                 "!document.querySelector('[data-testid=\"stSkeleton\"]')")
        if not ok:
            print("aviso: tempo esgotado à espera do texto")
        await asyncio.sleep(3)                # deixa terminar os gráficos
        if tab:
            await cmd("Runtime.evaluate", expression=(
                f"[...document.querySelectorAll('[role=tab]')].find(t=>t.innerText.includes({json.dumps(tab)}))"
                "?.click(); true"))
            await asyncio.sleep(3)
        extra = os.environ.get("CAPTURA_JS")          # passos extra (ex.: escolher um setor) antes da captura
        if extra:
            await cmd("Runtime.evaluate", expression=extra, awaitPromise=True)
            await asyncio.sleep(4)
        if scroll_text:
            await cmd("Runtime.evaluate", expression=(
                f"[...document.querySelectorAll('h1,h2,h3')].find(e=>e.innerText.includes({json.dumps(scroll_text)}))"
                "?.scrollIntoView({block:'start'}); true"))
            await asyncio.sleep(2)
        shot = await cmd("Page.captureScreenshot", format="png")
        with open(out, "wb") as f:
            f.write(base64.b64decode(shot["data"]))
        comprimir(out)
        print(f"gravado {out} em {time.time() - t0:.0f}s")


if __name__ == "__main__":
    a = sys.argv[1:]
    asyncio.run(main(a[0], a[1], a[2], int(a[3]) if len(a) > 3 else 1440, int(a[4]) if len(a) > 4 else 1250,
                     a[5] if len(a) > 5 and a[5] else None, a[6] if len(a) > 6 else None))

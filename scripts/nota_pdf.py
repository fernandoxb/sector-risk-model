"""docs/nota-metodologica.md → docs/nota-metodologica.pdf (ou outro .md com frontmatter → .pdf).

Markdown (markdown-it, CommonMark + tabelas) → HTML com as fórmulas em KaTeX → PDF A4 pelo Chrome headless.
As fórmulas ($…$ e $$…$$) são protegidas antes da conversão e repostas para o KaTeX as desenhar.
Uso:  .venv/bin/python -m scripts.nota_pdf [ENTRADA.md SAIDA.pdf]
No frontmatter, `capa` e `rodape` substituem o texto da capa e do rodapé, e `lingua` o atributo lang (padrão pt-PT).
"""
import asyncio
import base64
import html
import re
import sys
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt

from scripts.cdp import chrome, wait_for

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "docs" / "nota-metodologica.md", ROOT / "docs" / "nota-metodologica.pdf"
KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"

DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.S)
INLINE = re.compile(r"(?<![\\$])\$(?![\s$])([^$\n]+?)(?<!\s)\$")

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
body { font: 10.5pt/1.5 -apple-system, "Helvetica Neue", Arial, sans-serif; color: #1f2937; }
h1 { font-size: 20pt; margin: 0 0 4pt; color: #111827; }
h2 { font-size: 14pt; margin: 18pt 0 6pt; padding-bottom: 3pt; border-bottom: 1px solid #d1d5db;
     break-after: avoid; color: #111827; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; break-after: avoid; color: #374151; }
p, li { orphans: 3; widows: 3; }
code { font: 9pt Menlo, monospace; background: #f3f4f6; padding: 0 3px; border-radius: 3px; }
table { border-collapse: collapse; margin: 6pt 0; font-size: 9pt; break-inside: avoid; }
th, td { border: 1px solid #d1d5db; padding: 3pt 6pt; text-align: left; vertical-align: top; }
th { background: #f3f4f6; }
hr { border: 0; border-top: 1px solid #e5e7eb; margin: 14pt 0; }
.katex-display { margin: 6pt 0; overflow: hidden; }
.capa { color: #6b7280; margin-bottom: 14pt; font-size: 9.5pt; }
"""


def split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        return {}, text
    head, body = text[4:].split("\n---\n", 1)
    meta = dict(line.split(": ", 1) for line in head.splitlines() if ": " in line)
    return meta, body


def to_html(md_text: str) -> str:
    """Markdown com fórmulas → HTML (pura)."""
    meta, body = split_frontmatter(md_text)
    maths: list[str] = []

    def keep(m, display):
        maths.append((m.group(1), display))
        return f"MATHPH{len(maths) - 1}X"

    body = DISPLAY.sub(lambda m: keep(m, True), body)
    body = INLINE.sub(lambda m: keep(m, False), body)
    out = MarkdownIt("commonmark").enable("table").render(body)

    def put(m):
        tex, display = maths[int(m.group(1))]
        tex = html.escape(tex.strip(), quote=False)
        return f"<div>\\[{tex}\\]</div>" if display else f"\\({tex}\\)"

    out = re.sub(r"<p>MATHPH(\d+)X</p>", put, out)
    out = re.sub(r"MATHPH(\d+)X", put, out)
    capa = (f"<div class='capa'>{html.escape(meta['capa'] if 'capa' in meta else meta.get('projeto', '') + ' · versão ' + meta.get('versao', '') + ' · código e dados: github.com/fernandoxb/sector-risk-model')}</div>"
            if meta else "")
    out = out.replace("</h1>", "</h1>" + capa, 1)
    return f"""<!doctype html><html lang="{meta.get('lingua', 'pt-PT') if meta else 'pt-PT'}"><head><meta charset="utf-8">
<title>{html.escape(meta.get('rodape', 'Nota metodológica') if meta else 'Nota metodológica')}</title><style>{CSS}</style>
<link rel="stylesheet" href="{KATEX}/katex.min.css">
<script defer src="{KATEX}/katex.min.js"></script>
<script defer src="{KATEX}/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{{delimiters:[{{left:'\\\\[',right:'\\\\]',display:true}},
  {{left:'\\\\(',right:'\\\\)',display:false}}],throwOnError:false}});window.__ok=true;"></script>
</head><body>{out}</body></html>"""


async def main(src: Path = SRC, out: Path = OUT):
    texto = src.read_text()
    rodape = re.search(r"^rodape:\s*(.+)$", texto, re.M)
    rodape = rodape.group(1).strip() if rodape else "Nota metodológica · Modelo de Risco Setorial"
    page = Path(tempfile.mkdtemp()) / "nota.html"
    page.write_text(to_html(texto), encoding="utf-8")
    async with chrome(1000, 1400, 1) as cmd:
        await cmd("Page.navigate", url=page.as_uri())
        if not await wait_for(cmd, "window.__ok === true", 60):
            raise RuntimeError("O KaTeX não carregou (sem rede?)")
        await asyncio.sleep(1)
        erros = await cmd("Runtime.evaluate", expression="document.querySelectorAll('.katex-error').length",
                          returnByValue=True)
        footer = ("<div style='font-size:8px;width:100%;text-align:center;color:#9ca3af'>"
                  f"{html.escape(rodape)} · <span class='pageNumber'></span>/"
                  "<span class='totalPages'></span></div>")
        pdf = await cmd("Page.printToPDF", printBackground=True, preferCSSPageSize=True,
                        displayHeaderFooter=True, headerTemplate="<div></div>", footerTemplate=footer)
    out.write_bytes(base64.b64decode(pdf["data"]))
    print(f"gravado {out} ({out.stat().st_size / 1e3:.0f} KB); fórmulas com erro no KaTeX: "
          f"{erros['result']['value']}")


if __name__ == "__main__":
    a = sys.argv[1:]
    asyncio.run(main(Path(a[0]), Path(a[1])) if len(a) == 2 else main())

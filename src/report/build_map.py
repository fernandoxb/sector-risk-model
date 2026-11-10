"""Bot Repórter (v1): embebe as métricas medidas no mapa de riscos interativo.

Lê data/features/risk_map_data.json e substitui o bloco /*DATA*/…/*END*/ em app/risk-map/index.html.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import FEATURES, ROOT  # noqa: E402

MAP = ROOT / "app" / "risk-map" / "index.html"


def main() -> Path:
    data = json.loads((FEATURES / "risk_map_data.json").read_text())
    html = MAP.read_text()
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    new, n = re.subn(r"/\*DATA\*/.*?/\*END\*/", lambda _: f"/*DATA*/{blob}/*END*/", html, flags=re.S)
    if n != 1:
        raise RuntimeError("Marcador /*DATA*/…/*END*/ não encontrado no mapa")
    MAP.write_text(new)
    print(f"[repórter] mapa atualizado com dados até {data['asof']}")
    return MAP


if __name__ == "__main__":
    main()

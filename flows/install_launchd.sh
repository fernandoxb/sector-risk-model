#!/bin/sh
# Instala (ou reinstala) o agendamento launchd do pipeline: dias úteis às 22:30.
# Uso:  sh flows/install_launchd.sh            instalar
#       sh flows/install_launchd.sh --remover  desinstalar
#       LABEL=com.exemplo.risco sh flows/install_launchd.sh   com outro nome (padrão: com.sectorrisk.pipeline)
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="${LABEL:-com.sectorrisk.pipeline}"
DEST="$HOME/Library/LaunchAgents/$LABEL.plist"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
if [ "$1" = "--remover" ]; then
    rm -f "$DEST"; echo "Removido $DEST"; exit 0
fi
mkdir -p "$REPO/logs" "$HOME/Library/LaunchAgents"
sed -e "s|__REPO__|$REPO|g" -e "s|__LABEL__|$LABEL|g" "$REPO/flows/pipeline.plist" > "$DEST"
plutil -lint "$DEST"
launchctl bootstrap "gui/$(id -u)" "$DEST"
echo "Instalado $DEST. Testar já:  launchctl kickstart gui/$(id -u)/$LABEL ; logs em $REPO/logs/"

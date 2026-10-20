#!/usr/bin/env bash
# Start Sweep on your Mac and print the URL to open on your iPhone (same Wi-Fi).
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8501}"
PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "No .venv found. Set up first:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

# Local IP of the interface that carries the default route (Wi-Fi or Ethernet).
IFACE="$(route -n get default 2>/dev/null | awk '/interface:/{print $2}')"
IP="$( { [ -n "$IFACE" ] && ipconfig getifaddr "$IFACE"; } 2>/dev/null || true)"
if [ -z "$IP" ]; then
  for i in en0 en1 en2 en3 en4 en5; do IP="$(ipconfig getifaddr "$i" 2>/dev/null || true)"; [ -n "$IP" ] && break; done
fi
URL="http://${IP:-localhost}:${PORT}"

echo
echo "  Sweep is starting. On your iPhone (same Wi-Fi), open:"
echo
echo "      $URL"
echo
"$PY" -c "import segno, sys; segno.make('$URL').terminal(compact=True)" 2>/dev/null || true
echo "  (Point the iPhone camera at the QR code. If macOS asks to allow incoming"
echo "   connections for Python, click Allow.)  Ctrl+C to stop."
echo

# caffeinate keeps the Mac awake while the server runs, so scans don't stall.
exec caffeinate -i "$PY" -m streamlit run app.py \
  --server.address 0.0.0.0 --server.port "$PORT" \
  --server.maxUploadSize 1024 --server.headless true \
  --browser.gatherUsageStats false --client.toolbarMode minimal

#!/bin/bash
# Build AudioTranscribe.app and wrap it in a shareable .dmg.
# Apple Silicon only; see the README for why.
set -e
cd "$(dirname "$0")"

VERSION="1.0.0"
DMG="dist/AudioTranscribe-$VERSION-AppleSilicon.dmg"

./.venv/bin/pip install -q -r requirements-build.txt

rm -rf build dist
./.venv/bin/pyinstaller --noconfirm --clean --log-level WARN AudioTranscribe.spec

# Staging folder: the app plus a drag-to-install shortcut and the read-me.
rm -rf build/dmg && mkdir -p build/dmg
cp -R "dist/AudioTranscribe.app" build/dmg/
ln -s /Applications build/dmg/Applications
cp packaging/FIRST-RUN.txt build/dmg/ 2>/dev/null || true

rm -f "$DMG"
hdiutil create -volname "AudioTranscribe" -srcfolder build/dmg \
               -ov -format UDZO "$DMG" >/dev/null

echo
echo "built $DMG"
du -sh "dist/AudioTranscribe.app" "$DMG"

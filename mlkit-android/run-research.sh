#!/usr/bin/env bash
set -euo pipefail
gradle -p mlkit-android --no-daemon :app:assembleDebug :app:assembleDebugAndroidTest
test "$(adb shell getprop ro.build.version.sdk | tr -d '\r')" = 35
adb install -r mlkit-android/app/build/outputs/apk/debug/app-debug.apk
adb install -r -t mlkit-android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
# Only the disposable emulator is disconnected. The bundled recognizer must work offline.
adb shell svc wifi disable
adb shell svc data disable
adb shell pm clear org.mindloop.mlkitlab
adb shell am instrument -w -r org.mindloop.mlkitlab.test/androidx.test.runner.AndroidJUnitRunner | tee "$RUNNER_TEMP/mlkit-instrument.log"
grep -Fq 'OK (5 tests)' "$RUNNER_TEMP/mlkit-instrument.log"
adb exec-out run-as org.mindloop.mlkitlab cat files/observations.json > "$RUNNER_TEMP/mlkit-observations.json"
python3 - <<'PY'
import hashlib, json, os, subprocess
from pathlib import Path
root = Path(os.environ.get('GRADLE_USER_HOME', str(Path.home() / '.gradle'))) / 'caches/modules-2/files-2.1/com.google.mlkit'
files = sorted(root.rglob('*.aar'))
assert any(p.name == 'text-recognition-16.0.1.aar' for p in files), 'Bundled SDK AAR missing'
data = {'sdk': 'com.google.mlkit:text-recognition:16.0.1', 'api': '35', 'abi': subprocess.check_output(['adb', 'shell', 'getprop', 'ro.product.cpu.abi'], text=True).strip(),
        'fingerprint': subprocess.check_output(['adb', 'shell', 'getprop', 'ro.build.fingerprint'], text=True).strip(),
        'aar_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(Path(os.environ['RUNNER_TEMP']) / 'mlkit-runtime.json').write_text(json.dumps(data))
PY
/usr/bin/python3 mlkit_lab.py score --candidate "$CANDIDATE" --truth "$RUNNER_TEMP/mlkit-truth.json" --observations "$RUNNER_TEMP/mlkit-observations.json" --runtime "$RUNNER_TEMP/mlkit-runtime.json"

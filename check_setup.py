import shutil, subprocess

mods = ["pandas", "scipy", "mlxtend", "pydantic", "fastapi", "streamlit", "google.genai"]
for m in mods:
    try:
        __import__(m)
        print(f"[ok]      {m}")
    except ImportError:
        print(f"[MISSING] {m}")

try:
    from src.mining.huim import find_java
    java_bin = find_java()
except Exception:
    import os
    java_bin = "java" if shutil.which("java") else ("/opt/homebrew/opt/openjdk/bin/java" if os.path.exists("/opt/homebrew/opt/openjdk/bin/java") else None)

if java_bin:
    out = subprocess.run([java_bin, "-version"], capture_output=True, text=True)
    if out.returncode == 0:
        print("[ok]      java:", (out.stderr or out.stdout).splitlines()[0])
    else:
        print("[MISSING] java (needed for EFIM in Step 5)")
else:
    print("[MISSING] java (needed for EFIM in Step 5)")

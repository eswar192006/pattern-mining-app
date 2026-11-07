import shutil, subprocess

mods = ["pandas", "scipy", "mlxtend", "pydantic", "fastapi", "streamlit", "google.genai"]
for m in mods:
    try:
        __import__(m)
        print(f"[ok]      {m}")
    except ImportError:
        print(f"[MISSING] {m}")

if shutil.which("java"):
    out = subprocess.run(["java", "-version"], capture_output=True, text=True)
    print("[ok]      java:", (out.stderr or out.stdout).splitlines()[0])
else:
    print("[MISSING] java (needed for EFIM in Step 5)")

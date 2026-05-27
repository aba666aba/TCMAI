import zipfile, os
from pathlib import Path

ROOT = Path(r"E:\Tom\TCMAI")
OUT = ROOT / "dist" / "test_zip.zip"

count = 0
with zipfile.ZipFile(str(OUT), "w", zipfile.ZIP_DEFLATED) as zf:
    for dp, _, fs in os.walk(str(ROOT / "python")):
        dp = Path(dp)
        for f in fs:
            if count >= 500:
                break
            fp = dp / f
            rel = fp.relative_to(ROOT)
            zf.write(str(fp), str(rel.as_posix()))
            count += 1
    if count < 500:
        print("Only %d files found" % count)

sz = OUT.stat().st_size
print("OK: %d files, %.1f MB" % (count, sz / (1024 * 1024)))

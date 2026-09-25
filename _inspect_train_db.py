import os
import sqlite3

p = "data/indexes/s2s3.sqlite"
t = "data/indexes/test_s2s3.sqlite"
print("train_size_mb", os.path.getsize(p) / 1e6)
print("test_exists", os.path.isfile(t), "test_size_mb", os.path.getsize(t) / 1e6 if os.path.isfile(t) else None)
c = sqlite3.connect(p)
print("refs", c.execute("SELECT COUNT(*) FROM refs").fetchone()[0])
print("schema", c.execute("SELECT type, name FROM sqlite_master ORDER BY type, name").fetchall())
for tbl in ("inv_name", "inv_addr"):
    print(tbl, c.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0])
c.close()

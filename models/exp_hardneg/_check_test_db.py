import os
import sqlite3

p = "data/indexes/test_s2s3.sqlite"
print("exists", os.path.isfile(p))
print("mb", os.path.getsize(p) / 1e6)
c = sqlite3.connect(p)
print("refs", c.execute("SELECT COUNT(*) FROM refs").fetchone()[0])
print("indexes", [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='index'")])

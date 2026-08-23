import sys
from pathlib import Path
sys.path.insert(0, r"e:\TEAM ENTROPY")
from scripts.finetune_water import _build_water_pairs

pairs, st = _build_water_pairs(r"/tmp/water_test")
print("stats:", st)
assert st["n_sar"] == 5 and st["n_rgb"] == 4, st
assert st["n_pairs"] == 4 and st["unmatched_sar"] == 1 and st["unmatched_rgb"] == 0, st
assert st["dup_ids"] == 0, st
assert len(pairs) == 4
# every pair is the same stem with _s1/_s2
for p in pairs:
    a, b = Path(p[0]).name, Path(p[1]).name
    assert a[:-len("_s1.png")] == b[:-len("_s2.png")], (a, b)
    print(a, "<->", b)
# empty folder never touches pairs[0]
p2, st2 = _build_water_pairs(r"/tmp/nonexistent_dir_xyz")
assert p2 == [] and st2["n_pairs"] == 0, (p2, st2)
print("empty dir ->", st2, "(no IndexError)")
print("OK: water flat-folder pairer verified.")
"""Audit mendalam: over-alignment GRL, lintasan per-epoch, korelasi terkumpul."""
import re, sys, math
from pathlib import Path
from statistics import mean, stdev

sys.stdout.reconfigure(encoding="utf-8")

LOG = Path(sys.argv[1])
RE_METHOD = re.compile(r"^=== \[(\d+)/(\d+)\] (.+?) ===")
RE_SEED = re.compile(r">>> SEED (\d+) —")
RE_EPOCH = re.compile(
    r"\[GRL-EPOCH (\S+)\] λ=([\d.]+) \| domain_loss=([\d.]+)/([\d.]+) "
    r"\(alignment=(\d+)%\) \| domain_acc=([\d.]+) vs chance=([\d.]+)"
)
RE_TEST = re.compile(r"\[SEED (\d+) FOLD \d+\] UNSEEN-torque test metrics:")
RE_F1 = re.compile(r"Win-F1-macro\s*:\s*([\d.]+)%")

data, method, seed, await_f1 = {}, None, None, None
for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
    if (m := RE_METHOD.search(line)):
        method = m.group(3).strip(); data.setdefault(method, {}); seed = None; continue
    if (m := RE_SEED.search(line)):
        seed = m.group(1)
        if method: data[method].setdefault(seed, {"ep": [], "f1": None})
        continue
    if (m := RE_EPOCH.search(line)) and method and seed:
        data[method][seed]["ep"].append(dict(
            lam=float(m.group(2)), dloss=float(m.group(3)), maxloss=float(m.group(4)),
            dacc=float(m.group(6)), chance=float(m.group(7))))
        continue
    if (m := RE_TEST.search(line)):
        await_f1 = m.group(1); continue
    if await_f1 and (m := RE_F1.search(line)) and method:
        data[method].setdefault(await_f1, {"ep": [], "f1": None})
        data[method][await_f1]["f1"] = float(m.group(1)); await_f1 = None

ORD = ["42", "123", "2025", "7", "17", "2024", "2026", "88"]
try:
    from scipy import stats; SP = True
except Exception:
    SP = False


def pear(x, y):
    n = len(x); mx, my = mean(x), mean(y)
    num = sum((a-mx)*(b-my) for a, b in zip(x, y))
    den = math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
    r = num/den if den else float("nan")
    if not SP or n < 3 or abs(r) >= 1: return r, float("nan")
    t = r*math.sqrt((n-2)/(1-r*r))
    return r, 2*(1-stats.t.cdf(abs(t), n-2))


print("\n" + "="*100)
print("BAGIAN 1 — OVER-ALIGNMENT: apakah domain_loss melampaui batas teoretis log(n)?")
print("="*100)
print("domain_loss maksimum teoretis = log(6) = 1.7918 (prediktor seragam).")
print("Melampauinya = diskriminator SALAH DENGAN PERCAYA DIRI, bukan sekadar bingung.\n")
print(f"{'metode':38} {'ekor10 d_loss':>14} {'kelebihan':>10} {'ekor10 dacc':>12} {'vs chance':>10} {'n seed>max':>11}")
for meth, seeds in data.items():
    rows = [seeds[s] for s in ORD if s in seeds and seeds[s]["ep"]]
    if not rows: continue
    if rows[0]["ep"][-1]["lam"] == 0.0: continue          # lewati baseline
    maxloss = rows[0]["ep"][-1]["maxloss"]; ch = rows[0]["ep"][-1]["chance"]
    tl = [mean(e["dloss"] for e in r["ep"][-10:]) for r in rows]
    ta = [mean(e["dacc"] for e in r["ep"][-10:]) for r in rows]
    over = sum(1 for v in tl if v > maxloss)
    print(f"{meth:38} {mean(tl):>14.4f} {mean(tl)-maxloss:>+10.4f} "
          f"{mean(ta):>12.4f} {mean(ta)-ch:>+10.4f} {over:>8d}/{len(tl)}")

print("\n" + "="*100)
print("BAGIAN 2 — KAPAN domain_acc MENEMBUS chance (kecepatan diskriminator dikalahkan)")
print("="*100)
print(f"{'metode':32} {'seed':>6} {'Win-F1':>8} {'epoch tembus':>13} {'dacc@ep5':>10} {'dacc@ep10':>11}")
speed = {}
for meth, seeds in data.items():
    rows = [(s, seeds[s]) for s in ORD if s in seeds and seeds[s]["ep"]]
    if not rows or rows[0][1]["ep"][-1]["lam"] == 0.0: continue
    ch = rows[0][1]["ep"][-1]["chance"]
    speed[meth] = []
    for s, r in rows:
        eps = r["ep"]
        cross = next((i for i, e in enumerate(eps) if e["dacc"] < ch), None)
        d5 = eps[4]["dacc"] if len(eps) > 4 else float("nan")
        d10 = eps[9]["dacc"] if len(eps) > 9 else float("nan")
        speed[meth].append((s, r["f1"], cross, d5, d10))
        print(f"{meth:32} {s:>6} {r['f1']:>8.2f} "
              f"{(str(cross) if cross is not None else 'tak pernah'):>13} {d5:>10.3f} {d10:>11.3f}")
    print()

print("="*100)
print("BAGIAN 3 — KORELASI TERKUMPUL lintas varian DANN (dipusatkan per-metode)")
print("="*100)
print("Setiap varian dipusatkan pada rata-ratanya sendiri, lalu digabung.")
print("Menghilangkan efek utama λ, menyisakan hubungan dalam-varian.\n")
pool_d, pool_f, pool_dev, pool_cross = [], [], [], []
for meth, seeds in data.items():
    rows = [seeds[s] for s in ORD if s in seeds and seeds[s]["ep"]]
    if not rows or rows[0]["ep"][-1]["lam"] == 0.0: continue
    if "ELS" in meth: continue                     # ELS mengubah loss domain, tidak sebanding
    ch = rows[0]["ep"][-1]["chance"]
    d = [mean(e["dacc"] for e in r["ep"][-10:]) for r in rows]
    f = [r["f1"] for r in rows]
    cr = [next((i for i, e in enumerate(r["ep"]) if e["dacc"] < ch), len(r["ep"])) for r in rows]
    md, mf, mc = mean(d), mean(f), mean(cr)
    pool_d += [x-md for x in d]; pool_f += [x-mf for x in f]
    pool_dev += [abs(x-ch)-mean(abs(y-ch) for y in d) for x in d]
    pool_cross += [x-mc for x in cr]

n = len(pool_d)
for lbl, v in [("domain_acc ekor-10", pool_d),
               ("|domain_acc - chance|", pool_dev),
               ("epoch tembus chance", pool_cross)]:
    r, p = pear(v, pool_f)
    star = "  <-- SIGNIFIKAN" if p == p and p < 0.05 else ""
    print(f"  {lbl:26} vs Win-F1 :  r={r:+.3f}  p={p:.4f}  (n={n}){star}")

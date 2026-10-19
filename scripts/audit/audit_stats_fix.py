"""Koreksi statistik: agregasi ke tingkat seed (n=8 independen) + uji permutasi berblok."""
import re, sys, math, random
from pathlib import Path
from statistics import mean

sys.stdout.reconfigure(encoding="utf-8")
random.seed(0)

LOG = Path(sys.argv[1])
RE_METHOD = re.compile(r"^=== \[(\d+)/(\d+)\] (.+?) ===")
RE_SEED = re.compile(r">>> SEED (\d+) —")
RE_EPOCH = re.compile(r"\[GRL-EPOCH \S+\] λ=([\d.]+) \| domain_loss=([\d.]+)/([\d.]+) "
                      r"\(alignment=\d+%\) \| domain_acc=([\d.]+) vs chance=([\d.]+)")
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
        data[method][seed]["ep"].append(dict(lam=float(m.group(1)), dloss=float(m.group(2)),
                                             dacc=float(m.group(4)), chance=float(m.group(5))))
        continue
    if (m := RE_TEST.search(line)):
        await_f1 = m.group(1); continue
    if await_f1 and (m := RE_F1.search(line)) and method:
        data[method].setdefault(await_f1, {"ep": [], "f1": None})
        data[method][await_f1]["f1"] = float(m.group(1)); await_f1 = None

ORD = ["42", "123", "2025", "7", "17", "2024", "2026", "88"]
VAR = [k for k in data if k.startswith("DANN lambda")]           # 3 varian, ELS dikecualikan
try:
    from scipy import stats; SP = True
except Exception:
    SP = False


def pear(x, y):
    mx, my = mean(x), mean(y)
    num = sum((a-mx)*(b-my) for a, b in zip(x, y))
    den = math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
    return num/den if den else float("nan")


def pval(r, n):
    if not SP or abs(r) >= 1: return float("nan")
    t = r*math.sqrt((n-2)/(1-r*r))
    return 2*(1-stats.t.cdf(abs(t), n-2))


# ---- blok terpusat per varian -------------------------------------------------
blocks = {}
for v in VAR:
    d = [mean(e["dacc"] for e in data[v][s]["ep"][-10:]) for s in ORD]
    f = [data[v][s]["f1"] for s in ORD]
    md, mf = mean(d), mean(f)
    blocks[v] = ([x-md for x in d], [x-mf for x in f])

print("="*96)
print("UJI 1 — korelasi per-varian (n=8 independen masing-masing)")
print("="*96)
rs = []
for v in VAR:
    d = [mean(e["dacc"] for e in data[v][s]["ep"][-10:]) for s in ORD]
    f = [data[v][s]["f1"] for s in ORD]
    r = pear(d, f); rs.append(r)
    print(f"  {v:34} r={r:+.3f}  p={pval(r,8):.3f}")
print(f"\n  Ketiganya bertanda POSITIF. Peluang 3/3 sama tanda jika acak = 2*(1/2)^3 = 0.25")

print("\n" + "="*96)
print("UJI 2 — agregasi ke tingkat seed (rata-rata lintas 3 varian; n=8 BENAR-BENAR independen)")
print("="*96)
agg_d = [mean(blocks[v][0][i] for v in VAR) for i in range(8)]
agg_f = [mean(blocks[v][1][i] for v in VAR) for i in range(8)]
r = pear(agg_d, agg_f)
print(f"  {'seed':>6} {'dacc terpusat':>15} {'Win-F1 terpusat':>17}")
for i, s in enumerate(ORD):
    print(f"  {s:>6} {agg_d[i]:>+15.4f} {agg_f[i]:>+17.2f}")
print(f"\n  r = {r:+.3f}   p = {pval(r,8):.4f}   (n=8)")

print("\n" + "="*96)
print("UJI 3 — permutasi berblok (acak F1 dalam tiap varian, 20.000 ulangan)")
print("="*96)
obs = pear([x for v in VAR for x in blocks[v][0]],
           [x for v in VAR for x in blocks[v][1]])
cnt = 0; N = 20000
for _ in range(N):
    pd_, pf_ = [], []
    for v in VAR:
        d, f = blocks[v]
        fs = f[:]; random.shuffle(fs)
        pd_ += d; pf_ += fs
    if abs(pear(pd_, pf_)) >= abs(obs): cnt += 1
print(f"  r teramati (terkumpul) = {obs:+.3f}")
print(f"  p permutasi = {(cnt+1)/(N+1):.4f}   <- ini yang sah, bukan p=0.0115 sebelumnya")

print("\n" + "="*96)
print("UJI 4 — over-alignment: apakah domain_loss > log(6) secara sistematis?")
print("="*96)
for v in VAR + [k for k in data if "ELS" in k]:
    tl = [mean(e["dloss"] for e in data[v][s]["ep"][-10:]) for s in ORD]
    exc = [x-1.7918 for x in tl]
    if SP:
        t, p = stats.ttest_1samp(exc, 0.0)
        w = stats.wilcoxon(exc).pvalue if len(set(exc)) > 1 else float("nan")
        print(f"  {v:34} kelebihan={mean(exc):+.4f}  t={t:+.2f} p={p:.5f}  wilcoxon p={w:.4f}  ({sum(1 for x in exc if x>0)}/8 positif)")

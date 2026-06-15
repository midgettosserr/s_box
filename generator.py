import random
import time
import sys
import math
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# GF(2^8)  AES primitive polynomial 0x11B
# ─────────────────────────────────────────────
POLY = 0x11B


def gf_mul(a: int, b: int) -> int:
    res = 0
    for _ in range(8):
        if b & 1:
            res ^= a
        msb = a & 0x80
        a = (a << 1) & 0xFF
        if msb:
            a ^= 0x1B
        b >>= 1
    return res


# Быстрая таблица обратных
GF_INV = [0] * 256
for x in range(1, 256):
    for y in range(1, 256):
        if gf_mul(x, y) == 1:
            GF_INV[x] = y
            break


# ─────────────────────────────────────────────
# GF(2)^8 affine transform
# ─────────────────────────────────────────────
def mat_vec_mul(mat: list, vec: int) -> int:
    result = 0
    for i, row in enumerate(mat):
        bit = (row & vec).bit_count() & 1
        result |= (bit << i)
    return result


def is_invertible(mat: list) -> bool:
    m = mat[:]
    for col in range(8):
        pivot = -1
        for row in range(col, 8):
            if (m[row] >> col) & 1:
                pivot = row
                break
        if pivot == -1:
            return False
        m[col], m[pivot] = m[pivot], m[col]
        for row in range(8):
            if row != col and ((m[row] >> col) & 1):
                m[row] ^= m[col]
    return True


def random_invertible_matrix() -> list:
    while True:
        mat = [random.randint(1, 255) for _ in range(8)]
        if is_invertible(mat):
            return mat


def build_aes_matrix():
    rows = []
    for i in range(8):
        row = 0
        for bit in [0, 4, 5, 6, 7]:
            row |= (1 << ((i + bit) % 8))
        rows.append(row)
    return rows


AES_MAT = build_aes_matrix()
AES_CONST = 0x63


def affine_sbox(mat: list, const: int) -> list:
    result = []
    for x in range(256):
        inv_x = GF_INV[x]
        y = mat_vec_mul(mat, inv_x) ^ const
        result.append(y & 0xFF)
    return result


def verify_bijective(sbox: list) -> bool:
    return len(set(sbox)) == 256


# ─────────────────────────────────────────────
# Exact metrics
# ─────────────────────────────────────────────
def walsh_transform(f: list) -> list:
    n = 256
    W = [1 - 2 * b for b in f]
    step = 1
    while step < n:
        for i in range(0, n, step * 2):
            for j in range(i, i + step):
                u = W[j]
                v = W[j + step]
                W[j] = u + v
                W[j + step] = u - v
        step <<= 1
    return W


def nonlinearity_and_bias(sbox: list) -> tuple:
    max_corr = 0
    for b_mask in range(1, 256):
        f_vals = [((b_mask & sbox[x]).bit_count() & 1) for x in range(256)]
        W = walsh_transform(f_vals)
        local_max = max(abs(w) for w in W)
        if local_max > max_corr:
            max_corr = local_max
    nl = 128 - max_corr // 2
    bias = max_corr // 2
    return nl, bias


def diff_uniformity_full(sbox: list) -> int:
    max_du = 0
    for a in range(1, 256):
        counts = {}
        for x in range(256):
            dy = sbox[x] ^ sbox[x ^ a]
            counts[dy] = counts.get(dy, 0) + 1
        local_max = max(counts.values())
        if local_max > max_du:
            max_du = local_max
        if max_du > 8:
            return max_du
    return max_du


def avalanche_effect(sbox: list) -> float:
    flips = 0
    for x in range(256):
        sx = sbox[x]
        for bit in range(8):
            x2 = x ^ (1 << bit)
            flips += (sx ^ sbox[x2]).bit_count()
    return flips / (256 * 8 * 8)


# ─────────────────────────────────────────────
# Quality + diversity
# ─────────────────────────────────────────────
def fitness(nl: int, av: float, bias: int, du: int) -> float:
    du_penalty = 0 if du <= 4 else 150 * (du - 4)   # чуть мягче
    av_penalty = abs(av - 0.5)
    return (
        4.0 * nl
        - 0.16 * (bias ** 1.15)
        - 220.0 * av_penalty
        - du_penalty
    )


def quality_gate(m: dict) -> bool:
    # ослабили пороги, чтобы было больше кандидатов
    return (
        m["nl"] >= 100 and
        m["du"] <= 8 and
        m["bias"] <= 36 and
        0.45 <= m["av"] <= 0.55
    )


def metrics_vector(m: dict):
    return (m["nl"], m["du"], m["bias"], m["av"])


def metrics_distance(m1: dict, m2: dict) -> float:
    d_nl = abs(m1["nl"] - m2["nl"]) / 12.0
    d_du = abs(m1["du"] - m2["du"]) / 6.0
    d_bias = abs(m1["bias"] - m2["bias"]) / 32.0
    d_av = abs(m1["av"] - m2["av"]) / 0.05
    return math.sqrt(d_nl * d_nl + d_du * d_du + d_bias * d_bias + d_av * d_av)


def too_similar(m: dict, stored_values, min_dist: float = 0.45) -> bool:
    for _, sm in stored_values:
        if metrics_distance(m, sm) < min_dist:
            return True
    return False


# ─────────────────────────────────────────────
# Mutations
# ─────────────────────────────────────────────
def mutate_swap(sbox: list, n_swaps: int = 3) -> list:
    s = sbox[:]
    for _ in range(n_swaps):
        i, j = random.sample(range(256), 2)
        s[i], s[j] = s[j], s[i]
    return s


def mutate_segment_reverse(sbox: list) -> list:
    s = sbox[:]
    a, b = sorted(random.sample(range(256), 2))
    s[a:b + 1] = s[a:b + 1][::-1]
    return s


def mutate_cyclic_shift(sbox: list) -> list:
    s = sbox[:]
    a = random.randint(0, 240)
    length = random.randint(4, 16)
    segment = s[a:a + length]
    k = random.randint(1, len(segment) - 1)
    s[a:a + length] = segment[k:] + segment[:k]
    return s


def mutate_scramble_segment(sbox: list) -> list:
    s = sbox[:]
    a = random.randint(0, 235)
    length = random.randint(5, 20)
    seg = s[a:a + length]
    random.shuffle(seg)
    s[a:a + length] = seg
    return s


def mutate_double_swap_far(sbox: list) -> list:
    s = sbox[:]
    pairs = [
        (random.randint(0, 63), random.randint(192, 255)),
        (random.randint(64, 127), random.randint(128, 191)),
    ]
    for i, j in pairs:
        s[i], s[j] = s[j], s[i]
    return s


# ─────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────
def generate_affine_pool(size: int) -> list:
    pool = []
    aes = affine_sbox(AES_MAT, AES_CONST)
    pool.append(aes)
    print(f"  [init] AES bijective: {verify_bijective(aes)}")

    attempts = 0
    while len(pool) < size:
        mat = random_invertible_matrix()
        const = random.randint(0, 255)
        sbox = affine_sbox(mat, const)
        if verify_bijective(sbox):
            pool.append(sbox)
        attempts += 1
        if attempts > size * 30:
            print(f"  [warn] pool generation slow at {len(pool)}/{size}")
            break

    return pool


# ─────────────────────────────────────────────
# Metrics cache
# ─────────────────────────────────────────────
_metric_cache = {}


def compute_metrics(sbox: list) -> dict:
    key = tuple(sbox)
    if key in _metric_cache:
        return _metric_cache[key]

    du = diff_uniformity_full(sbox)
    av = avalanche_effect(sbox)
    nl, bias = nonlinearity_and_bias(sbox)
    sc = fitness(nl, av, bias, du)

    m = {
        "nl": nl,
        "bias": bias,
        "du": du,
        "av": av,
        "score": sc,
    }
    _metric_cache[key] = m
    return m


# ─────────────────────────────────────────────
# Diversity-aware storage
# ─────────────────────────────────────────────
def try_add_to_storage(storage: dict, sbox: list, m: dict,
                       max_size: int = 40,
                       min_dist: float = 0.55):
    key = tuple(sbox)
    if key in storage:
        return

    if not quality_gate(m):
        return

    values = list(storage.values())

    if len(values) < max_size:
        if not too_similar(m, values, min_dist=min_dist):
            storage[key] = (sbox, m)
        else:
            closest_key = None
            closest_dist = 1e9
            for k, (_, sm) in storage.items():
                d = metrics_distance(m, sm)
                if d < closest_dist:
                    closest_dist = d
                    closest_key = k
            if closest_key is not None and m["score"] > storage[closest_key][1]["score"] + 0.25:
                del storage[closest_key]
                storage[key] = (sbox, m)
        return

    closest_key = None
    closest_dist = 1e9
    for k, (_, sm) in storage.items():
        d = metrics_distance(m, sm)
        if d < closest_dist:
            closest_dist = d
            closest_key = k

    if closest_dist >= min_dist:
        worst_key = min(storage, key=lambda k: storage[k][1]["score"])
        if m["score"] > storage[worst_key][1]["score"]:
            del storage[worst_key]
            storage[key] = (sbox, m)
    else:
        if closest_key is not None and m["score"] > storage[closest_key][1]["score"] + 0.25:
            del storage[closest_key]
            storage[key] = (sbox, m)


def select_elites_diverse(evaluated, elite_size: int):
    chosen = []
    chosen_metrics = []

    for s, m in sorted(evaluated, key=lambda x: x[1]["score"], reverse=True):
        if len(chosen) == 0:
            chosen.append(s)
            chosen_metrics.append(m)
            if len(chosen) >= elite_size:
                break
            continue

        good_gap = True
        for cm in chosen_metrics:
            if metrics_distance(m, cm) < 0.35:
                good_gap = False
                break

        if good_gap or len(chosen) < elite_size // 2:
            chosen.append(s)
            chosen_metrics.append(m)

        if len(chosen) >= elite_size:
            break

    i = 0
    while len(chosen) < elite_size and i < len(evaluated):
        s, _ = evaluated[i]
        if tuple(s) not in {tuple(x) for x in chosen}:
            chosen.append(s)
        i += 1

    return chosen


# ─────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────
def label_du(d):
    if d == 4:
        return "🟢 excellent (AES-level)"
    elif d <= 6:
        return "🟡 good"
    else:
        return "🔴 weak"


def print_sbox(sbox):
    print("\nS-BOX (16x16 view):\n")
    for i in range(0, 256, 16):
        row = sbox[i:i + 16]
        print(" ".join(f"{x:3d}" for x in row))
    print()


def print_candidate(rank, s, m):
    print(f"\nRank {rank}")
    print(f"  Score:        {m['score']:.2f}")
    print(f"  Nonlinearity: {m['nl']}")
    print(f"  Diff uniform: {m['du']} → {label_du(m['du'])}")
    print(f"  Linear bias:  {m['bias']}")
    print(f"  Avalanche:    {m['av']:.4f}  (ideal=0.5)")
    print_sbox(s)


# ─────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────
POP_SIZE = 64
GENERATIONS = 220
ELITE_SIZE = 16
TARGET_STORAGE = 40
RESTART_EVERY = 35
DIVERSITY_MIN_DIST = 0.45  # было 0.55
RANDOM_SEED = 42

FINAL_TOP_K = 10


random.seed(RANDOM_SEED)


print("=" * 72)
print("S-BOX EVOLUTIONARY SEARCH WITH DIVERSITY-PRESERVING STORAGE")
print("=" * 72)
print(f"Population: {POP_SIZE} | Generations: {GENERATIONS} | Elite: {ELITE_SIZE}")
print(f"Target storage: {TARGET_STORAGE} | Diversity min dist: {DIVERSITY_MIN_DIST}")
print()


# ─────────────────────────────────────────────
# Init
# ─────────────────────────────────────────────
print("[init] Generating affine pool...")
t0 = time.time()
population = generate_affine_pool(POP_SIZE)
print(f"[init] Pool ready in {time.time() - t0:.1f}s, size={len(population)}")
print("[init] Computing initial metrics...")


evaluated = []
for i, s in enumerate(population):
    m = compute_metrics(s)
    evaluated.append((s, m))
    if (i + 1) % 8 == 0 or i + 1 == len(population):
        print(f"  {i + 1}/{len(population)} done...")


evaluated.sort(key=lambda x: x[1]["score"], reverse=True)
best_init = evaluated[0][1]
print(
    f"\n[init] Best initial: "
    f"NL={best_init['nl']}, DU={best_init['du']}, "
    f"bias={best_init['bias']}, av={best_init['av']:.4f}, score={best_init['score']:.2f}"
)
print()


# ─────────────────────────────────────────────
# Storage + history
# ─────────────────────────────────────────────
storage = {}
for s, m in evaluated:
    try_add_to_storage(
        storage, s, m,
        max_size=TARGET_STORAGE,
        min_dist=DIVERSITY_MIN_DIST
    )

best_score_ever = evaluated[0][1]["score"]
stagnation = 0
history = []   # ← сюда пишем динамику


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────
for gen in range(GENERATIONS):
    evaluated.sort(key=lambda x: x[1]["score"], reverse=True)
    elites = select_elites_diverse(evaluated, ELITE_SIZE)

    new_pop = elites[:]

    while len(new_pop) < POP_SIZE:
        parent = random.choice(elites)
        r = random.random()

        if r < 0.28:
            child = mutate_swap(parent, n_swaps=random.randint(1, 5))
        elif r < 0.48:
            child = mutate_segment_reverse(parent)
        elif r < 0.68:
            child = mutate_cyclic_shift(parent)
        elif r < 0.84:
            child = mutate_scramble_segment(parent)
        else:
            child = mutate_double_swap_far(parent)

        if verify_bijective(child):
            new_pop.append(child)

    if stagnation > 0 and stagnation % RESTART_EVERY == 0:
        n_inject = 12
        fresh = generate_affine_pool(n_inject)
        new_pop[-n_inject:] = fresh
        print(f"  [restart] Injected {n_inject} fresh affine bases at gen {gen}")

    evaluated = []
    for s in new_pop:
        m = compute_metrics(s)
        evaluated.append((s, m))

    evaluated.sort(key=lambda x: x[1]["score"], reverse=True)
    best = evaluated[0]
    best_m = best[1]
    avg_score = sum(e[1]["score"] for e in evaluated) / len(evaluated)
    avg_nl = sum(e[1]["nl"] for e in evaluated) / len(evaluated)
    avg_du = sum(e[1]["du"] for e in evaluated) / len(evaluated)
    avg_bias = sum(e[1]["bias"] for e in evaluated) / len(evaluated)
    avg_av = sum(e[1]["av"] for e in evaluated) / len(evaluated)

    for s, m in evaluated[:24]:
        try_add_to_storage(
            storage, s, m,
            max_size=TARGET_STORAGE,
            min_dist=DIVERSITY_MIN_DIST
        )

    if best_m["score"] > best_score_ever + 0.01:
        best_score_ever = best_m["score"]
        stagnation = 0
    else:
        stagnation += 1

    history.append({
        "gen": gen,
        "best_score": best_m["score"],
        "avg_score": avg_score,
        "best_nl": best_m["nl"],
        "avg_nl": avg_nl,
        "best_du": best_m["du"],
        "avg_du": avg_du,
        "best_bias": best_m["bias"],
        "avg_bias": avg_bias,
        "best_av": best_m["av"],
        "avg_av": avg_av,
        "stored": len(storage),
        "stag": stagnation,
    })

    print(
        f"Gen {gen:03d} | score={best_m['score']:7.2f} | "
        f"NL={best_m['nl']:3d} | DU={best_m['du']} | "
        f"bias={best_m['bias']:3d} | av={best_m['av']:.4f} | "
        f"stored={len(storage):2d} | stag={stagnation:3d}"
    )
    sys.stdout.flush()


# ─────────────────────────────────────────────
# Final analysis
# ─────────────────────────────────────────────
print("\n" + "=" * 72)
print("TOP DIVERSE S-BOXES (FINAL ANALYSIS)")
print("=" * 72)

# Жёсткая чистка storage от дублей по содержимому S-box
clean_storage = {}
for s, m in storage.values():
    key = tuple(s)
    if key not in clean_storage:
        clean_storage[key] = (s, m)
    else:
        # если вдруг дубль всё же встретился — оставляем более сильный
        if m["score"] > clean_storage[key][1]["score"]:
            clean_storage[key] = (s, m)

storage = clean_storage

all_stored = list(storage.values())
all_stored.sort(
    key=lambda x: (
        x[1]["du"],
        -x[1]["nl"],
        x[1]["bias"],
        abs(x[1]["av"] - 0.5),
        -x[1]["score"]
    )
)

print(f"\nUnique diverse high-quality S-boxes stored: {len(all_stored)}")

# 1) Основные кандидаты из storage
best_candidates = []
seen_keys = set()

for s, m in all_stored:
    key = tuple(s)
    if key in seen_keys:
        continue
    seen_keys.add(key)
    best_candidates.append((s, m))
    if len(best_candidates) >= FINAL_TOP_K:
        break

# 2) Если storage дал меньше FINAL_TOP_K, добираем из evaluated
if len(best_candidates) < FINAL_TOP_K:
    print(f"[info] Only {len(best_candidates)} unique candidates in storage, filling from current population...")
    evaluated_sorted = sorted(
        evaluated,
        key=lambda x: (
            x[1]["du"],
            -x[1]["nl"],
            x[1]["bias"],
            abs(x[1]["av"] - 0.5),
            -x[1]["score"]
        )
    )

    for s, m in evaluated_sorted:
        key = tuple(s)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        best_candidates.append((s, m))
        if len(best_candidates) >= FINAL_TOP_K:
            break

# 3) Если даже после добора всё ещё мало — просто честно покажем сколько есть
print(f"[final] Showing {len(best_candidates)} final unique S-boxes")

for i, (s, m) in enumerate(best_candidates, 1):
    print_candidate(i, s, m)


# ─────────────────────────────────────────────
# Grouped view by metric profile
# ─────────────────────────────────────────────
print("\n" + "=" * 72)
print("PROFILE GROUPS")
print("=" * 72)

groups = {}
for s, m in best_candidates:
    key = (m["nl"], m["du"])
    groups.setdefault(key, []).append((s, m))

for key in sorted(groups.keys(), key=lambda k: (-k[0], k[1])):
    group = sorted(
        groups[key],
        key=lambda x: (x[1]["bias"], abs(x[1]["av"] - 0.5), -x[1]["score"])
    )
    print(f"\nNL={key[0]}, DU={key[1]} | count={len(group)}")
    for _, m in group:
        print(
            f"  score={m['score']:.2f}, bias={m['bias']:2d}, "
            f"av={m['av']:.4f}"
        )


# ─────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────
with open("diverse_sboxes_result.txt", "w", encoding="utf-8") as f:
    f.write("DIVERSE HIGH-QUALITY S-BOXES\n")
    f.write("=" * 72 + "\n\n")
    f.write(f"Population: {POP_SIZE}\n")
    f.write(f"Generations: {GENERATIONS}\n")
    f.write(f"Elite size: {ELITE_SIZE}\n")
    f.write(f"Target storage: {TARGET_STORAGE}\n")
    f.write(f"Diversity min distance: {DIVERSITY_MIN_DIST}\n")
    f.write(f"Requested final top K: {FINAL_TOP_K}\n")
    f.write(f"Stored unique candidates: {len(all_stored)}\n")
    f.write(f"Final shown candidates: {len(best_candidates)}\n")
    f.write(f"Cache entries computed: {len(_metric_cache)}\n\n")

    for i, (s, m) in enumerate(best_candidates, 1):
        f.write(f"Rank {i}\n")
        f.write(f"  Score:        {m['score']:.2f}\n")
        f.write(f"  Nonlinearity: {m['nl']}\n")
        f.write(f"  Diff uniform: {m['du']}\n")
        f.write(f"  Linear bias:  {m['bias']}\n")
        f.write(f"  Avalanche:    {m['av']:.4f}\n")
        f.write(f"  Vector:       {metrics_vector(m)}\n")
        f.write("  S-BOX flat:\n")
        f.write(f"  {s}\n")
        f.write("  S-BOX 16x16:\n")
        for j in range(0, 256, 16):
            row = s[j:j + 16]
            f.write("  " + " ".join(f"{x:3d}" for x in row) + "\n")
        f.write("\n")

print("\n[saved] Results → diverse_sboxes_result.txt")
print(f"Cache entries computed: {len(_metric_cache)}")


# ─────────────────────────────────────────────
# Visualization with matplotlib
# ─────────────────────────────────────────────
if history:
    gens = [h["gen"] for h in history]
    best_score = [h["best_score"] for h in history]
    avg_score = [h["avg_score"] for h in history]
    best_nl = [h["best_nl"] for h in history]
    avg_nl = [h["avg_nl"] for h in history]
    best_du = [h["best_du"] for h in history]
    avg_du = [h["avg_du"] for h in history]
    best_bias = [h["best_bias"] for h in history]
    avg_bias = [h["avg_bias"] for h in history]
    best_av = [h["best_av"] for h in history]
    avg_av = [h["avg_av"] for h in history]
    stored_counts = [h["stored"] for h in history]
    stag_vals = [h["stag"] for h in history]

    # 1. Score dynamics
    plt.figure(figsize=(9, 4.5))
    plt.plot(gens, best_score, label="best score", linewidth=2)
    plt.plot(gens, avg_score, label="avg score", linewidth=1.8)
    plt.xlabel("generation")
    plt.ylabel("score")
    plt.title("Score dynamics over generations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("score_dynamics.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. NL and DU dynamics
    plt.figure(figsize=(9, 4.5))
    plt.plot(gens, best_nl, label="best NL", linewidth=2)
    plt.plot(gens, avg_nl, label="avg NL", linewidth=1.8)
    plt.plot(gens, best_du, label="best DU", linewidth=2)
    plt.plot(gens, avg_du, label="avg DU", linewidth=1.8)
    plt.xlabel("generation")
    plt.ylabel("value")
    plt.title("NL and DU dynamics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("nl_du_dynamics.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Bias and avalanche dynamics
    plt.figure(figsize=(9, 4.5))
    plt.plot(gens, best_bias, label="best bias", linewidth=2)
    plt.plot(gens, avg_bias, label="avg bias", linewidth=1.8)
    plt.plot(gens, best_av, label="best av", linewidth=2)
    plt.plot(gens, avg_av, label="avg av", linewidth=1.8)
    plt.xlabel("generation")
    plt.ylabel("value")
    plt.title("Bias and avalanche dynamics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("bias_av_dynamics.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Storage / stagnation dynamics
    plt.figure(figsize=(9, 4.5))
    plt.plot(gens, stored_counts, label="stored unique", linewidth=2)
    plt.plot(gens, stag_vals, label="stagnation", linewidth=1.8)
    plt.xlabel("generation")
    plt.ylabel("count")
    plt.title("Storage and stagnation dynamics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("storage_stagnation_dynamics.png", dpi=150, bbox_inches="tight")
    plt.close()

print("Done.")

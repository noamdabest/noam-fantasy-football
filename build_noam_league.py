#!/usr/bin/env python3
"""
NOAM league builder
===================
Reads your single-file app (Rest-of-Season_Projections_v3.html), pulls the 2026 nflverse data,
and writes a new HTML file whose `const D` holds the whole league (QB/RB/WR/TE).

    python build_noam_league.py --in Rest-of-Season_Projections_v3.html --out Rest-of-Season_Projections_v4.html

Safe to re-run any time (for example every day): players added by an earlier run are rebuilt
from fresh data, and your original players are never changed (only extra data fields are added).

Data: nflverse (CC-BY-4.0). Uses `nflreadpy` when it is installed, otherwise downloads the same
files straight from the nflverse GitHub releases. Files are cached in ./nflverse_cache
(use --refresh to download again).

What is exact and what is estimated
-----------------------------------
* Your original players (entries without "x":1): projections, ranges, matchups are kept exactly as they are.
* Added players ("x":1): projected with the documented NOAM method, re-created from the data:
    base PPG = 70% season PPG + 30% last-3 PPG, each first divided by the strength of the defenses
    faced (season PPG alone when there is no last-3 data); each upcoming week x defense multiplier
    for that position (capped about +/-12%); the first projected week (Week 11) also x a Vegas-line factor; byes from the real
    schedule; injuries from the nflverse injury report and roster status.
  This is calibrated against your original players but is NOT your original engine, so the numbers
  are close, not identical. The printed calibration report shows how close.
"""
import argparse, json, math, os, re, sys, urllib.request
import numpy as np
import pandas as pd

POS = ["QB", "RB", "WR", "TE"]
BASE = "https://github.com/nflverse/nflverse-data/releases/download"
URL = {
    "stats":    BASE + "/stats_player/stats_player_week_{s}.csv",
    "injuries": BASE + "/injuries/injuries_{s}.csv",
    "snaps":    BASE + "/snap_counts/snap_counts_{s}.csv",
    "roster":   BASE + "/rosters/roster_{s}.csv",
    "games":    BASE + "/schedules/games.csv",
}
TEAMFIX = {"JAX": "JAC", "LA": "LAR", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LAR"}  # nflverse -> page codes
SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}
BYE = ["BYE", 0.0, 1, 0, 0.0, 0, None, None]


def nt(t):
    return TEAMFIX.get(t, t) if isinstance(t, str) else t


def nname(n):
    s = re.sub(r"[.'’\-]", " ", str(n).lower()).split()
    while len(s) > 1 and s[-1] in SUFFIX:
        s.pop()
    return re.sub(r"[^a-z]", "", "".join(s))


def fetch(kind, season, cache, refresh=False, force_csv=False):
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"{kind}_{season}.csv")
    if not refresh and os.path.exists(path):
        return pd.read_csv(path, low_memory=False), "cache"
    df, src = None, None
    if not force_csv:
        try:
            import nflreadpy as nfl
            f = {"stats": lambda: nfl.load_player_stats(season), "games": lambda: nfl.load_schedules(season),
                 "injuries": lambda: nfl.load_injuries(season), "snaps": lambda: nfl.load_snap_counts(season),
                 "roster": lambda: nfl.load_rosters(season)}[kind]
            df, src = f().to_pandas(), "nflreadpy"
        except Exception:
            df = None
    if df is None:
        url = URL[kind].format(s=season)
        with urllib.request.urlopen(url, timeout=120) as r:
            df = pd.read_csv(r, low_memory=False)
        src = "nflverse release"
    if kind == "games" and "season" in df.columns:
        df = df[df.season == season]
    df.to_csv(path, index=False)
    return df, src


def clean(o):
    """make numpy / NaN safe for JSON"""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (o != o or math.isinf(o)) else float(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--cache", default="nflverse_cache")
    ap.add_argument("--refresh", action="store_true", help="download the nflverse files again")
    ap.add_argument("--csv-only", action="store_true", help="skip nflreadpy and use the release files")
    a = ap.parse_args()
    S = a.season

    html = open(a.inp, encoding="utf8").read()
    m = re.search(r"const D=(\[.*?\]),W=(\[.*?\]),P=", html, re.S)
    if not m:
        sys.exit("Could not find 'const D=[...],W=[...],P=' in the HTML file.")
    D_all, W = json.loads(m.group(1)), json.loads(m.group(2))
    orig = [p for p in D_all if not p.get("x")]
    print(f"HTML: {len(orig)} original players kept as-is, weeks {W[0]}-{W[-1]}")

    # ---------- data ----------
    src = {}
    stats, src["stats"] = fetch("stats", S, a.cache, a.refresh, a.csv_only)
    games, src["games"] = fetch("games", S, a.cache, a.refresh, a.csv_only)
    inj, src["injuries"] = fetch("injuries", S, a.cache, a.refresh, a.csv_only)
    roster, src["roster"] = fetch("roster", S, a.cache, a.refresh, a.csv_only)
    snaps, src["snaps"] = fetch("snaps", S, a.cache, a.refresh, a.csv_only)
    print("data sources:", src)

    st = stats[stats.position.isin(POS)].copy()
    if "season_type" in st.columns:
        st = st[st.season_type == "REG"]
    st["team"], st["opponent_team"] = st.team.map(nt), st.opponent_team.map(nt)
    last_wk = int(st.week.max())
    win_weeks = list(range(last_wk - 2, last_wk + 1))
    print(f"stats through Week {last_wk} (last-3 window = weeks {win_weeks[0]}-{win_weeks[-1]})")

    games = games[(games.season == S) & (games.game_type == "REG")].copy()
    games["home_team"], games["away_team"] = games.home_team.map(nt), games.away_team.map(nt)
    teams = sorted(set(games.home_team) | set(games.away_team))

    # ---------- schedule, byes ----------
    sched = {t: {} for t in teams}
    for _, g in games[games.week.isin(W)].iterrows():
        sched[g.home_team][g.week] = (g.away_team, 1)
        sched[g.away_team][g.week] = (g.home_team, 0)

    # ---------- defense tables (points allowed by position) ----------
    mine = st.groupby(["opponent_team", "position", "week"]).fantasy_points_ppr.sum().groupby(level=[0, 1]).mean()
    known = {}
    for p in orig:
        for x in p["w"]:
            if x[0] != "BYE" and x[6] is not None:
                known[(p["p"], x[0])] = (x[2], x[6], x[7])
    fit_m, fit_a = {}, {}
    for pos in POS:
        pr = [(mine[(o, pos)], v[0], v[2]) for (ps, o), v in known.items() if ps == pos and (o, pos) in mine.index]
        xs, ym, ya = zip(*pr)
        fit_m[pos] = np.polyfit(xs, ym, 1)
        fit_a[pos] = np.polyfit(xs, ya, 1)
    table = {}
    for pos in POS:
        est_allow = {o: (known[(pos, o)][2] if (pos, o) in known else float(np.polyval(fit_a[pos], mine.get((o, pos), np.nan))))
                     for o in teams}
        order = sorted(teams, key=lambda o: -est_allow[o])
        for o in teams:
            if (pos, o) in known:
                table[(pos, o)] = known[(pos, o)]
            else:
                mv = float(np.clip(np.polyval(fit_m[pos], mine.get((o, pos), np.nan)), 0.88, 1.12))
                mv = 1.0 if mv != mv else mv
                table[(pos, o)] = (round(mv, 3), order.index(o) + 1, round(est_allow[o], 1))
    mult_hist = lambda pos, opp: table.get((pos, opp), (1.0, None, None))[0]

    # ---------- range ratios (fixed per position in the original data) ----------
    rr, rros = {}, {}
    for pos in POS:
        lo = [x[3] / x[1] for p in orig if p["p"] == pos for x in p["w"] if x[0] != "BYE" and x[1] > 0]
        hi = [x[4] / x[1] for p in orig if p["p"] == pos for x in p["w"] if x[0] != "BYE" and x[1] > 0]
        rl = [p["lo"] / p["r"] for p in orig if p["p"] == pos and p["r"] > 0]
        rh = [p["hi"] / p["r"] for p in orig if p["p"] == pos and p["r"] > 0]
        rr[pos], rros[pos] = (np.mean(lo), np.mean(hi)), (np.mean(rl), np.mean(rh))

    # ---------- Vegas factor (only for weeks where every game has a line) ----------
    implied, vfac_fit = {}, (0.321, 0.004)
    for w in W[:1]:  # your original players only get Vegas lines in the first projected week, so the added players match
        gw = games[games.week == w]
        if len(gw) and gw.spread_line.notna().all() and gw.total_line.notna().all():
            imp = {}
            for _, g in gw.iterrows():
                imp[g.home_team] = (g.total_line + g.spread_line) / 2
                imp[g.away_team] = (g.total_line - g.spread_line) / 2
            implied[w] = (imp, float(np.mean(list(imp.values()))))
    if implied:
        w0 = min(implied)
        pts = [(math.log(p["w"][W.index(w0)][1] / (p["g"] * p["w"][W.index(w0)][2])), math.log(implied[w0][0][p["t"]] / implied[w0][1]))
               for p in orig if p["t"] in implied[w0][0] and p["w"][W.index(w0)][0] != "BYE" and p["w"][W.index(w0)][1] > 0 and p["g"] > 0]
        if len(pts) >= 10:
            k, c = np.polyfit([q[1] for q in pts], [q[0] for q in pts], 1)
            vfac_fit = (float(k), float(c))
    vegas = lambda team, w: (math.exp(vfac_fit[1] + vfac_fit[0] * math.log(implied[w][0][team] / implied[w][1]))
                             if w in implied and team in implied[w][0] else 1.0)
    print(f"Vegas weeks used: {sorted(implied)} (exponent {vfac_fit[0]:.3f})")

    # ---------- players ----------
    r_ = roster.copy()
    r_["team"] = r_.team.map(nt)
    if "week" in r_.columns:
        r_ = r_.sort_values("week")
    r_ = r_.groupby("gsis_id").tail(1).set_index("gsis_id")
    pfr2gsis = r_.dropna(subset=["pfr_id"]).pfr_id.to_dict()
    pfr2gsis = {v: k for k, v in pfr2gsis.items()}
    snaps = snaps.copy()
    snaps["gsis"] = snaps.pfr_player_id.map(pfr2gsis)
    sc = 100 if snaps.offense_pct.max() <= 1.5 else 1
    snap_share = snaps[snaps.offense_snaps > 0].groupby("gsis").offense_pct.mean().mul(sc).round(1).to_dict()

    first_w = W[0]
    inj_now = {}
    if "week" in inj.columns:
        ij = inj[inj.week == first_w]
        inj_now = {r.gsis_id: r.report_status for r in ij.itertuples() if isinstance(r.report_status, str)}

    appear = st[(st.attempts.fillna(0) > 0) | (st.carries.fillna(0) > 0) | (st.targets.fillna(0) > 0)]
    base = {}
    for gid, g in appear.groupby("player_id"):
        g = g.sort_values("week")
        pos = g.position.mode().iloc[0]
        adj = np.array([v / mult_hist(pos, o) for v, o in zip(g.fantasy_points_ppr.values, g.opponent_team.values)])
        sea = adj.mean()
        msk = g.week.isin(win_weeks).values
        l3 = adj[msk].mean() if msk.any() else None
        gb = 0.7 * sea + 0.3 * l3 if l3 is not None else sea
        rec = g.receptions.fillna(0).values
        rc = 0.7 * rec.mean() + 0.3 * rec[msk].mean() if msk.any() else rec.mean()
        ppr, std = g.fantasy_points_ppr.values, g.fantasy_points.values
        last3 = slice(-3, None)
        lw = int(g.week.max())
        ts = g.target_share.mean() if g.target_share.notna().any() else None
        base[gid] = dict(name=g.player_display_name.iloc[-1], pos=pos, n=len(g), lw=lw, gb=gb, rc=rc,
                         sp=[ppr.mean(), (ppr.mean() + std.mean()) / 2, std.mean()],
                         l3=[ppr[last3].mean(), (ppr[last3].mean() + std[last3].mean()) / 2, std[last3].mean()],
                         ts=ts, ru=[g.carries.fillna(0).values[last3].mean(), g.rushing_yards.fillna(0).values[last3].mean(),
                                    g.rushing_tds.fillna(0).values[last3].mean()])
    rep = {pos: float(np.percentile([b["gb"] for b in base.values() if b["pos"] == pos and b["n"] >= 6], 20)) for pos in POS}

    # match the original players to nflverse ids
    idx = {}
    for gid, b in base.items():
        idx.setdefault((nname(b["name"]), b["pos"]), []).append(gid)
    o_gid, unmatched = {}, []
    for p in orig:
        c = idx.get((nname(p["n"]), p["p"]), [])
        if c:
            o_gid[p["n"]] = max(c, key=lambda x: base[x]["n"])
        else:
            unmatched.append(p["n"])
    if unmatched:
        print("WARNING original players not matched to nflverse:", unmatched)

    # ---------- build rows ----------
    def make_week(team, pos, w, g, out_all, out_first, ):
        s = sched.get(team, {}).get(w)
        if s is None:
            return list(BYE)
        opp, home = s
        mult, rank, allows = table.get((pos, opp), (1.0, None, None))
        proj = 0.0 if (out_all or (out_first and w == first_w)) else g * mult * vegas(team, w)
        lo, hi = (proj * rr[pos][0], proj * rr[pos][1]) if proj > 0 else (0, 0.0)
        return [opp, round(proj, 1), round(mult, 3), round(lo, 1) if lo else 0, round(hi, 1), home, rank, allows]

    new, skipped = [], {"status": {}, "no_schedule": 0, "dup_of_original": 0}
    orig_gids = set(o_gid.values())
    for gid, b in base.items():
        if gid in orig_gids:
            skipped["dup_of_original"] += 1
            continue
        rr_ = r_.loc[gid] if gid in r_.index else None
        status = rr_.status if rr_ is not None else "NONE"
        if status not in ("ACT", "INA", "RES"):
            skipped["status"][status] = skipped["status"].get(status, 0) + 1
            continue
        team = rr_.team
        if team not in sched:
            skipped["no_schedule"] += 1
            continue
        pos, n = b["pos"], b["n"]
        small = n < 3
        stale = b["lw"] < last_wk - 1  # has not played in the last two weeks: may be hurt or benched
        g = (n * b["gb"] + (3 - n) * rep[pos]) / 3 if small else b["gb"]
        g = max(g, 0.0)
        out_all = status == "RES"
        rep_status = inj_now.get(gid)
        out_first = rep_status in ("Out", "Doubtful")
        wk = [make_week(team, pos, w, g, out_all, out_first) for w in W]
        r = round(sum(x[1] for x in wk), 1)
        played = [x for x in wk if x[0] != "BYE"]
        sos = 0 if out_all or not played else round((np.mean([x[2] for x in played]) - 1) * 100, 1)
        scale = g / b["gb"] if b["gb"] > 0 else 0
        p = {"n": b["name"], "t": team, "p": pos, "i": "out" if out_all else "healthy",
             "o": f"Wk {first_w}" if (out_first and not out_all) else "", "g": round(g, 1), "r": r,
             "e": 1 if (small or stale) else 0, "lo": round(r * rros[pos][0]) if r else 0, "hi": round(r * rros[pos][1], 1) if r else 0.0,
             "ts": round(b["ts"] * 100, 1) if (b["ts"] is not None and pos != "QB") else None,
             "ru": [round(v, 1) for v in b["ru"]] if pos == "QB" else None, "sos": sos, "w": wk, "x": 1}
        new.append((p, gid, b, scale))

    def extras(p, gid, b, scale):
        rc = b["rc"] * scale
        p["gid"], p["gp"], p["lw"] = gid, b["n"], b["lw"]
        p["sp"], p["l3"] = [round(v, 1) for v in b["sp"]], [round(v, 1) for v in b["l3"]]
        p["sn"] = snap_share.get(gid)
        p["rc"] = round(rc, 2)
        for x in p["w"]:  # index 8 = projected receptions that week (Half PPR = pts - 0.5*rec, Standard = pts - rec)
            if len(x) == 8:
                x.append(round(rc * x[1] / p["g"], 1) if (x[0] != "BYE" and p["g"] > 0 and x[1] > 0) else 0)
            else:
                x[8] = round(rc * x[1] / p["g"], 1) if (x[0] != "BYE" and p["g"] > 0 and x[1] > 0) else 0

    # originals: only add the extra fields (their numbers are not touched)
    EXTRA = ("gid", "gp", "lw", "sp", "l3", "sn", "rc")
    strip = lambda L: json.dumps(clean([{k: v for k, v in p.items() if k not in EXTRA and k != "w"} | {"w": [x[:8] for x in p["w"]]} for p in L]), sort_keys=True)
    before = strip(orig)
    for p in orig:
        gid = o_gid.get(p["n"])
        if gid:
            extras(p, gid, base[gid], 1.0)
    for p, gid, b, scale in new:
        extras(p, gid, b, scale)
    after = strip(orig)
    assert before == after, "original players were modified - aborting"

    # ---------- validation report ----------
    print("\n=== REPORT ===")
    mism = 0
    for p in orig:
        for w, x in zip(W, p["w"]):
            s = sched.get(p["t"], {}).get(w)
            exp = "BYE" if s is None else s[0]
            if x[0] != exp or (s and x[5] != s[1]):
                mism += 1
    print(f"schedule check vs your original players: {mism} mismatches out of {sum(len(p['w']) for p in orig)} player-weeks")
    cmpl = [(p["n"], p["g"], base[o_gid[p["n"]]]["gb"]) for p in orig if p["n"] in o_gid and not p["e"] and base[o_gid[p["n"]]]["n"] >= 3]
    err = np.array([abs(m_ / g_ - 1) for _, g_, m_ in cmpl])
    print(f"calibration of base PPG vs your engine ({len(cmpl)} players): median abs error {np.median(err)*100:.1f}%, "
          f"within 5%: {(err < .05).sum()}, within 10%: {(err < .10).sum()}")
    allp = [p for p, *_ in new]
    print(f"added {len(allp)} players:", {pos: sum(1 for p in allp if p['p'] == pos) for pos in POS})
    print("not added:", skipped)
    print("tagged est. (fewer than 3 games, or not played in the last 2 weeks):", sum(p["e"] for p in allp),
          "| on injured reserve (all weeks 0):", sum(1 for p in allp if p["i"] == "out"),
          f"| out/doubtful Week {first_w}:", sum(1 for p in allp if p["o"]))
    qn = [b["name"] for gid, b in base.items() if inj_now.get(gid) == "Questionable" and gid in r_.index and r_.loc[gid].status == "ACT"]
    print(f"questionable Week {first_w} (not adjusted): {len(qn)}")
    for pos in POS:
        top = sorted([p for p in allp if p["p"] == pos], key=lambda p: -p["r"])[:3]
        print(" top added", pos, [(p["n"], p["t"], p["r"]) for p in top])

    # ---------- write ----------
    allp.sort(key=lambda p: (POS.index(p["p"]), -p["r"]))
    D_new = orig + allp
    js = json.dumps(clean(D_new), separators=(",", ":"), ensure_ascii=False)
    out = html[:m.start(1)] + js + html[m.end(1):]
    note = (" League-wide players are added from nflverse data (through Week " + str(last_wk) + "). Players tagged est. have fewer than three games or have not played in the last two weeks (they may be injured or benched; check the injury report). "
            "Projections for added players are NOAM estimates built with the same method (70% season, 30% last three, adjusted for defenses faced) "
            "but they are not the original engine output; the original players are unchanged.")
    marker = "League-wide players are added from nflverse"
    if marker not in out:
        tail = "two-sided view with the other team’s roster."
        assert tail in out, "note anchor not found"
        out = out.replace(tail, tail + note, 1)
    open(a.out, "w", encoding="utf8").write(out)
    print(f"\nwrote {a.out}: {len(D_new)} players, {len(out)/1024:.0f} KB")


if __name__ == "__main__":
    main()

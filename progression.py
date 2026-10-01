"""Weather-driven maize foliar disease progression benchmark with real leaf imagery as observations.
Latent lesion fraction follows a favourability-driven logistic process (mechanistic simulator, parameters set
from published temperature ranges); each field visit is observed through a real held-out image whose
colour-derived lesion fraction is closest to the latent state, optionally degraded."""
import numpy as np, pandas as pd
DIS = ["gls", "nlb", "rust"]
SIM = dict(gls=dict(Topt=26.0, sig=5.0, r=0.30, onset=6.0), nlb=dict(Topt=22.0, sig=5.0, r=0.24, onset=5.0),
           rust=dict(Topt=20.0, sig=4.0, r=0.27, onset=4.0))

def load_weather(site):
    f = f"weather/{site}_climate.txt"
    df = pd.read_csv(f, sep=r"\s+")
    df.columns = ["day", "month", "year", "tmin", "tmax", "prcp", "et0"]
    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=df.day))
    df["doy"] = df.date.dt.dayofyear
    return df

def wetness(tmin, tmax, prcp):
    # proxy for leaf-wetness opportunity: rain events and humid nights (small diurnal range)
    return np.clip(prcp / 5.0, 0, 1) * 0.7 + 0.3 * (tmax - tmin < 11).astype(float) + 0.15

def favour(d, tmin, tmax, prcp):
    p = SIM[d]; T = 0.5 * (tmin + tmax)
    return np.exp(-0.5 * ((T - p["Topt"]) / p["sig"]) ** 2) * np.clip(wetness(tmin, tmax, prcp), 0, 1)

def simulate_season(w, d, rng, start=160, length=100):
    """returns daily latent lesion fraction y (length,), weather arrays"""
    ws = w[(w.doy >= start) & (w.doy < start + length)]
    tmin, tmax, prcp = ws.tmin.values, ws.tmax.values, ws.prcp.values
    y = np.zeros(len(ws))
    if d == "healthy": return y, ws
    f = favour(d, tmin, tmax, prcp); thr = SIM[d]["onset"] * rng.uniform(0.6, 1.6)
    on = np.argmax(np.cumsum(f) > thr) if (np.cumsum(f) > thr).any() else None
    if on is None: return y, ws
    eta = np.log(0.003 / 0.997); r = SIM[d]["r"] * rng.uniform(0.75, 1.25)
    for t in range(on, len(ws)):
        eta = eta + r * f[t] + rng.normal(0, 0.05)
        y[t] = 1 / (1 + np.exp(-eta))
    return np.minimum(y, 0.95), ws

def weather_feats(ws):
    T = 0.5 * (ws.tmin.values + ws.tmax.values)
    return np.stack([T / 30.0, (ws.tmax.values - ws.tmin.values) / 15.0, np.clip(ws.prcp.values / 10.0, 0, 3)], 1)

def build(site, years, plots_per_year, seed, start=160, length=100, visit_every=7, p_healthy=0.15):
    rng = np.random.default_rng(seed); w = load_weather(site); seasons = []
    for yr in years:
        wy = w[w.year == yr]
        for k in range(plots_per_year):
            d = "healthy" if rng.random() < p_healthy else DIS[rng.integers(3)]
            y, ws = simulate_season(wy, d, rng, start, length)
            visits = []; t = int(rng.integers(0, visit_every))
            while t < len(y):
                visits.append(t); t += visit_every + int(rng.integers(-2, 3))
            seasons.append(dict(year=yr, disease=d, y=y, wf=weather_feats(ws), visits=np.array(visits)))
    return seasons

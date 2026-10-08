import os, sys, glob
import numpy as np, pandas as pd
from pyproj import Transformer
sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
t = pd.read_csv(glob.glob(os.path.join(BASE, 'raw', '*Tier0*'))[0], encoding='utf-8-sig')
X, Y = Transformer.from_crs(5174, 5181, always_xy=True).transform(t.X.values, t.Y.values)
rng = np.random.default_rng(0); i = rng.integers(0, len(t), 200000); j = rng.integers(0, len(t), 200000)
d0 = np.hypot(t.X.values[i] - t.X.values[j], t.Y.values[i] - t.Y.values[j])
d1 = np.hypot(X[i] - X[j], Y[i] - Y[j]); m = (d0 > 50) & (d0 < 600)
err = (d1 - d0)[m]
print('쌍 거리 변화(m), 50~600m 쌍:', pd.Series(err).describe(percentiles=[.01, .99]).round(3).to_dict())
print('이동량 min/max (m):', np.hypot(X - t.X.values, Y - t.Y.values).min().round(2), np.hypot(X - t.X.values, Y - t.Y.values).max().round(2))
# 300m 경계 근처 쌍 비율
b = (np.abs(d0 - 300) < 0.5)
print('300m 경계 ±0.5m 쌍 중 판정이 바뀌는 비율:', ((d0[b] <= 300) != (d1[b] <= 300)).mean().round(3) if b.any() else 'n/a')

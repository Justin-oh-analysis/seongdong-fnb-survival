"""인허가 좌표의 CRS 가설 검증: 5181 vs 5174 vs 2097. (1) 지오코딩 기준 오차 (2) 상권 폴리곤 within 비율"""
import os, sys, glob, json, time, urllib.parse, urllib.request
import numpy as np, pandas as pd, geopandas as gpd
from pyproj import Transformer
sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
key = os.environ.get('GEOCODE_API_KEY', '').strip()
def rd(p):
    for e in ('utf-8-sig', 'cp949'):
        try: return pd.read_csv(p, encoding=e, low_memory=False)
        except Exception: pass
t = rd(glob.glob(os.path.join(BASE, 'raw', '*Tier0*'))[0])
poly = gpd.read_file(glob.glob(os.path.join(BASE, 'raw', '**', '*영역-상권*.shp'), recursive=True)[0])
tos = {c: Transformer.from_crs(c, 5181, always_xy=True) for c in (5174, 2097)}

# (2) 폴리곤 포함 비율
print('[상권 폴리곤 within 비율]')
for name, (x, y) in {'5181(그대로)': (t.X.values, t.Y.values),
                     '5174→5181': tos[5174].transform(t.X.values, t.Y.values),
                     '2097→5181': tos[2097].transform(t.X.values, t.Y.values)}.items():
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs=5181)
    j = gpd.sjoin(pts, poly[['geometry']], how='left', predicate='within')
    print(f'  {name}: {j.index_right.notna().groupby(level=0).any().mean():.1%}')
off = np.hypot(*(np.array(tos[5174].transform(t.X.values[:5], t.Y.values[:5])) - np.array([t.X.values[:5], t.Y.values[:5]])))
print('  5174→5181 변환 시 평균 이동량(m):', off.round(0).tolist())

# (1) 지오코딩 기준 오차
lic = rd(glob.glob(os.path.join(BASE, 'raw', '*일반음식점*'))[0])
x = pd.to_numeric(lic['좌표정보(X)'], errors='coerce'); lic = lic[x.notna()].sample(40, random_state=7)
to4326 = Transformer.from_crs(4326, 5181, always_xy=True)
res = []
for _, r in lic.iterrows():
    q = urllib.parse.urlencode(dict(service='address', request='getcoord', version='2.0', crs='epsg:4326', address=str(r['도로명주소']).strip(),
                                    refine='true', simple='false', format='json', type='road', key=key))
    try:
        j = json.loads(urllib.request.urlopen('https://api.vworld.kr/req/address?' + q, timeout=15).read().decode())
        if j['response']['status'] != 'OK': continue
        p = j['response']['result']['point']; g = to4326.transform(float(p['x']), float(p['y']))
    except Exception:
        continue
    X, Y = float(r['좌표정보(X)']), float(r['좌표정보(Y)'])
    e = {'5181': np.hypot(g[0] - X, g[1] - Y)}
    for c in (5174, 2097):
        a = tos[c].transform(X, Y); e[str(c)] = np.hypot(g[0] - a[0], g[1] - a[1])
    res.append(e); time.sleep(0.12)
df = pd.DataFrame(res); print('\n[지오코딩 기준 거리 오차(m)] n =', len(df))
print(df.describe(percentiles=[.5, .75, .9]).round(1).to_string())

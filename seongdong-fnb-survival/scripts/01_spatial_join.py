"""②-1 Tier0 점포(point) -> 상권분석서비스 상권(polygon) 공간조인 (EPSG:5181 동일, 변환 없음)"""
import os, glob, sys
import pandas as pd, geopandas as gpd
sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R, O = os.path.join(BASE, 'raw'), os.path.join(BASE, 'out')
os.makedirs(O, exist_ok=True)

t = pd.read_csv(glob.glob(os.path.join(R, '*Tier0*'))[0], encoding='utf-8-sig')
# 인허가 좌표는 EPSG:5174(구 TM중부원점, Bessel)임을 검증함(06b_crs_hypothesis.py: 지오코딩 오차 중앙값 5.4m, 5181 가정 시 317m)
from pyproj import Transformer
t['X_5181'], t['Y_5181'] = Transformer.from_crs(5174, 5181, always_xy=True).transform(t.X.values, t.Y.values)
pts = gpd.GeoDataFrame(t, geometry=gpd.points_from_xy(t.X_5181, t.Y_5181), crs=5181)
poly = gpd.read_file(glob.glob(os.path.join(R, '**', '*영역-상권*.shp'), recursive=True)[0])  # cpg=UTF-8
poly = poly.rename(columns={'TRDAR_SE_C': '상권구분코드', 'TRDAR_SE_1': '상권구분명', 'TRDAR_CD': '상권_코드',
                            'TRDAR_CD_N': '상권_코드_명', 'ADSTRD_CD_': '행정동명', 'RELM_AR': '상권면적'})
print('polygon CRS', poly.crs.to_epsg(), '| sample:', poly[['상권구분명', '상권_코드_명', '행정동명']].head(3).values.tolist())

j = gpd.sjoin(pts, poly[['상권구분코드', '상권구분명', '상권_코드', '상권_코드_명', '행정동명', '상권면적', 'geometry']],
              how='left', predicate='within')
n_match = j.groupby('관리번호')['상권_코드'].apply(lambda s: s.notna().sum())
print('점포 수', len(t), '| 매칭 0개:', (n_match == 0).sum(), f'({(n_match == 0).mean():.1%})',
      '| 1개:', (n_match == 1).sum(), '| 2개 이상(폴리곤 중복):', (n_match > 1).sum())

# 중복 매칭은 면적이 가장 작은(가장 구체적인) 상권으로 확정
j = j.sort_values(['관리번호', '상권면적']).drop_duplicates('관리번호', keep='first')
j['in_trdar'] = j['상권_코드'].notna().astype(int)
print('\n상권 구분별 점포수:\n', j['상권구분명'].value_counts(dropna=False))
print('\n매칭된 상권 수:', j['상권_코드'].nunique())
print('\n폐업 비율 (상권 내/외):\n', j.groupby('in_trdar')['영업상태명'].apply(lambda s: (s == '폐업').mean()).round(3))
print('\n연대별 상권 내 비율:\n', j.assign(yr=pd.to_datetime(j['인허가일자']).dt.year // 10 * 10).groupby('yr')['in_trdar'].mean().round(3))

out = pd.DataFrame(j.drop(columns=['geometry', 'index_right']))
out.to_csv(os.path.join(O, 'tier0_trdar.csv'), index=False, encoding='utf-8-sig')
print('\nsaved', os.path.join(O, 'tier0_trdar.csv'), out.shape)

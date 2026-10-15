"""최종 풀 데이터셋 (기준일 2021-04-01 횡단면, 반경 300m) = 팀 파일 28개 컬럼 + 우리 계산 컬럼
행: 기준일 2021-04-01에 영업 중이던 인허가 점포 중 좌표를 가진 3,779곳 (팀 파일 3,775곳 + 규칙 동일하게 계산한 4곳, in_team_file로 구분)
설계: 한 시점(기준일)의 스냅샷 + 이후 2026-09-12까지 추적. 공변량은 모두 기준일 값(시계열 사용 안 함).
입력: 팀 team_dataset.csv(수정 안 함), data/processed/baseline2021_network_r300.csv, data/processed/baseline2021_dataset.csv, raw/ 인허가 원본
출력: data/processed/final_dataset.csv

컬럼 그룹
 A 식별·시간·결과 : restaurant_id(=관리번호), 사업장명, 인허가일자·폐업일자(팀 원문 형식), open_date·close_date(ISO), baseline_date, observation_end,
                    duration_days, duration_months, event, business_age_at_baseline_years, in_team_file
 B 위치           : 지번주소, longitude, latitude, dong(읍면동), TRDAR_CD, TRDAR_CD_N(팀: 상권 코드·이름)
 C 업종·규모      : 업태구분명(인허가 26개 세부), 업종(7개 층화), 서울시_서비스_업종_코드(_명)(팀 8개 업종), Area(소재지면적 ㎡)
 D 상권 매출      : area_category_sales_per_store_final (팀, 산출식·시점 미확인)
 E 렌트           : rent_dong, rent_mean, rent_median (팀, 산출 불명) / rent_base (우리: 기준일 직전 12개월 읍면동 중앙값)
 F 경쟁(팀)       : restaurants_all_300m, competitors_same_300m
 G 경쟁·네트워크(우리, 300m): *_r300 (Competitor_C, Adjacent_BIZ, Business_D, Competitor_R, Clust_Coef, NbrDeg_Mean/Max, Ego_Density, Ego_Edges, Clust_all),
                    cluster_same(_size), cluster_all(_size)
 H 변환값         : log_age, log_area, log_rent
 I 리뷰(팀)       : place_id, review_volume, review_segment, review_segment_n, review_revisit_share, has_review  (※ 폐업 점포는 구조적으로 0 → 생존모형에 사용 금지)
"""
import os, sys, glob
import numpy as np, pandas as pd
from pyproj import Transformer
sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(BASE, 'data', 'processed'); RES = os.path.join(BASE, 'results'); os.makedirs(RES, exist_ok=True); RAW = os.path.join(BASE, 'data', 'raw')
TEAM = os.path.join(BASE, 'data', 'raw', 'team_dataset.csv')
B, END = pd.Timestamp('2021-04-01'), pd.Timestamp('2026-09-12')
iso = lambda s: pd.to_datetime(s.astype(str).str.strip().str.replace('.', '-', regex=False), errors='coerce')
dotted = lambda t: f'{t.year}.{t.month}.{t.day}'

T = pd.read_csv(TEAM, encoding='utf-8-sig', low_memory=False)
NW = pd.read_csv(os.path.join(OUT, 'baseline2021_network_r300.csv'), low_memory=False)
DS = pd.read_csv(os.path.join(OUT, 'baseline2021_dataset.csv'), low_memory=False)[['관리번호', 'Area', 'dong', 'rent_base', '업종', '업태구분명']]
for e in ('utf-8-sig', 'cp949'):
    try: lic = pd.read_csv(os.path.join(RAW, 'restaurant_licenses.csv'), encoding=e, low_memory=False); break
    except Exception: pass
lic['open_date'] = iso(lic['인허가일자']); lic['close_date'] = iso(lic['폐업일자'])

# ---- 1. 팀의 duration/event 규칙 검증 (이 규칙을 팀에 없는 4곳에도 동일 적용)
T['open_date'] = iso(T['인허가일자']); T['close_date'] = iso(T['폐업일자'])
end_calc = T['close_date'].where(T['event'] == 1, END); dur_calc = (end_calc - B).dt.days
print('[검증] 팀 duration_days = (폐업일 또는 2026-09-12) - 2021-04-01 일치율:', round((dur_calc == T['duration_days']).mean(), 4), '| event=1 ⇔ 폐업일 존재:', bool(((T['event'] == 1) == T['close_date'].notna()).all()))

# ---- 2. 우리 기준일 영업 점포 3,779 중 팀에 없는 곳 추가 (같은 규칙으로 계산)
ids_team = set(T['restaurant_id']); extra = NW[~NW['관리번호'].isin(ids_team)]['관리번호']
L = lic[lic['관리번호'].isin(extra)].copy()
tf = Transformer.from_crs(5174, 4326, always_xy=True)
xy = pd.read_csv(os.path.join(OUT, 'tier0_expanded_base.csv'), low_memory=False)[['관리번호', 'X', 'Y']]
L = L.merge(xy, on='관리번호'); L['longitude'], L['latitude'] = tf.transform(L.X.values, L.Y.values)
L['event'] = (L['close_date'].notna() & (L['close_date'] <= END)).astype(int)
endd = L['close_date'].where(L['event'] == 1, END)
add = pd.DataFrame({'restaurant_id': L['관리번호'], '사업장명': L['사업장명'], '인허가일자': L['open_date'].map(dotted), '폐업일자': L['close_date'].map(lambda t: dotted(t) if pd.notna(t) else np.nan),
                    '지번주소': L['지번주소'], 'longitude': L['longitude'], 'latitude': L['latitude'], 'event': L['event'], 'observation_end': endd.map(dotted),
                    'duration_days': (endd - B).dt.days, 'duration_months': (endd - B).dt.days / 30.4375,
                    'business_age_at_baseline_years': (B - L['open_date']).dt.days / 365.25, 'open_date': L['open_date'], 'close_date': L['close_date']})
print('[추가] 팀 파일에 없는 기준일 영업 점포:', len(add), '→', add[['사업장명', '인허가일자', '폐업일자', 'event']].to_dict('records'))
T['in_team_file'] = 1; add['in_team_file'] = 0
F = pd.concat([T, add], ignore_index=True)
F['baseline_date'] = B.date().isoformat()

# ---- 3. 우리 컬럼 결합 (키: restaurant_id = 관리번호)
F = F.merge(DS.rename(columns={'관리번호': 'restaurant_id'}), on='restaurant_id', how='left')
nw = NW.rename(columns={'관리번호': 'restaurant_id'}); nw = nw.rename(columns={c: c + '_r300' for c in nw.columns if c not in ('restaurant_id', 'cluster_same', 'cluster_same_size', 'cluster_all', 'cluster_all_size')})
F = F.merge(nw, on='restaurant_id', how='left')
# 우리 dataset에 없는 4곳 등의 보강 (dong, 업태, 업종, Area)
miss = F['dong'].isna()
sub = lic.set_index('관리번호')
F.loc[miss, 'dong'] = F.loc[miss, 'restaurant_id'].map(sub['지번주소'].astype(str).str.extract(r'성동구\s+(\S+)')[0])
F.loc[F['업태구분명'].isna(), '업태구분명'] = F.loc[F['업태구분명'].isna(), 'restaurant_id'].map(sub['업태구분명'])
F.loc[F['Area'].isna(), 'Area'] = pd.to_numeric(F.loc[F['Area'].isna(), 'restaurant_id'].map(sub['소재지면적']), errors='coerce')
rb = DS.dropna(subset=['rent_base']).drop_duplicates('dong').set_index('dong')['rent_base']   # 읍면동 단위 값이라 면적 결측 점포도 같은 동 값으로 채움
F['rent_base'] = F['rent_base'].fillna(F['dong'].map(rb))
vc = F['업태구분명'].value_counts(); F['업종'] = F['업종'].fillna(F['업태구분명'].where(F['업태구분명'].isin(vc[vc >= 100].index), '기타'))
F['log_age'] = np.log1p(F['business_age_at_baseline_years']); F['log_area'] = np.log1p(pd.to_numeric(F['Area'], errors='coerce')); F['log_rent'] = np.log(F['rent_base'])

order = (['restaurant_id', '사업장명', '인허가일자', '폐업일자', 'open_date', 'close_date', 'baseline_date', 'observation_end', 'duration_days', 'duration_months', 'event', 'business_age_at_baseline_years', 'in_team_file',
          '지번주소', 'longitude', 'latitude', 'dong', 'TRDAR_CD', 'TRDAR_CD_N',
          '업태구분명', '업종', '서울시_서비스_업종_코드', '서울시_서비스_업종_코드_명', 'Area',
          'area_category_sales_per_store_final',
          'rent_dong', 'rent_mean', 'rent_median', 'rent_base',
          'restaurants_all_300m', 'competitors_same_300m'] + [c for c in F.columns if c.endswith('_r300')] + ['cluster_same', 'cluster_same_size', 'cluster_all', 'cluster_all_size'] +
         ['log_age', 'log_area', 'log_rent', 'place_id', 'review_volume', 'review_segment', 'review_segment_n', 'review_revisit_share', 'has_review'])
assert set(order) <= set(F.columns), set(order) - set(F.columns)
F = F[order]
assert F['restaurant_id'].is_unique
fn = os.path.join(OUT, 'final_dataset.csv'); F.to_csv(fn, index=False, encoding='utf-8-sig')
print(f'\n[저장] {fn}\n  {F.shape[0]}행 x {F.shape[1]}열 | event=1 {int(F.event.sum())} ({F.event.mean():.1%}) | 팀 파일 포함 {int(F.in_team_file.sum())} / 추가 {int((F.in_team_file == 0).sum())}')
na = F.isna().sum(); print('  결측 있는 컬럼:', {k: int(v) for k, v in na[na > 0].items()})
# ---- 4. 상권 코드 매핑 진단 (선택: data/processed/tier0_trdar.csv 가 있을 때만)
if os.path.exists(os.path.join(OUT, 'tier0_trdar.csv')):
    if os.path.exists(os.path.join(OUT, 'tier0_trdar.csv')):
        tr = pd.read_csv(os.path.join(OUT, 'tier0_trdar.csv'), low_memory=False)[['관리번호', '상권_코드']].rename(columns={'관리번호': 'restaurant_id'})
        x = F[['restaurant_id', 'TRDAR_CD']].merge(tr, on='restaurant_id', how='left')
        print('[진단] 상권 코드: 팀 값 있음 %d / 우리 폴리곤 매핑 있음 %d / 둘 다 있고 일치 %d / 팀만 있음 %d' % (x.TRDAR_CD.notna().sum(), x.상권_코드.notna().sum(),
              (x.TRDAR_CD.notna() & x.상권_코드.notna() & (x.TRDAR_CD == x.상권_코드)).sum(), (x.TRDAR_CD.notna() & x.상권_코드.isna()).sum()))
print('[진단] 렌트 읍면동 일치(팀 rent_dong vs 우리 dong):', round((F.rent_dong == F.dong).mean(), 4), '| has_review 비율 event별:', F.groupby('event').has_review.mean().round(3).to_dict())

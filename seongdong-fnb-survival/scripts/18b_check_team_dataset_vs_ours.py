"""팀원 final_dataset_with_reviews.csv 를 우리 데이터(out/tier0_expanded_base.csv 등)와 대조 검증"""
import os, sys, glob
import numpy as np, pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = pd.read_csv(os.path.join(BASE, 'raw', 'team', 'final_dataset_with_reviews.csv'), encoding='utf-8-sig', low_memory=False)
dt = lambda s: pd.to_datetime(s.astype(str).str.strip().str.replace('.', '-', regex=False), errors='coerce')
T['open'] = dt(T['인허가일자']); T['close'] = dt(T['폐업일자']); T['end'] = dt(T['observation_end'])
# ---------- 1. 기준일 역산
T['base'] = T['end'] - pd.to_timedelta(T['duration_days'], unit='D')
print('[1] 역산한 기준일(관측종료일 - duration_days):', T['base'].value_counts().head(3).to_dict())
print('    관측종료일:', T['end'].value_counts().head(3).to_dict(), '| 폐업이면 종료=폐업일 일치율', round((T[T.event == 1]['end'] == T[T.event == 1]['close']).mean(), 3))
base = T['base'].mode()[0]
print('    event 비율', round(T.event.mean(), 3), '| duration<=0:', int((T.duration_days <= 0).sum()), '| 폐업일<기준일:', int((T.close < base).sum()), '| 개업일>기준일:', int((T.open > base).sum()))
age_chk = ((base - T['open']).dt.days / 365.25 - T['business_age_at_baseline_years']).abs().max(); print('    업력(business_age) = (기준일-개업일)/365.25 와 최대 차이(년):', round(age_chk, 4))

# ---------- 2. 우리 데이터와 대조: 기준일에 영업 중이던 점포
ours = pd.read_csv(os.path.join(BASE, 'out', 'tier0_expanded_base.csv'), low_memory=False)
ours['open'] = pd.to_datetime(ours['인허가일자'].astype(str).str.strip(), errors='coerce'); ours['close'] = pd.to_datetime(ours['폐업일자'].astype(str).str.strip().replace({'': None, 'nan': None}), errors='coerce')
act = ours[(ours.open <= base) & (ours.close.isna() | (ours.close >= base))]
print(f'\n[2] 우리 데이터(14,042)에서 {base.date()} 영업 중 점포: {len(act)}개 | 팀 데이터 3,775개 | 교집합 {len(set(act["관리번호"]) & set(T["restaurant_id"]))}개')
print('    팀에만 있는 ID', len(set(T['restaurant_id']) - set(ours['관리번호'])), '| 우리에만 있는(기준일 영업 중) ID', len(set(act['관리번호']) - set(T['restaurant_id'])))
# ---------- 3. 좌표 대조
m = T.merge(ours[['관리번호', 'X', 'Y', '업태구분명']], left_on='restaurant_id', right_on='관리번호', how='inner')
x5174, y5174 = Transformer.from_crs(4326, 5174, always_xy=True).transform(m.longitude.values, m.latitude.values)
d = np.hypot(x5174 - m.X.values, y5174 - m.Y.values)
print('\n[3] 좌표 대조(팀 lon/lat→5174 vs 우리 X,Y) n=', len(m), '| 거리(m):', pd.Series(d).describe(percentiles=[.5, .9, .99]).round(1).to_dict())
# ---------- 4. 경쟁 변수 재계산(기준일, 우리 방식) vs 팀 값
xy = ours[['X', 'Y']].values; tree = cKDTree(xy); op = ours.open.values; cl = ours.close.fillna(pd.Timestamp('2262-01-01')).values
b64 = np.datetime64(base); typ = ours['업태구분명'].values; idx = {k: i for i, k in enumerate(ours['관리번호'])}
rows = []
for _, r in m.iterrows():
    i = idx[r['restaurant_id']]
    nb = [j for j in tree.query_ball_point(xy[i], 300) if j != i and op[j] <= b64 and cl[j] >= b64]
    rows.append((len(nb), sum(typ[j] == typ[i] for j in nb)))
m['our_all'] = [a for a, _ in rows]; m['our_same'] = [b for _, b in rows]
for a, b in (('restaurants_all_300m', 'our_all'), ('competitors_same_300m', 'our_same')):
    print(f'[4] {a} vs 우리(기준일 {base.date()}) {b}: 상관 {m[a].corr(m[b]):.3f} | 팀 평균 {m[a].mean():.1f} vs 우리 {m[b].mean():.1f} | 완전일치 {(m[a] == m[b]).mean():.1%}')
print('    팀 competitors_same_300m 업종 기준 = 서울시_서비스_업종_코드_명(8개), 우리 = 업태구분명(26개 세부)')
print('    업종코드 결측 20.7% 해당 점포의 우리 업태 분포:'); print(m[m['서울시_서비스_업종_코드_명'].isna()]['업태구분명'].value_counts().head(6).to_dict())
print('    업종코드 분포:', T['서울시_서비스_업종_코드_명'].value_counts().to_dict())
# ---------- 5. 상권 매핑 일치
tr = pd.read_csv(os.path.join(BASE, 'out', 'tier0_trdar.csv'), low_memory=False)[['관리번호', '상권_코드']]
mm = T.merge(tr, left_on='restaurant_id', right_on='관리번호', how='inner'); both = mm.dropna(subset=['TRDAR_CD', '상권_코드'])
print(f'\n[5] 상권 매핑: 팀 결측 {T.TRDAR_CD.isna().mean():.1%} | 우리와 비교 가능 {len(both)}개 중 일치 {(both.TRDAR_CD.astype(int) == both.상권_코드.astype(int)).mean():.1%} | 우리 결측(상권밖) 17.9%')
# ---------- 6. 렌트
rp = glob.glob(os.path.join(BASE, 'raw', '*성동구*상가임대료*'))[0]
for e in ('utf-8-sig', 'cp949'):
    try: rent = pd.read_csv(rp, encoding=e); break
    except Exception: pass
r21 = rent[rent['기준년월'].astype(str).str.startswith('2021')]
g = T.groupby('rent_dong').agg(n=('restaurant_id', 'size'), rent_mean=('rent_mean', 'first'), rent_median=('rent_median', 'first')).reset_index()
g['원자료_전기간_평균'] = g['rent_dong'].map(rent.groupby('읍면동')['평당전세가'].mean()); g['원자료_2021_평균'] = g['rent_dong'].map(r21.groupby('읍면동')['평당전세가'].mean())
g['원자료_2015~21_n'] = g['rent_dong'].map(rent.groupby('읍면동').size())
print('\n[6] 렌트: 팀 rent_mean vs 원자료(읍면동)'); print(g.round(1).to_string())
print('    원자료 평당전세가 0값 비율', round((rent['평당전세가'] == 0).mean(), 4), '| 원자료 읍면동 수', rent['읍면동'].nunique())
# ---------- 7. 리뷰 변수 누수·편향
T['rev'] = T['has_review']; T['close_year'] = T['close'].dt.year
print('\n[7] has_review 비율: 전체', round(T.rev.mean(), 3), '| 영업 중(event=0)', round(T[T.event == 0].rev.mean(), 3), '| 폐업(event=1)', round(T[T.event == 1].rev.mean(), 3))
print('    폐업연도별 has_review 비율:', T[T.event == 1].groupby('close_year').rev.agg(['mean', 'size']).round(3).to_dict('index'))
print('    place_id 있음', int(T.place_id.notna().sum()), '| review_volume 있음', int(T.review_volume.notna().sum()), '| exp(review_volume)-1 요약:', (np.expm1(T.review_volume.dropna())).describe().round(1).to_dict())
print('    review_segment 분포:', T.review_segment.value_counts().to_dict())
print('    review_segment_n/revisit 결측률:', round(T.review_segment_n.isna().mean(), 3), round(T.review_revisit_share.isna().mean(), 3))
# ---------- 8. 기본 통계
print('\n[8] 기본: 폐업(event=1) %d / 영업 %d | duration_months 중앙값 %.1f | 업력 평균 %.1f년' % ((T.event == 1).sum(), (T.event == 0).sum(), T.duration_months.median(), T.business_age_at_baseline_years.mean()))
print('    경쟁변수 평균: all_300m %.1f, same_300m %.1f | same>all인 행 %d' % (T.restaurants_all_300m.mean(), T.competitors_same_300m.mean(), int((T.competitors_same_300m > T.restaurants_all_300m).sum())))

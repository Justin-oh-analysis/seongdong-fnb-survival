"""기준일(2021-04-01) 횡단면 설계: 기준일에 영업 중이던 점포로 경쟁 네트워크 지표를 다시 계산하고 Cox 적합
- 생존시간/사건: 팀원 team_dataset.csv 의 duration_days, event (기준일~폐업 또는 2026-09-12 중도절단). 팀 원본은 수정하지 않음
- 업종: 우리 업태구분명(26개 세부, 100점포 미만은 '기타'로 합침)으로 층화 (팀 업종코드는 20.7% 결측이라 사용 안 함)
- 렌트: raw 성동구청 임대료에서 기준일 직전 12개월 읍면동 중앙값을 직접 계산
- 리뷰 변수는 폐업 점포에 구조적으로 없어서(has_review: 폐업 0%) 사용하지 않음
산출물: data/processed/baseline2021_network_r{반경}.csv, data/processed/baseline2021_dataset.csv, results/cox_by_radius.csv"""
import os, sys, glob, re
import numpy as np, pandas as pd, networkx as nx
from scipy.spatial import cKDTree
from scipy.stats import chi2
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'data', 'processed'); RES = os.path.join(BASE, 'results'); os.makedirs(RES, exist_ok=True); RAW = os.path.join(BASE, 'data', 'raw')
B = pd.Timestamp('2021-04-01'); TEAM = os.path.join(BASE, 'data', 'raw', 'team_dataset.csv')
RADII = (100, 200, 300, 500)

def rd(p):
    for e in ('utf-8-sig', 'cp949'):
        try: return pd.read_csv(p, encoding=e, low_memory=False)
        except Exception: pass

# ---------- 1. 기준일 영업 중 점포 (우리 데이터)
o = pd.read_csv(os.path.join(OUT, 'tier0_expanded_base.csv'), low_memory=False)
o['open'] = pd.to_datetime(o['인허가일자'].astype(str).str.strip(), errors='coerce')
o['close'] = pd.to_datetime(o['폐업일자'].astype(str).str.strip().replace({'': None, 'nan': None}), errors='coerce')
a = o[(o.open <= B) & (o.close.isna() | (o.close > B))].reset_index(drop=True)   # 기준일 당일 폐업은 영업 중이 아님(팀 파일과 동일 규칙, duration=0 방지)
lic = rd(os.path.join(RAW, 'restaurant_licenses.csv')); lic['open'] = pd.to_datetime(lic['인허가일자'].astype(str).str.strip(), errors='coerce')
lic['close'] = pd.to_datetime(lic['폐업일자'].astype(str).str.strip().replace({'': None, 'nan': None}), errors='coerce')
act_all = lic[(lic.open <= B) & (lic.close.isna() | (lic.close > B)) & (lic.open.dt.year >= 1970)]
print(f'[1] 기준일 영업 중: 인허가 원본 {len(act_all)} / 우리 분석 데이터(좌표 확보) {len(a)} / 좌표 미확보로 빠진 점포 {len(act_all) - len(a)}')
xy = a[['X', 'Y']].values; typ = a['업태구분명'].values; n = len(a)

# ---------- 2. 네트워크 지표 (반경별)
def metrics(R):
    tree = cKDTree(xy)
    nbr = [set(j for j in nb if j != i) for i, nb in enumerate(tree.query_ball_point(xy, R))]
    same = [set(j for j in s if typ[j] == typ[i]) for i, s in enumerate(nbr)]
    cols = ['Competitor_C', 'Adjacent_BIZ', 'Business_D', 'Competitor_R', 'Clust_Coef', 'NbrDeg_Mean', 'NbrDeg_Max', 'Ego_Density', 'Ego_Edges', 'Clust_all']
    res = np.zeros((n, len(cols)))
    for i in range(n):
        S, A = same[i], nbr[i]; k = len(S)
        res[i, 0], res[i, 1] = k, len(A); res[i, 2] = len(set(typ[list(A)])) if A else 0; res[i, 3] = k / len(A) if A else 0
        if k:
            degs = [len(same[j]) for j in S]; e = sum(len(same[j] & S) for j in S) // 2
            res[i, 4] = e / (k * (k - 1) / 2) if k > 1 else 0; res[i, 5], res[i, 6] = np.mean(degs), max(degs)
            res[i, 7] = (k + e) / ((k + 1) * k / 2); res[i, 8] = e
        ka = len(A)
        if ka > 1:
            ea = sum(len(nbr[j] & A) for j in A) // 2; res[i, 9] = ea / (ka * (ka - 1) / 2)
    return pd.DataFrame(res, columns=cols), nbr

out_tbl = {}
for R in RADII:
    m, nbr = metrics(R); m.insert(0, '관리번호', a['관리번호'].values)
    if R == 300:
        for tag, same_only in (('same', True), ('all', False)):                       # Louvain 클러스터 (영업 중 점포, 300m)
            G = nx.Graph(); G.add_nodes_from(range(n))
            G.add_edges_from([(i, j) for i in range(n) for j in nbr[i] if j > i and (not same_only or typ[i] == typ[j])])
            com = sorted(nx.community.louvain_communities(G, seed=1), key=len, reverse=True); lab = np.zeros(n, int); size = np.zeros(n, int)
            for k, c in enumerate(com, 1): lab[list(c)] = k; size[list(c)] = len(c)
            m[f'cluster_{tag}'] = lab; m[f'cluster_{tag}_size'] = size
            print(f'    클러스터({tag}): {len(com)}개, 크기 상위5 {[len(c) for c in com[:5]]}, 1~2곳짜리 {sum(len(c) <= 2 for c in com)}개')
    m.to_csv(os.path.join(OUT, f'baseline2021_network_r{R}.csv'), index=False, encoding='utf-8-sig'); out_tbl[R] = m
    print(f'[2] r={R}m 완료 | Competitor_C 평균 {m.Competitor_C.mean():.1f}, Adjacent 평균 {m.Adjacent_BIZ.mean():.1f}, Clust_Coef 평균 {m.Clust_Coef.mean():.2f}')

# ---------- 3. 렌트 (기준일 직전 12개월, 읍면동 중앙값)
rent = rd(os.path.join(RAW, 'commercial_rent.csv'))
fix = {'하욍십리동': '하왕십리동', '하왕실리동': '하왕십리동', '하왕십리': '하왕십리동', '상왕십리': '상왕십리동', '금호동1기': '금호동1가'}
rent['dong'] = rent['읍면동'].replace(fix); rent['ym'] = pd.to_datetime(rent['기준년월'].astype(str) + '-01', errors='coerce'); rent = rent[rent['평당전세가'] > 0]
lic2 = lic[['관리번호', '지번주소']].copy(); lic2['dong'] = lic2['지번주소'].astype(str).str.extract(r'성동구\s+(\S+)')
def rent_of(d):
    for months in (12, 24, 36):
        s = rent[(rent.dong == d) & (rent.ym >= B - pd.DateOffset(months=months)) & (rent.ym < B)]['평당전세가']
        if len(s) >= 5: return s.median(), months, len(s)
    s = rent[rent.dong == d]['평당전세가']; return (s.median(), 99, len(s)) if len(s) else (np.nan, 0, 0)
rt = {d: rent_of(d) for d in lic2.dong.dropna().unique()}
rt = pd.DataFrame(rt, index=['rent_base', 'rent_window_m', 'rent_n']).T.reset_index().rename(columns={'index': 'dong'})
print(f'\n[3] 렌트: 읍면동 {len(rt)}개 | 12개월 창 사용 {int((rt.rent_window_m == 12).sum())}, 24개월 {int((rt.rent_window_m == 24).sum())}, 36개월 {int((rt.rent_window_m == 36).sum())}, 전기간 대체 {int((rt.rent_window_m == 99).sum())}, 결측 {int(rt.rent_base.isna().sum())}')

# ---------- 4. 팀 데이터와 결합
T = pd.read_csv(TEAM, encoding='utf-8-sig', low_memory=False)[['restaurant_id', 'duration_days', 'event', 'rent_mean', 'business_age_at_baseline_years']]
D = T.merge(a[['관리번호', '업태구분명', 'Area']], left_on='restaurant_id', right_on='관리번호', how='inner').merge(lic2[['관리번호', 'dong']], on='관리번호', how='left').merge(rt[['dong', 'rent_base']], on='dong', how='left')
base_m = {R: out_tbl[R] for R in RADII}
D = D.merge(base_m[300][['관리번호', 'cluster_same', 'cluster_all']], on='관리번호', how='left')
op = pd.read_csv(os.path.join(OUT, 'network_expanded_r300.csv'), low_memory=False)[['관리번호', 'Competitor_C', 'Adjacent_BIZ', 'Clust_Coef']].rename(columns=lambda c: c if c == '관리번호' else c + '_open')
D = D.merge(op, on='관리번호', how='left')
vc = D['업태구분명'].value_counts(); D['업종'] = D['업태구분명'].where(D['업태구분명'].isin(vc[vc >= 100].index), '기타')
D['Area'] = pd.to_numeric(D['Area'], errors='coerce'); D['log_area'] = np.log1p(D.Area); D['log_age'] = np.log1p(D.business_age_at_baseline_years); D['log_rent'] = np.log(D.rent_base)
print(f'[4] 결합: 팀 3,775 중 우리와 매칭 {len(D)} | Area 결측 {int(D.Area.isna().sum())} | 렌트 결측 {int(D.rent_base.isna().sum())} | 업종 {D.업종.nunique()}개 층화')
D = D.dropna(subset=['log_area', 'log_rent']).copy()
D.to_csv(os.path.join(OUT, 'baseline2021_dataset.csv'), index=False, encoding='utf-8-sig')
z = lambda s: (s - s.mean()) / s.std()

def prep(R):
    d = D.merge(base_m[R].drop(columns=[c for c in base_m[R].columns if c.startswith('cluster_')]), on='관리번호', how='inner')
    for c in ['Competitor_C', 'Adjacent_BIZ', 'Business_D', 'Clust_Coef', 'Ego_Density', 'log_area', 'log_age', 'log_rent']: d['z_' + c] = z(d[c])
    return d
CTRL = ['z_log_age', 'z_log_area', 'z_log_rent']; COMP = ['z_Competitor_C', 'z_Adjacent_BIZ', 'z_Business_D']; NET = ['z_Clust_Coef', 'z_Ego_Density']
def fit(d, cols, strata=('업종',)):
    return CoxPHFitter().fit(d[['duration_days', 'event'] + list(strata) + cols], 'duration_days', 'event', strata=list(strata))
def hr(m, v): s = m.summary.loc[v]; return s['exp(coef)'], s['exp(coef) lower 95%'], s['exp(coef) upper 95%'], s['p']

rows = []
for R in RADII:
    d = prep(R); m0, m1, m2 = fit(d, CTRL), fit(d, CTRL + COMP), fit(d, CTRL + COMP + NET)
    lr = 2 * (m2.log_likelihood_ - m1.log_likelihood_); lr1 = 2 * (m1.log_likelihood_ - m0.log_likelihood_)
    r = dict(반경=R, n=len(d), event=int(d.event.sum()), conc_M0=m0.concordance_index_, conc_M1=m1.concordance_index_, conc_M2=m2.concordance_index_, LR_comp_p=chi2.sf(lr1, 3), LR_net_p=chi2.sf(lr, 2))
    for nm, mm, v in (('동종경쟁', m1, 'z_Competitor_C'), ('전체이웃', m1, 'z_Adjacent_BIZ'), ('군집계수', m2, 'z_Clust_Coef'), ('ego밀도', m2, 'z_Ego_Density'), ('업력', m2, 'z_log_age'), ('렌트', m2, 'z_log_rent')):
        h = hr(mm, v); r.update({f'{nm}_HR': h[0], f'{nm}_lo': h[1], f'{nm}_hi': h[2], f'{nm}_p': h[3]})
    rows.append(r)
S = pd.DataFrame(rows); S.to_csv(os.path.join(RES, 'cox_by_radius.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250)
print('\n[5] 반경별 Cox (기준일 2021-04-01 횡단면, 업종 층화, 표준편차당 HR)')
print(S[['반경', 'n', 'event', '동종경쟁_HR', '동종경쟁_p', '전체이웃_HR', '전체이웃_p', '군집계수_HR', '군집계수_p', 'ego밀도_HR', 'LR_comp_p', 'LR_net_p', 'conc_M0', 'conc_M2']].round(4).to_string(index=False))
print('    업력 HR(M2):', S['업력_HR'].round(3).tolist(), '| 렌트 HR(M2):', S['렌트_HR'].round(3).tolist(), '렌트 p:', S['렌트_p'].round(4).tolist())

# ---------- 6. 설계 모순 진단 (300m): 같은 점포·같은 시간축에서 '기준일 경쟁' vs '개업시점 경쟁'
d = prep(300).dropna(subset=['Competitor_C_open']).copy()
for c in ('Competitor_C_open', 'Adjacent_BIZ_open', 'Clust_Coef_open'): d['z_' + c] = z(d[c])
print('\n[6] 설계 진단 (300m, 동일 점포 n=%d): 동종경쟁 HR' % len(d))
for nm, cols in (('기준일 경쟁만', CTRL + ['z_Competitor_C', 'z_Adjacent_BIZ']), ('개업시점 경쟁만', CTRL + ['z_Competitor_C_open', 'z_Adjacent_BIZ_open']),
                 ('둘 다', CTRL + ['z_Competitor_C', 'z_Adjacent_BIZ', 'z_Competitor_C_open', 'z_Adjacent_BIZ_open'])):
    m = fit(d, cols); print(f'  {nm}: ' + ', '.join(f'{v.replace("z_", "")}={m.summary.loc[v, "exp(coef)"]:.3f}(p={m.summary.loc[v, "p"]:.3f})' for v in cols if 'Compet' in v or 'Adjacent' in v))
print('    상관(기준일 동종 vs 개업시점 동종):', round(d.Competitor_C.corr(d.Competitor_C_open), 3), '| 업력과 상관: 기준일 %.3f / 개업시점 %.3f' % (d.Competitor_C.corr(d.log_age), d.Competitor_C_open.corr(d.log_age)))

# ---------- 7. 강건성: 클러스터 층화, PH 가정
d = prep(300).dropna(subset=['cluster_all']).copy()
for st, nm in ((('업종', 'cluster_all'), '업종 × 업종무관 클러스터 층화'),):
    try:
        m = fit(d, CTRL + COMP + NET, strata=st)
        print(f'\n[7] {nm}: 동종경쟁 HR {hr(m, "z_Competitor_C")[0]:.3f}(p={hr(m, "z_Competitor_C")[3]:.3f}), 군집계수 HR {hr(m, "z_Clust_Coef")[0]:.3f}(p={hr(m, "z_Clust_Coef")[3]:.3f})')
    except Exception as e: print('[7] 층화 적합 실패', str(e)[:120])
m2 = fit(prep(300), CTRL + COMP + NET)
ph = proportional_hazard_test(m2, prep(300)[['duration_days', 'event', '업종'] + CTRL + COMP + NET], time_transform='rank').summary
print('    PH 가정 검정(M2, 300m) p<0.05인 변수:', {k.replace('z_', ''): round(v, 4) for k, v in ph['p'].items() if v < 0.05} or '없음')

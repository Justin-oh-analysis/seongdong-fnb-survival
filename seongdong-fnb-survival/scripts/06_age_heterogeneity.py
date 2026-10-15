"""업력(가게 나이)에 따라 경쟁 효과가 달라지는가? (기준일 2021-04-01 횡단면, 300m) — 04_baseline_cox.py 산출물 사용"""
import os, sys
import numpy as np, pandas as pd
from lifelines import CoxPHFitter
sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(BASE, 'data', 'processed'); RES = os.path.join(BASE, 'results'); os.makedirs(RES, exist_ok=True)
D = pd.read_csv(os.path.join(OUT, 'baseline2021_dataset.csv'), low_memory=False)
R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
N = pd.read_csv(os.path.join(OUT, f'baseline2021_network_r{R}.csv'), low_memory=False).drop(columns=[c for c in pd.read_csv(os.path.join(OUT, f'baseline2021_network_r{R}.csv'), nrows=1).columns if c.startswith('cluster_')])
d = D.drop(columns=[c for c in D.columns if c.startswith('cluster_')]).merge(N, on='관리번호')
z = lambda s: (s - s.mean()) / s.std()
d['age'] = d['business_age_at_baseline_years']
def fit(x, cols):
    x = x.copy()
    for c in cols: x['z_' + c] = z(x[c]) if x[c].std() > 0 else 0
    zc = ['z_' + c for c in cols]
    vc = x['업종'].value_counts(); x['업종'] = x['업종'].where(x['업종'].isin(vc[vc >= 30].index), '기타')
    return CoxPHFitter().fit(x[['duration_days', 'event', '업종'] + zc], 'duration_days', 'event', strata=['업종'])
cols = ['log_age', 'log_area', 'log_rent', 'Competitor_C', 'Adjacent_BIZ', 'Clust_Coef']
d['log_age'] = np.log1p(d.age)
print(f'### 반경 {R}m, 기준일 2021-04-01 | 업력 구간별 Cox (구간 내에서 표준화, 업종 층화)')
rows = []
for nm, mask in (('업력 ≤5년', d.age <= 5), ('업력 5~15년', (d.age > 5) & (d.age <= 15)), ('업력 >15년', d.age > 15), ('전체', d.age >= 0)):
    x = d[mask]
    try:
        m = fit(x, cols); s = m.summary
        rows.append(dict(구간=nm, n=len(x), event=int(x.event.sum()), 폐업률=round(x.event.mean(), 3), **{f'{k}_HR': round(s.loc['z_' + k, 'exp(coef)'], 3) for k in ('Competitor_C', 'Adjacent_BIZ', 'Clust_Coef')},
                         동종_p=round(s.loc['z_Competitor_C', 'p'], 3), 전체이웃_p=round(s.loc['z_Adjacent_BIZ', 'p'], 3), 군집_p=round(s.loc['z_Clust_Coef', 'p'], 3)))
    except Exception as e: rows.append(dict(구간=nm, n=len(x), 오류=str(e)[:60]))
print(pd.DataFrame(rows).to_string(index=False))
# 상호작용: 경쟁 × 로그업력
x = d.copy()
for c in cols: x['z_' + c] = z(x[c])
x['int_comp_age'] = x['z_Competitor_C'] * x['z_log_age']; x['int_adj_age'] = x['z_Adjacent_BIZ'] * x['z_log_age']
m = CoxPHFitter().fit(x[['duration_days', 'event', '업종'] + ['z_' + c for c in cols] + ['int_comp_age', 'int_adj_age']], 'duration_days', 'event', strata=['업종'])
print('\n상호작용 모형 (전체): ', m.summary[['exp(coef)', 'p']].round(4).loc[['z_Competitor_C', 'z_Adjacent_BIZ', 'z_log_age', 'int_comp_age', 'int_adj_age']].to_string())
print('(int_comp_age HR>1이면 업력이 많을수록 동종 경쟁의 위험 효과가 커지는 방향)')

"""Cox(생존시간+폐업여부)  vs  로지스틱(기준일 후 3년 내 폐업 여부) 비교 — 최종 데이터셋, 반경 300m, 동일 점포·동일 설명 변수
설명 변수(표준화, 1 표준편차 증가당): 업력(log), 면적(log), 렌트(log), 동종 경쟁, 전체 이웃, 업종 다양성, 군집계수, ego 밀도. 업종은 Cox=층화, Logit=더미.
p값 2종: 일반(naive) / 동네 묶음 보정(cluster_all 기준 cluster-robust). 이웃 점포들의 공간적 상관을 반영하기 위함.
3년 라벨: 폐업일이 기준일+1096일 이내면 1 (모든 점포가 5.4년 관측되므로 라벨 불확실성 없음)."""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter
try:
    import statsmodels.api as sm
except ImportError:
    os.system(f'"{sys.executable}" -m pip install statsmodels -q'); import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = pd.read_csv(os.path.join(BASE, 'out', 'final_dataset_baseline2021_r300.csv'), low_memory=False)
V = {'log_age': '업력(log)', 'log_area': '면적(log)', 'log_rent': '렌트(log)', 'Competitor_C_r300': '동종 경쟁', 'Adjacent_BIZ_r300': '전체 이웃', 'Business_D_r300': '업종 다양성',
     'Clust_Coef_r300': '군집계수', 'Ego_Density_r300': 'ego 밀도'}
D = F.dropna(subset=list(V) + ['cluster_all']).copy()
vc = D['업종'].value_counts(); D['업종'] = D['업종'].where(D['업종'].isin(vc[vc >= 100].index), '기타')
for c in V: D['z_' + c] = (D[c] - D[c].mean()) / D[c].std()
Z = ['z_' + c for c in V]
D['y3'] = ((D.event == 1) & (D.duration_days <= 1096)).astype(int)
print(f'n={len(D)} | 폐업(전체 추적) {int(D.event.sum())} ({D.event.mean():.1%}) | 3년 내 폐업 y3=1: {int(D.y3.sum())} ({D.y3.mean():.1%}) | 업종 {D.업종.nunique()}개 | 클러스터 {D.cluster_all.nunique()}개')

# ---- Cox (업종 층화)
cols = ['duration_days', 'event', '업종', 'cluster_all'] + Z
cx = CoxPHFitter().fit(D[cols].drop(columns='cluster_all'), 'duration_days', 'event', strata=['업종'])
cr = CoxPHFitter().fit(D[cols], 'duration_days', 'event', strata=['업종'], cluster_col='cluster_all', robust=True)
# ---- Logit (업종 더미)
X = pd.concat([D[Z], pd.get_dummies(D['업종'], drop_first=True, dtype=float)], axis=1); X = sm.add_constant(X)
lg = sm.Logit(D['y3'], X).fit(disp=0)
lr_ = sm.Logit(D['y3'], X).fit(disp=0, cov_type='cluster', cov_kwds={'groups': D['cluster_all']})
rows = []
for c, nm in V.items():
    z = 'z_' + c
    rows.append(dict(변수=nm, Cox_HR=cx.summary.loc[z, 'exp(coef)'], Cox_p=cx.summary.loc[z, 'p'], Cox_p_보정=cr.summary.loc[z, 'p'],
                     Logit_OR=np.exp(lg.params[z]), Logit_p=lg.pvalues[z], Logit_p_보정=lr_.pvalues[z]))
T = pd.DataFrame(rows); T['방향일치'] = np.where((T.Cox_HR - 1) * (T.Logit_OR - 1) > 0, '일치', '불일치')
pd.set_option('display.width', 200)
print('\n[결과] 1 표준편차 증가당 위험비(HR) / 승산비(OR). p: 일반 / 동네 묶음 보정')
print(T.round(4).to_string(index=False))
# ---- 예측 성능
print('\n[성능] Cox concordance %.3f | Logit AUC(전체 데이터, 학습=평가) %.3f' % (cx.concordance_index_, roc_auc_score(D['y3'], lg.predict(X))))
gkf = GroupKFold(n_splits=5); aucs = []
Xv = X.drop(columns='const').values
for tr, te in gkf.split(Xv, D['y3'], groups=D['cluster_all']):
    m = LogisticRegression(max_iter=2000).fit(Xv[tr], D['y3'].values[tr]); aucs.append(roc_auc_score(D['y3'].values[te], m.predict_proba(Xv[te])[:, 1]))
print('       Logit AUC(동네 단위 5-fold 교차검증) %.3f  (각 fold: %s)' % (np.mean(aucs), ', '.join(f'{a:.3f}' for a in aucs)))
print('       업력만 넣은 Logit AUC: %.3f' % roc_auc_score(D['y3'], sm.Logit(D['y3'], sm.add_constant(D[['z_log_age']])).fit(disp=0).predict(sm.add_constant(D[['z_log_age']]))))

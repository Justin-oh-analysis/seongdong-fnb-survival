"""교수님 피드백 반영 확장 분석 (기준일 2021-04-01 코호트, 반경 300m) — 입력: data/processed/final_dataset.csv, data/processed/tier0_expanded_base.csv, data/processed/network_expanded_r300.csv
A 강건성(렌트 제외, 군집 고정효과, 군집 robust SE, 확장 구조지표)  B 확장 네트워크 지표(중심성·모듈성)  C 렌트 군집 vs 네트워크 군집
D 비례위험 검정 + 기간 분할(초기/후기)  E ML 비교(3년 내 폐업, 교차검증)  F 키페이퍼식 장기/단기 분류 비교  G 영업 점포 리뷰 확장 분석
산출물: results/ext_*.csv"""
import os, sys, warnings, time
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, networkx as nx
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import chi2, spearmanr, ttest_ind
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(BASE, 'data', 'processed'); RES = os.path.join(BASE, 'results'); os.makedirs(RES, exist_ok=True)
B = pd.Timestamp('2021-04-01'); T0 = time.time()
def rd(p):
    for e in ('utf-8-sig', 'cp949'):
        try: return pd.read_csv(p, encoding=e, low_memory=False)
        except Exception: pass
def sv(df, name): df.to_csv(os.path.join(RES, name), index=False, encoding='utf-8-sig')
def rank_z(s): r = s.rank(pct=True); return (r - r.mean()) / r.std()

# ---------------- 공통 데이터 ----------------
d = pd.read_csv(os.path.join(OUT, 'final_dataset.csv'), low_memory=False)
for c in ('Competitor_C', 'Adjacent_BIZ', 'Business_D', 'Clust_Coef', 'Ego_Density'): d[c] = d[c + '_r300']
d['age'] = d['business_age_at_baseline_years']
vc = d['업종'].value_counts(); d['업종'] = d['업종'].where(d['업종'].isin(vc[vc >= 30].index), '기타')
d = d.dropna(subset=['Area', 'rent_base']).reset_index(drop=True)
z = lambda s: (s - s.mean()) / s.std()
BASEV = ['log_age', 'log_area', 'log_rent', 'Competitor_C', 'Adjacent_BIZ', 'Business_D', 'Clust_Coef', 'Ego_Density']
for c in BASEV: d['z_' + c] = z(d[c])
NM = {'log_age': '업력', 'log_area': '면적', 'log_rent': '렌트', 'Competitor_C': '동종 경쟁', 'Adjacent_BIZ': '인접 음식점', 'Business_D': '업태 다양성', 'Clust_Coef': '군집계수', 'Ego_Density': 'ego 밀도'}
print(f'n={len(d)} 폐업 {int(d.event.sum())} 업종층 {d["업종"].nunique()}')

# ---------------- B. 확장 네트워크 지표 (동종 업태, 300m) ----------------
t0 = rd(os.path.join(OUT, 'tier0_expanded_base.csv'))
t0['open'] = pd.to_datetime(t0['인허가일자'].astype(str).str.strip(), errors='coerce'); t0['close'] = pd.to_datetime(t0['폐업일자'].astype(str).str.strip().replace({'': None, 'nan': None}), errors='coerce')
a = t0[(t0.open <= B) & (t0.close.isna() | (t0.close > B))].reset_index(drop=True); n = len(a)
xy = a[['X', 'Y']].values; typ = a['업태구분명'].values; tree = cKDTree(xy)
nbr = [set(j for j in nb if j != i) for i, nb in enumerate(tree.query_ball_point(xy, 300))]; same = [set(j for j in s if typ[j] == typ[i]) for i, s in enumerate(nbr)]
G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from([(i, j) for i in range(n) for j in same[i] if j > i])
rows_ = np.array([i for i in range(n) for j in same[i]]); cols_ = np.array([j for i in range(n) for j in same[i]])
A = coo_matrix((np.ones(len(rows_)), (rows_, cols_)), shape=(n, n)).tocsr()
dist = shortest_path(A, method='D', directed=False, unweighted=True); inv = np.where(np.isfinite(dist) & (dist > 0), 1.0 / np.where(dist > 0, dist, 1), 0.0); harm = inv.sum(1) / (n - 1); del dist, inv
btw = nx.betweenness_centrality(G, k=600, seed=1, normalized=True); pr = nx.pagerank(G)
cl = a[['관리번호']].merge(d[['restaurant_id', 'cluster_same', 'cluster_same_size']].rename(columns={'restaurant_id': '관리번호'}), on='관리번호', how='left')
ext = pd.DataFrame({'restaurant_id': a['관리번호'], 'harmonic': harm, 'betweenness': [btw[i] for i in range(n)], 'pagerank': [pr[i] for i in range(n)]})
cid = cl['cluster_same'].values
ext['within_share'] = [(sum(cid[j] == cid[i] for j in same[i]) / len(same[i]) if same[i] else 0) for i in range(n)]
ext['cluster_size'] = cl['cluster_same_size'].values
comm = nx.community.louvain_communities(G, seed=1)          # 19번 스크립트와 같은 방식(동종 업태, 300m)
Q = nx.community.modularity(G, comm)
print(f'[B] 동종 업태 네트워크 노드 {n}, 엣지 {G.number_of_edges()}, 구성요소 {nx.number_connected_components(G)}개, Louvain 군집 {len(comm)}개, 모듈성 Q={Q:.3f}')
sv(ext, 'ext_structure_metrics_r300.csv')
d = d.merge(ext, on='restaurant_id', how='left')
EXTV = ['harmonic', 'betweenness', 'pagerank', 'within_share', 'cluster_size']
for c in EXTV: d['z_' + c] = rank_z(d[c].fillna(0))
print('[B] 확장 지표 간 상관(Spearman) vs 기존 군집계수/동종수:'); print(d[['Competitor_C', 'Clust_Coef'] + EXTV].corr(method='spearman').round(2).loc[['Competitor_C', 'Clust_Coef']].to_string())

# ---------------- A. 강건성 ----------------
def cox(cols, extra=None, robust=False, data=None):
    x = d if data is None else data
    keep = ['duration_days', 'event', '업종'] + cols + ([extra] if robust else [])
    m = CoxPHFitter().fit(x[keep], 'duration_days', 'event', strata=['업종'], cluster_col=extra if robust else None, robust=robust)
    return m, x[keep]
Z = lambda cs: ['z_' + c for c in cs]
M2 = Z(BASEV); M1 = Z(BASEV[:6]); NOREN = Z([c for c in BASEV if c != 'log_rent'])
fe = pd.get_dummies(d['cluster_all'], prefix='cl', drop_first=True, dtype=float)
for c in fe.columns: d[c] = fe[c]
specs = {}
specs['S0 기본 M2'] = cox(M2)
specs['S1 렌트 제외'] = cox(NOREN)
specs['S2 M1(구조 제외)'] = cox(M1)
specs['S3 M2+군집 고정효과'] = cox(M2 + list(fe.columns))
specs['S4 M2 군집 robust SE'] = cox(M2, extra='cluster_all', robust=True)
specs['S5 M2+확장 구조지표'] = cox(M2 + Z(EXTV))
specs['S4b M2 군집(동종 270개) robust SE'] = cox(M2, extra='cluster_same', robust=True)
e2 = d.copy(); e2['entry'] = d.age * 365.25; e2['dur'] = e2.entry + d.duration_days          # 업력을 시간축으로, 기준일 업력에서 좌측절단(기존 영업 점포 코호트의 생존자 편향 보정)
m6 = CoxPHFitter().fit(e2[['dur', 'event', '업종', 'entry'] + Z([c for c in BASEV if c != 'log_age'])], 'dur', 'event', entry_col='entry', strata=['업종']); specs['S6 업력 시간축+좌측절단'] = (m6, None)
from statsmodels.stats.outliers_influence import variance_inflation_factor
Xv = d[Z(BASEV)].assign(c=1.0); print('[A] VIF:', {NM[v[2:]]: round(variance_inflation_factor(Xv.values, i), 2) for i, v in enumerate(Z(BASEV))}); print('[A] 상관 동종-인접(Spearman):', round(spearmanr(d.Competitor_C, d.Adjacent_BIZ)[0], 3))
rows = []; ll0 = specs['S0 기본 M2'][0].log_likelihood_
for sp, (m, _) in specs.items():
    s = m.summary
    for v in M2 + Z(EXTV):
        if 'z_' + v.replace('z_', '') in s.index or v in s.index:
            r = s.loc[v]; rows.append(dict(spec=sp, var=v[2:], HR=r['exp(coef)'], lo=r['exp(coef) lower 95%'], hi=r['exp(coef) upper 95%'], p=r['p'], concordance=m.concordance_index_, ll=m.log_likelihood_, n=len(d)))
R = pd.DataFrame(rows); sv(R, 'robustness_by_spec.csv')
print('\n[A] 강건성 (HR [95%CI] p)')
for sp in specs:
    r = R[(R.spec == sp) & R['var'].isin(['log_age', 'log_area', 'log_rent', 'Competitor_C', 'Adjacent_BIZ', 'Business_D', 'Clust_Coef', 'Ego_Density'])]
    print(f' {sp} | conc {specs[sp][0].concordance_index_:.4f} | ' + ' ; '.join(f'{NM[v]} {h:.2f}[{l:.2f},{u:.2f}] p={p:.3f}' for v, h, l, u, p in zip(r['var'], r.HR, r.lo, r.hi, r.p)))
m_ext = specs['S5 M2+확장 구조지표'][0]; print(' 확장 구조지표 개별 HR:', {v[2:]: round(m_ext.summary.loc[v, 'exp(coef)'], 3) for v in Z(EXTV)}, 'p:', {v[2:]: round(m_ext.summary.loc[v, 'p'], 3) for v in Z(EXTV)})
lr_chi = 2 * (m_ext.log_likelihood_ - ll0); print(f' LR(확장 구조지표 5개 추가 vs M2): chi2={lr_chi:.2f}, df=5, p={chi2.sf(lr_chi, 5):.4f}')
lr_fe = 2 * (specs['S3 M2+군집 고정효과'][0].log_likelihood_ - ll0); print(f' LR(군집 고정효과 {fe.shape[1]}개 vs M2): chi2={lr_fe:.1f}, p={chi2.sf(lr_fe, fe.shape[1]):.2e}')
for nm, ma, mb, df_ in (('구조 지표 2개(M2 vs M1)', specs['S2 M1(구조 제외)'][0], specs['S0 기본 M2'][0], 2), ('렌트(M2 vs 렌트 제외)', specs['S1 렌트 제외'][0], specs['S0 기본 M2'][0], 1)):
    c2 = 2 * (mb.log_likelihood_ - ma.log_likelihood_); print(f' LR {nm}: chi2={c2:.2f}, p={chi2.sf(c2, df_):.4f}')

# ---------------- C. 렌트 군집 vs 네트워크 군집 ----------------
dr = d.groupby('dong').agg(rent=('rent_base', 'first'), n=('event', 'size')).reset_index(); km = KMeans(5, random_state=0, n_init=20).fit(np.log(dr[['rent']]))
order = pd.Series(km.cluster_centers_.ravel()).rank().astype(int).values; dr['rent_grp'] = [order[l] for l in km.labels_]; d = d.merge(dr[['dong', 'rent_grp']], on='dong', how='left')
rg = d.groupby('rent_grp').agg(동수=('dong', 'nunique'), 점포=('event', 'size'), 폐업률=('event', 'mean'), 평균렌트=('rent_base', 'mean')).round(3); rg['읍면동'] = d.groupby('rent_grp')['dong'].agg(lambda s: ','.join(sorted(s.unique())))
print('\n[C] 렌트 5개 군집(읍면동 평당 전세가 k-means):'); print(rg.to_string()); sv(rg.reset_index(), 'ext_rent_clusters.csv')
print(f' 렌트군집 vs 네트워크 군집: NMI(cluster_all)={normalized_mutual_info_score(d.rent_grp, d.cluster_all):.3f}, ARI={adjusted_rand_score(d.rent_grp, d.cluster_all):.3f} | NMI(cluster_same)={normalized_mutual_info_score(d.rent_grp, d.cluster_same):.3f}')
print(' 점포 단위 Spearman(log_rent): 동종', round(spearmanr(d.log_rent, d.Competitor_C)[0], 3), '인접', round(spearmanr(d.log_rent, d.Adjacent_BIZ)[0], 3), '군집계수', round(spearmanr(d.log_rent, d.Clust_Coef)[0], 3))
rgd = pd.get_dummies(d['rent_grp'], prefix='rg', drop_first=True, dtype=float)
for c in rgd.columns: d[c] = rgd[c]
mrg, _ = cox(Z([c for c in BASEV if c != 'log_rent']) + list(rgd.columns)); print(' 렌트 5군집 더미 Cox(1군집=저렴 기준):', {c: round(mrg.summary.loc[c, 'exp(coef)'], 2) for c in rgd.columns}, 'p', {c: round(mrg.summary.loc[c, 'p'], 3) for c in rgd.columns}, f'conc {mrg.concordance_index_:.4f}')

# ---------------- D. 비례위험 검정 + 초기/후기 분할 ----------------
m0, x0 = specs['S0 기본 M2']; ph = proportional_hazard_test(m0, x0, time_transform='rank').summary
print('\n[D] 비례위험 검정(Schoenfeld, rank):'); print(ph.round(4).loc[M2, ['test_statistic', 'p']].to_string())
sv(ph.reset_index(), 'ext_ph_test.csv')
CUT = 730; e = d.copy(); e['duration_days'] = np.minimum(d.duration_days, CUT); e['event'] = ((d.event == 1) & (d.duration_days <= CUT)).astype(int)
me, _ = cox(M2, data=e); late = d[d.duration_days > CUT].copy(); late['entry'] = CUT
ml = CoxPHFitter().fit(late[['duration_days', 'event', '업종', 'entry'] + M2], 'duration_days', 'event', entry_col='entry', strata=['업종'])
tt = pd.DataFrame({'var': [NM[c] for c in BASEV], 'HR_초기(0-2년)': me.summary.loc[M2, 'exp(coef)'].values, 'p_초기': me.summary.loc[M2, 'p'].values, 'HR_후기(2년 이후)': ml.summary.loc[M2, 'exp(coef)'].values, 'p_후기': ml.summary.loc[M2, 'p'].values,
                   'HR_전체': m0.summary.loc[M2, 'exp(coef)'].values})
print(f' 초기 사건 {int(e.event.sum())}건 / 후기 표본 {len(late)}곳 사건 {int(late.event.sum())}건'); print(tt.round(3).to_string(index=False)); sv(tt, 'ext_ph_timesplit.csv')

# ---------------- E. ML 비교: 3년 내 폐업 ----------------
d['y3'] = ((d.event == 1) & (d.duration_days <= 1096)).astype(int); dum = pd.get_dummies(d['업종'], drop_first=True, dtype=float)
Xb = pd.concat([d[Z(BASEV)], dum], axis=1); Xe = pd.concat([d[Z(BASEV) + Z(EXTV)], dum], axis=1); y = d.y3.values; grp = d.cluster_all.values
MODELS = {'로지스틱': lambda: LogisticRegression(max_iter=2000), '그래디언트부스팅(HGB)': lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=.05, max_iter=200, l2_regularization=1.0, random_state=0),
          '랜덤포레스트': lambda: RandomForestClassifier(300, min_samples_leaf=20, random_state=0, n_jobs=-1)}
def cv(X, mk, split):
    aucs = []
    for tr, te in split:
        m = mk().fit(X.iloc[tr], y[tr]); aucs.append(roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:, 1]))
    return np.mean(aucs), np.std(aucs)
gsp = list(GroupKFold(5).split(Xb, y, grp)); rsp = list(StratifiedKFold(5, shuffle=True, random_state=1).split(Xb, y))
res = []
for fs, X in (('기본 8변수+업종', Xb), ('+확장 구조지표', Xe)):
    for nm, mk in MODELS.items():
        for cvn, sp in (('군집 단위 분할', gsp), ('무작위 분할', rsp)):
            mu, sd = cv(X, mk, sp); res.append(dict(변수=fs, 모델=nm, 검증=cvn, AUC=round(mu, 3), 표준편차=round(sd, 3)))
ag = pd.DataFrame({'age': d.log_age}); res.append(dict(변수='업력만', 모델='로지스틱', 검증='군집 단위 분할', AUC=round(cv(ag, MODELS['로지스틱'], gsp)[0], 3), 표준편차=np.nan))
E = pd.DataFrame(res); sv(E, 'ext_ml_cv.csv'); print(f'\n[E] 3년 내 폐업(y3={d.y3.mean():.1%}) 예측 AUC (5-fold)'); print(E.to_string(index=False))
imp = []
for tr, te in gsp:
    m = MODELS['그래디언트부스팅(HGB)']().fit(Xb.iloc[tr], y[tr]); pi = permutation_importance(m, Xb.iloc[te], y[te], scoring='roc_auc', n_repeats=5, random_state=0); imp.append(pi.importances_mean)
imp = pd.Series(np.mean(imp, 0), index=Xb.columns).sort_values(ascending=False); imp = imp[[c.startswith('z_') for c in imp.index]]; imp.index = [NM[i[2:]] for i in imp.index]
print(' 순열 중요도(AUC 감소, 군집 분할 평균):', imp.round(4).to_dict()); sv(imp.rename('AUC_감소').reset_index().rename(columns={'index': '변수'}), 'ext_ml_importance.csv')

# ---------------- F. 키페이퍼식 장기/단기 이진 분류 (개업 시점 지표) ----------------
nx_ = rd(os.path.join(OUT, 'network_expanded_r300.csv')); nx_['open'] = pd.to_datetime(nx_['인허가일자'].astype(str).str.strip(), errors='coerce'); nx_['close'] = pd.to_datetime(nx_['폐업일자'].astype(str).str.strip().replace({'': None, 'nan': None}), errors='coerce')
nx_['Area'] = pd.to_numeric(nx_['Area'], errors='coerce'); nx_ = nx_[nx_.Area > 0]; nx_['life'] = (nx_.close - nx_.open).dt.days; c_ = nx_[nx_.close.notna() & nx_.open.dt.year.between(2017, 2019) & nx_.Area.notna()].copy()
c_['y'] = np.where(c_.life <= 183, 0, np.where(c_.life >= 1826, 1, -1)); c_ = c_[c_.y >= 0]
print(f'\n[F] 키페이퍼식 분류(개업 2017~2019, 폐업 점포만): 단기(≤6개월) {int((c_.y==0).sum())}, 장기(≥5년) {int((c_.y==1).sum())}')
FJ = ['Competitor_C', 'Adjacent_BIZ', 'Business_D', 'Competitor_R', 'Clust_Coef', 'NbrDeg_Mean', 'NbrDeg_Max', 'Ego_Density']
XJ = pd.concat([c_[FJ].reset_index(drop=True), np.log(c_[['Area']]).reset_index(drop=True), pd.get_dummies(c_['업태구분명'], dtype=float, drop_first=True).reset_index(drop=True)], axis=1); yj = c_.y.values
JM = {'Logistic Regression': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced')), 'KNN': lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
      'SVM': lambda: make_pipeline(StandardScaler(), SVC(probability=True, class_weight='balanced', random_state=0)), 'Decision Tree': lambda: DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, class_weight='balanced', random_state=0),
      'Random Forest': lambda: RandomForestClassifier(300, min_samples_leaf=5, class_weight='balanced', random_state=0, n_jobs=-1), 'Gradient Boosting(XGBoost 계열)': lambda: GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=.05, random_state=0),
      'HistGB(LightGBM 계열)': lambda: HistGradientBoostingClassifier(max_depth=4, learning_rate=.05, max_iter=200, class_weight='balanced', random_state=0)}
resj = []
for nm, mk in JM.items():
    au, ba = [], []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0).split(XJ, yj):
        m = mk().fit(XJ.iloc[tr], yj[tr]); p = m.predict_proba(XJ.iloc[te])[:, 1]; au.append(roc_auc_score(yj[te], p)); ba.append(balanced_accuracy_score(yj[te], p > .5))
    resj.append(dict(모델=nm, AUC=round(np.mean(au), 3), 균형정확도=round(np.mean(ba), 3), n=len(yj)))
J = pd.DataFrame(resj); sv(J, 'ext_jang_style.csv'); print(J.to_string(index=False))

# ---------------- G. 영업 점포 리뷰 확장 분석 ----------------
s_ = d[d.event == 0].copy(); s_['matched'] = s_.has_review.fillna(0).astype(int)
print(f'\n[G] 기준일 영업 점포 중 관측 종료까지 영업: {len(s_)}곳, 리뷰 매칭 {int(s_.matched.sum())}곳 ({s_.matched.mean():.1%})')
cmp_ = []
for v, nm in (('age', '업력'), ('Area', '면적'), ('rent_base', '렌트'), ('Competitor_C', '동종 경쟁'), ('Adjacent_BIZ', '인접 음식점')):
    u, w = s_[s_.matched == 1][v], s_[s_.matched == 0][v]; cmp_.append(dict(변수=nm, 매칭=round(u.mean(), 2), 미매칭=round(w.mean(), 2), p=round(ttest_ind(u, w, equal_var=False)[1], 4)))
print(pd.DataFrame(cmp_).to_string(index=False))
r_ = s_[s_.matched == 1].dropna(subset=['review_volume']).copy(); dm = pd.get_dummies(r_['업종'], drop_first=True, dtype=float); orows = []
for yv, nm in (('review_volume', '리뷰 규모(로그)'), ('review_revisit_share', '재방문 비율')):
    rr = r_.dropna(subset=[yv]); Xo = sm.add_constant(pd.concat([rr[Z(BASEV)], dm.loc[rr.index]], axis=1)); mo = sm.OLS(rr[yv], Xo).fit(cov_type='cluster', cov_kwds={'groups': rr['cluster_all']})
    for v in Z(BASEV): orows.append(dict(종속=nm, 변수=NM[v[2:]], 계수=round(mo.params[v], 4), p=round(mo.pvalues[v], 4), n=int(mo.nobs), R2=round(mo.rsquared, 3)))
O = pd.DataFrame(orows); sv(O, 'ext_review_ols.csv'); print(O.to_string(index=False))
print(f'\n완료 {time.time() - T0:.0f}초')

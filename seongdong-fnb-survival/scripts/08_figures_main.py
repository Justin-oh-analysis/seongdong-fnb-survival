"""7주차 Model Development용 시각화 (기준일 2021-04-01 코호트, 300m 기준) — 입력: data/processed/final_dataset.csv 외
산출물: data/processed/fig_model_dev/fig01~fig12.png, fig_model_dev_numbers.csv (그림에 쓴 수치)
모든 수치는 이 스크립트가 데이터에서 직접 계산한다(개념도 fig01, fig08 제외)."""
import os, sys, glob
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from lifelines import CoxPHFitter, KaplanMeierFitter
from scipy.spatial import cKDTree
sys.stdout.reconfigure(encoding='utf-8')
if os.environ.get('NOFIGNUM', '1') != '0':    # 논문 본문에서 그림 번호를 따로 붙이므로 그림 안의 번호는 제거
    import re
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    _st, _su = Axes.set_title, Figure.suptitle
    Axes.set_title = lambda self, label, *a, **k: _st(self, re.sub(r'^그림 \d+\.\s*', '', label), *a, **k)
    Figure.suptitle = lambda self, t, *a, **k: _su(self, re.sub(r'^그림 \d+\.\s*', '', t), *a, **k)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'data', 'processed'); RES = os.path.join(BASE, 'results'); os.makedirs(RES, exist_ok=True); RAW = os.path.join(BASE, 'data', 'raw'); FIG = os.environ.get('FIGDIR') or os.path.join(RES, 'figures'); os.makedirs(FIG, exist_ok=True)
B = pd.Timestamp('2021-04-01')
BLUE, ORANGE, AQUA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7'
INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'savefig.dpi': 200, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'axes.axisbelow': True, 'font.size': 10})
NUM = []   # (그림, 항목, 값)
def dodge(vals, gap):
    """라벨이 겹치지 않도록 y 위치를 최소 gap만큼 벌린다"""
    vals = np.asarray(vals, float); out = vals.copy(); o = np.argsort(vals)
    for u, v in zip(o[:-1], o[1:]):
        if out[v] - out[u] < gap: out[v] = out[u] + gap
    return out
def save(fig, name): fig.savefig(os.path.join(FIG, name), bbox_inches='tight'); plt.close(fig); print('저장', name)
def rd(p):
    for e in ('utf-8-sig', 'cp949'):
        try: return pd.read_csv(p, encoding=e, low_memory=False)
        except Exception: pass

# ---------------- 데이터 ----------------
d = pd.read_csv(os.path.join(OUT, 'final_dataset.csv'), low_memory=False)
for c in ('Competitor_C', 'Adjacent_BIZ', 'Business_D', 'Clust_Coef', 'Ego_Density'): d[c] = d[c + '_r300']
d['age'] = d['business_age_at_baseline_years']; d['yrs'] = d['duration_days'] / 365.25
vc = d['업종'].value_counts(); d['업종'] = d['업종'].where(d['업종'].isin(vc[vc >= 30].index), '기타')
d = d.dropna(subset=['Area', 'rent_base']).copy()
z = lambda s: (s - s.mean()) / s.std()
print(f'분석 표본 {len(d)}곳, 폐업 {int(d.event.sum())}, 중도절단 {int((1 - d.event).sum())}, 업종 층 {d["업종"].nunique()}개')
def fit(x, cols, extra=()):
    x = x.copy(); zc = []
    for c in cols:
        x['z_' + c] = z(x[c]) if x[c].std() > 0 else 0; zc.append('z_' + c)
    m = CoxPHFitter().fit(x[['duration_days', 'event', '업종'] + zc + list(extra)], 'duration_days', 'event', strata=['업종'])
    return m, x
LAB = {'log_age': '업력(로그)', 'log_area': '면적(로그)', 'log_rent': '임대료(로그, 읍면동)', 'Competitor_C': '동종 경쟁 점포 수', 'Adjacent_BIZ': '인접 음식점 수(전체)',
       'Business_D': '업태 다양성', 'Clust_Coef': '군집계수(구조)', 'Ego_Density': 'ego 밀도(구조)'}

# ---------------- fig01 연구 틀 ----------------
def box(ax, x, y, w, h, txt, fc, ec, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=fs, color=INK, fontweight='bold' if bold else 'normal', linespacing=1.5)
def arrow(ax, p, q, color=INK2, rad=0, ls='-', lw=1.6): ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=14, color=color, lw=lw, connectionstyle=f'arc3,rad={rad}', linestyle=ls))
fig, ax = plt.subplots(figsize=(11, 4.9)); ax.set_xlim(0, 11); ax.set_ylim(0, 4.9); ax.axis('off'); ax.grid(False)
cols = [('① 데이터', '성동구 음식점 3,775곳\n(기준일 2021-04-01)\n인허가·상권·임대료', '#eaf2fc', BLUE), ('② 경쟁 네트워크', '같은 업태 점포를\n반경으로 연결\n수·다양성·군집계수', '#eaf2fc', BLUE),
        ('③ Cox 생존모형', '폐업 위험을 바꾸는 요인\n→ 위험비(HR)\n(영업 중 점포 포함)', '#fdeee7', ORANGE), ('④ SD 정책 시뮬레이션', '폐업률 = 기저 × exp(β·Δ)\n임대료·밀집 시나리오\n점포 수 경로 비교', '#e3f6ef', AQUA)]
for i, (t, s, fc, ec) in enumerate(cols):
    x = 0.25 + i * 2.75; ax.text(x + 1.15, 4.55, t, ha='center', fontsize=11.5, fontweight='bold', color=ec if ec != AQUA else '#0f7a56')
    box(ax, x, 2.55, 2.3, 1.75, s, fc, ec)
    if i < 3: arrow(ax, (x + 2.35, 3.42), (x + 2.72, 3.42))
ax.text(5.5, 2.15, '선행연구가 다룬 범위', ha='center', fontsize=10, color=INK2)
for (x, w, t, ec) in ((0.25, 5.05, 'Cox 선행: Cho(2023), Son(2023), Bagley(2019)\n밀도의 비선형: Lee, Kim, & Lee(2021)', BLUE), (5.5, 2.3, 'ML 선행: Jang(2024)\n폐업 점포만 사용', ORANGE), (8.0, 2.65, '시뮬레이션: Zhu(2021), 이윤석 외(2004)\nLee, Lee, Kang, & Cheon(2025)', AQUA)):
    box(ax, x, 0.95, w, 0.95, t, 'white', ec, fs=9)
for x in (1.4, 4.15): ax.plot([x, x], [2.5, 1.95], color=BLUE, lw=1)
ax.plot([6.65, 6.65], [2.5, 1.95], color=ORANGE, lw=1); ax.plot([9.3, 9.3], [2.5, 1.95], color=AQUA, lw=1)
box(ax, 0.25, 0.05, 10.4, 0.62, '본 연구의 확장:  영업 중 점포 포함(중도절단)  +  경쟁 연결 구조 지표  +  Cox 결과를 SD 파라미터로 직접 연결', '#f3f3f0', INK2, fs=10.5, bold=True)
ax.set_title('그림 1. 연구 틀: 선행연구의 세 갈래를 하나의 흐름으로 연결', loc='left', fontsize=12, fontweight='bold', pad=4)
save(fig, 'fig01_framework.png')

# ---------------- fig02 업종별 폐업률 (표본 기술) ----------------
g = d.groupby('업종').agg(n=('event', 'size'), rate=('event', 'mean')).sort_values('rate')
fig, ax = plt.subplots(figsize=(7, 4.2)); ax.barh(g.index, g.rate * 100, color=BLUE, height=.62)
ov = d.event.mean() * 100; ax.axvline(ov, color=ORANGE, lw=1.6, label=f'전체 평균 {ov:.1f}%'); ax.legend(loc='lower right', frameon=False, fontsize=9.5)
for y, (n, r) in enumerate(zip(g.n, g.rate)): ax.text(r * 100 + 1, y, f'{r*100:.0f}%  (n={n})', va='center', fontsize=9, color=INK2)
ax.set_xlim(0, 100); ax.set_xlabel('기준일 이후 5.4년 내 폐업 비율 (%)'); ax.grid(axis='y', visible=False)
ax.set_title('그림 2. 업태마다 폐업 속도가 다르다 → 업종 층화가 필요', loc='left', fontsize=11.5, fontweight='bold')
for k, v in g.iterrows(): NUM.append(('fig02', f'폐업률 {k}', round(v.rate, 3)))
save(fig, 'fig02_closure_by_type.png')

# ---------------- fig03 위치 지도 ----------------
t0 = rd(os.path.join(OUT, 'tier0_expanded_base.csv'))
t0['open'] = pd.to_datetime(t0['인허가일자'].astype(str).str.strip(), errors='coerce'); t0['close'] = pd.to_datetime(t0['폐업일자'].astype(str).str.strip().replace({'': None, 'nan': None}), errors='coerce')
a = t0[(t0.open <= B) & (t0.close.isna() | (t0.close > B))].reset_index(drop=True)
assert set(d.restaurant_id) <= set(a['관리번호']), '키 불일치'
a = a.merge(d[['restaurant_id', 'event']].rename(columns={'restaurant_id': '관리번호'}), on='관리번호', how='left')
fig, ax = plt.subplots(figsize=(6.6, 5.4)); ax.grid(False)
op, cl = a[a.event == 0], a[a.event == 1]
ax.scatter(op.X, op.Y, s=7, c=BLUE, alpha=.55, lw=0, label=f'영업 지속 {len(op):,}곳'); ax.scatter(cl.X, cl.Y, s=7, c=ORANGE, alpha=.55, lw=0, label=f'폐업 {len(cl):,}곳')
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
ax.legend(loc='lower left', frameon=False, markerscale=2.5, fontsize=10); ax.set_title('그림 3. 분석 대상: 기준일 영업 점포 3,775곳의 위치와 이후 상태', loc='left', fontsize=11.5, fontweight='bold')
save(fig, 'fig03_map_cohort.png')

# ---------------- fig04 Kaplan-Meier ----------------
def kmplot(ax, groups, cols, title):
    ends = []
    for (nm, x), c in zip(groups, cols):
        k = KaplanMeierFitter().fit(x.yrs, x.event); s = k.survival_function_.iloc[:, 0]
        ax.step(s.index, s.values, where='post', color=c, lw=2.2); ends.append((s.index[-1], s.values[-1]))
        NUM.append(('fig04', f'5년 생존율 {title} {nm}', round(float(k.predict(5.0)), 3)))
    ys = dodge([e[1] for e in ends], .055)
    for (nm, x), (xe, ye), yl in zip(groups, ends, ys): ax.text(xe + .08, yl, f'{nm} (n={len(x):,})', color=INK2, fontsize=9, va='center')
    ax.set_xlim(0, 8.3); ax.set_ylim(.3, 1.01); ax.set_xlabel('기준일 이후 경과(년)'); ax.set_ylabel('생존 비율 (세로축 0.3부터)'); ax.set_title(title, loc='left', fontsize=10.5, fontweight='bold', color=INK2)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
ag = [('5년 이하', d[d.age <= 5]), ('5~15년', d[(d.age > 5) & (d.age <= 15)]), ('15년 초과', d[d.age > 15])]
kmplot(axs[0], ag, [ORANGE, BLUE, VIOLET], '(a) 가게 나이별')
rk = d['Competitor_C'].rank(method='first'); tr = pd.qcut(rk, 3, labels=False)
cg = [(f'동종 {int(d.Competitor_C[tr == i].min())}~{int(d.Competitor_C[tr == i].max())}곳', d[tr == i]) for i in range(3)]
kmplot(axs[1], cg, [AQUA, BLUE, VIOLET], '(b) 반경 300m 동종 경쟁 점포 수 3분위별')
fig.suptitle('그림 4. 생존곡선(Kaplan-Meier): 나이에 따른 차이가 가장 크고, 동종 경쟁은 상위 3분위만 다르다 (보정 전)', x=.07, ha='left', fontsize=11.5, fontweight='bold', y=1.0)
save(fig, 'fig04_km.png')

# ---------------- fig05 포레스트 (M2, 300m) ----------------
c2 = ['log_age', 'log_area', 'log_rent', 'Competitor_C', 'Adjacent_BIZ', 'Business_D', 'Clust_Coef', 'Ego_Density']
m2, _ = fit(d, c2); s = m2.summary
print('M2 HR:', {k: round(s.loc['z_' + k, 'exp(coef)'], 3) for k in c2}, 'concordance', round(m2.concordance_index_, 3))
grp = [('통제 변수', ['log_age', 'log_area', 'log_rent']), ('경쟁 (수·다양성)', ['Competitor_C', 'Adjacent_BIZ', 'Business_D']), ('구조 (연결 모양)', ['Clust_Coef', 'Ego_Density'])]
fig, ax = plt.subplots(figsize=(8.2, 5.0)); y = 0; yt = []; yl = []
for gname, cs in grp:
    ax.text(0.56, y, gname, fontsize=9.5, fontweight='bold', color=INK2, va='center'); y += 1
    for c in cs:
        r = s.loc['z_' + c]; hr, lo, hi, p = r['exp(coef)'], r['exp(coef) lower 95%'], r['exp(coef) upper 95%'], r['p']; sig = p < .05
        ax.plot([lo, hi], [y, y], color=BLUE if sig else MUTED, lw=2); ax.plot(hr, y, 'o', ms=8, mfc=BLUE if sig else 'white', mec=BLUE if sig else MUTED, mew=1.8)
        ax.text(1.36, y, f'{hr:.2f}  [{lo:.2f}–{hi:.2f}]', va='center', fontsize=9, color=INK2); yt.append(y); yl.append(LAB[c]); y += 1
        NUM.append(('fig05', f'M2 HR {c}', f'{hr:.3f} [{lo:.3f},{hi:.3f}] p={p:.4f}'))
    y += .4
ax.axvline(1, color=INK2, lw=1); ax.set_xscale('log'); ax.set_xlim(.55, 1.65); ax.set_xticks([.6, .7, .8, .9, 1, 1.1, 1.2, 1.3]); ax.set_xticklabels(['0.6', '0.7', '0.8', '0.9', '1.0', '1.1', '1.2', '1.3'])
ax.set_yticks(yt); ax.set_yticklabels(yl); ax.set_ylim(y - .2, -.6); ax.grid(axis='y', visible=False)
ax.set_xlabel('위험비 HR (표준편차 1 증가당, 업종 층화)      ←위험 낮음  |  위험 높음→')
fig.text(.13, -.01, f'● 파란 점: p<0.05    ○ 회색: 유의하지 않음 (경쟁 변수 3개는 p가 0.05 안팎의 경계)\n(n={len(d):,}, 폐업 {int(d.event.sum()):,}, concordance {m2.concordance_index_:.3f})', fontsize=8.5, color=MUTED, va='top')
ax.set_title('그림 5. Cox 결과(반경 300m): 업력·면적이 가장 뚜렷하고, 구조 지표는 효과가 없다', loc='left', fontsize=11, fontweight='bold')
save(fig, 'fig05_forest_m2.png')

# ---------------- fig06 반경별 강건성 + 모델 단계별 concordance ----------------
sm = pd.read_csv(os.path.join(RES, 'cox_by_radius.csv'))
fig, axs = plt.subplots(2, 2, figsize=(9.6, 6.6)); R = sm['반경'].values
for ax, (nm, k, col) in zip(axs.ravel()[:3], (('(a) 동종 경쟁', '동종경쟁', BLUE), ('(b) 인접 음식점 수(전체)', '전체이웃', ORANGE), ('(c) 군집계수(구조)', '군집계수', AQUA))):
    h, lo, hi = sm[k + '_HR'], sm[k + '_lo'], sm[k + '_hi']
    ax.axhline(1, color=INK2, lw=1); ax.errorbar(range(4), h, yerr=[h - lo, hi - h], fmt='o-', color=col, lw=2, ms=8, capsize=4, mfc=col)
    ax.set_xticks(range(4)); ax.set_xticklabels([f'{r}m' for r in R]); ax.set_ylim(.70, 1.2); ax.set_ylabel('HR (1 SD당)'); ax.set_title(nm, loc='left', fontsize=10.5, fontweight='bold', color=INK2)
    for i, (hh, pp) in enumerate(zip(h, sm[k + '_p'])): ax.text(i, .715, f'{hh:.2f}' + ('*' if pp < .05 else ''), ha='center', fontsize=9.5, color=INK, fontweight='bold')
    for r_, hh in zip(R, h): NUM.append(('fig06', f'{k} HR {r_}m', round(float(hh), 3)))
ax = axs[1, 1]; w = .26
for j, (mn, c) in enumerate((('conc_M0', MUTED), ('conc_M1', BLUE), ('conc_M2', ORANGE))):
    ax.bar(np.arange(4) + (j - 1) * w, sm[mn], w * .92, color=c, label=['M0 통제만', 'M1 +경쟁', 'M2 +구조'][j])
    for i, v in enumerate(sm[mn]): ax.text(i + (j - 1) * w, v + .002, f'{v:.3f}', ha='center', fontsize=7.5, color=INK2, rotation=90, va='bottom')
ax.set_ylim(.55, .66); ax.set_xticks(range(4)); ax.set_xticklabels([f'{r}m' for r in R]); ax.set_ylabel('concordance'); ax.legend(frameon=False, fontsize=8.5, loc='upper right', ncol=1); ax.grid(axis='x', visible=False)
ax.set_title('(d) 모델 단계별 구분력', loc='left', fontsize=10.5, fontweight='bold', color=INK2)
fig.suptitle('그림 6. 반경을 바꿔도 같은 결론: 경쟁은 방향만, 구조 지표는 추가 설명력 없음  (* p<0.05)', x=.06, ha='left', fontsize=11.5, fontweight='bold', y=1.0); fig.tight_layout(rect=(0, 0, 1, .97))
save(fig, 'fig06_radius_robustness.png')

# ---------------- fig07 업력 구간별 (H2) ----------------
bands = [('5년 이하', d.age <= 5), ('5~15년', (d.age > 5) & (d.age <= 15)), ('15년 초과', d.age > 15)]
ac = ['log_age', 'log_area', 'log_rent', 'Competitor_C', 'Adjacent_BIZ', 'Clust_Coef']; res = []
for nm, mk in bands:
    x = d[mk]; m, _ = fit(x, ac); sx = m.summary
    res.append((nm, len(x), x.event.mean(), {k: (sx.loc['z_' + k, 'exp(coef)'], sx.loc['z_' + k, 'exp(coef) lower 95%'], sx.loc['z_' + k, 'exp(coef) upper 95%'], sx.loc['z_' + k, 'p']) for k in ('Competitor_C', 'Adjacent_BIZ', 'Clust_Coef')}))
print('업력 구간별:', [(r[0], round(r[3]['Competitor_C'][0], 3), round(r[3]['Adjacent_BIZ'][0], 3)) for r in res])
fig, ax = plt.subplots(figsize=(8.2, 4.5)); ax.axhline(1, color=INK2, lw=1)
for j, (k, nm, c) in enumerate((('Competitor_C', '동종 경쟁', BLUE), ('Adjacent_BIZ', '인접 음식점 수(전체)', ORANGE), ('Clust_Coef', '군집계수(구조)', AQUA))):
    xs = np.arange(3) + (j - 1) * .2; h = np.array([r[3][k][0] for r in res]); lo = np.array([r[3][k][1] for r in res]); hi = np.array([r[3][k][2] for r in res]); p = np.array([r[3][k][3] for r in res])
    for xx, hh, l, u, pp in zip(xs, h, lo, hi, p):
        ax.plot([xx, xx], [l, u], color=c, lw=2); ax.plot(xx, hh, 'o', ms=8, mfc=c if pp < .05 else 'white', mec=c, mew=2); ax.text(xx + .035, hh, f'{hh:.2f}', fontsize=8.5, color=INK2, va='center')
    ax.plot([], [], 'o-', color=c, label=nm)
    for r, hh, pp in zip(res, h, p): NUM.append(('fig07', f'{r[0]} {k} HR', f'{hh:.3f} p={pp:.3f}'))
ax.set_xticks(range(3)); ax.set_xticklabels([f'{r[0]}\n(n={r[1]:,}, 폐업 {r[2]*100:.0f}%)' for r in res]); ax.set_ylabel('HR (구간 내 표준편차 1 증가당)'); ax.set_ylim(.4, 1.4); ax.grid(axis='x', visible=False)
ax.legend(frameon=False, loc='upper left', fontsize=9, ncol=3); ax.text(-.4, .42, '● p<0.05   ○ 유의하지 않음', ha='left', fontsize=8.5, color=MUTED)
ax.set_title('그림 7. 경쟁의 효과는 가게 나이에 따라 다르다 (탐색적, H2)', loc='left', fontsize=11.5, fontweight='bold')
save(fig, 'fig07_age_heterogeneity.png')

# ---------------- fig08 밀도의 비선형성 (5분위) ----------------
def quint(var):
    x = d.copy(); rk = x[var].rank(method='first'); q = pd.qcut(rk, 5, labels=False); x['q'] = q
    rng = [f'{int(x[var][q == i].min())}~{int(x[var][q == i].max())}' for i in range(5)]
    for i in range(1, 5): x[f'q{i}'] = (q == i).astype(float)
    for c in ('log_age', 'log_area', 'log_rent'): x['z_' + c] = z(x[c])
    cols_ = ['z_log_age', 'z_log_area', 'z_log_rent'] + [f'q{i}' for i in range(1, 5)]
    m = CoxPHFitter().fit(x[['duration_days', 'event', '업종'] + cols_], 'duration_days', 'event', strata=['업종']); sx = m.summary
    return rng, [(1, 1, 1, np.nan)] + [(sx.loc[f'q{i}', 'exp(coef)'], sx.loc[f'q{i}', 'exp(coef) lower 95%'], sx.loc[f'q{i}', 'exp(coef) upper 95%'], sx.loc[f'q{i}', 'p']) for i in range(1, 5)]
fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True)
for ax, (var, nm, c) in zip(axs, (('Competitor_C', '(a) 동종 경쟁 점포 수(300m)', BLUE), ('Adjacent_BIZ', '(b) 인접 음식점 수(300m, 전체)', ORANGE))):
    rng, rs = quint(var); h = np.array([r[0] for r in rs]); lo = np.array([r[1] for r in rs]); hi = np.array([r[2] for r in rs])
    ax.axhline(1, color=INK2, lw=1); ax.errorbar(range(5), h, yerr=[h - lo, hi - h], fmt='o-', color=c, lw=2, ms=8, capsize=4)
    ax.set_xticks(range(5)); ax.set_xticklabels([f'Q{i+1}\n{r}곳' for i, r in enumerate(rng)], fontsize=8.5); ax.set_title(nm, loc='left', fontsize=10.5, fontweight='bold', color=INK2); ax.set_ylim(.3, 1.2); ax.grid(axis='x', visible=False)
    for i in range(5): NUM.append(('fig08', f'{var} Q{i+1} HR(기준 Q1)', f'{h[i]:.3f} [{lo[i]:.3f},{hi[i]:.3f}]'))
axs[0].set_ylabel('HR (가장 적은 구간 Q1 = 1, 업력·면적·임대료 통제)')
mq, xq = fit(d, ['log_age', 'log_area', 'log_rent', 'Competitor_C'], extra=[]); xq['z_c2'] = xq['z_Competitor_C'] ** 2
mq2 = CoxPHFitter().fit(xq[['duration_days', 'event', '업종', 'z_log_age', 'z_log_area', 'z_log_rent', 'z_Competitor_C', 'z_c2']], 'duration_days', 'event', strata=['업종']); pq = mq2.summary.loc['z_c2', 'p']
NUM.append(('fig08', '동종 경쟁 이차항 p', round(float(pq), 4))); print('동종 이차항 p =', round(float(pq), 4), 'coef', round(float(mq2.summary.loc['z_c2', 'coef']), 4))
fig.suptitle(f'그림 8. 밀도 효과의 모양(탐색적): U자형이 아니라 최상위 밀도 구간에서 위험이 가장 낮다 (동종 이차항 p={pq:.2f})', x=.06, ha='left', fontsize=11.5, fontweight='bold', y=1.0)
fig.text(.06, -.01, '상권(입지) 효과와 분리되지 않은 연관성이며, 업력·면적·임대료와 업종 층화만 통제함', fontsize=8.5, color=MUTED)
save(fig, 'fig08_density_nonlinear.png')

# ---------------- fig09 군집계수 설명 (실제 점포 예) ----------------
xy = a[['X', 'Y']].values; typ = a['업태구분명'].values; tree = cKDTree(xy); Rm = 300
nbr = [set(j for j in nb if j != i) for i, nb in enumerate(tree.query_ball_point(xy, Rm))]; same = [set(j for j in s if typ[j] == typ[i]) for i, s in enumerate(nbr)]
def cc(i):
    S = same[i]; k = len(S); e = sum(len(same[j] & S) for j in S) // 2; return k, e, (e / (k * (k - 1) / 2) if k > 1 else 0)
kk = np.array([len(s) for s in same])
def distinct(i):   # 같은 건물(동일 좌표) 점포가 겹쳐 보이지 않는 예시만 사용
    pts = {(round(xy[j, 0]), round(xy[j, 1])) for j in same[i] | {i}}; return len(pts) == len(same[i]) + 1
cands = [i for i in range(len(a)) if kk[i] == 6 and distinct(i)]; cvals = np.array([cc(i)[2] for i in cands]); print('예시 후보', len(cands), 'C 범위', cvals.min().round(2), cvals.max().round(2))
ihi, ilo = cands[int(np.argmax(cvals))], cands[int(np.argmin(cvals))]
fd = d.set_index('restaurant_id'); chk = []
for i in (ihi, ilo): chk.append((round(cc(i)[2], 3), round(float(fd.loc[a.loc[i, '관리번호'], 'Clust_Coef']), 3), int(fd.loc[a.loc[i, '관리번호'], 'Competitor_C']), cc(i)[0]))
print('군집계수 검증 (직접계산, 데이터셋값, 데이터셋 k, 직접 k):', chk); assert all(abs(c[0] - c[1]) < 1e-3 and c[2] == c[3] for c in chk)
fig, axs = plt.subplots(1, 2, figsize=(11, 5.4))
for ax, i, ttl in ((axs[0], ihi, '뭉친 경쟁'), (axs[1], ilo, '흩어진 경쟁')):
    k, e, c = cc(i); ax.set_aspect('equal'); ax.grid(False); ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
    cx, cy = xy[i]; ax.add_patch(Circle((cx, cy), Rm, fc='none', ec=AXIS, lw=1.4, ls='--'))
    oth = [j for j in nbr[i] if j not in same[i]]; ax.scatter(xy[oth, 0], xy[oth, 1], s=22, c='#d8d7d0', lw=0, label=f'다른 업태 음식점 {len(oth)}곳')
    S = sorted(same[i]);
    for u in S:
        ax.plot([cx, xy[u, 0]], [cy, xy[u, 1]], color=MUTED, lw=.8, alpha=.7)
    for u in S:
        for v in S:
            if u < v and v in same[u]: ax.plot([xy[u, 0], xy[v, 0]], [xy[u, 1], xy[v, 1]], color=ORANGE, lw=2.4)
    ax.scatter(xy[S, 0], xy[S, 1], s=90, c=BLUE, ec='white', lw=1.2, zorder=3, label=f'같은 업태({typ[i]}) 이웃 {k}곳'); ax.scatter([cx], [cy], s=230, c=INK, marker='*', zorder=4, label='기준 점포')
    ax.set_xlim(cx - Rm * 1.12, cx + Rm * 1.12); ax.set_ylim(cy - Rm * 1.95, cy + Rm * 1.12)
    ax.set_title(f'{ttl}: 군집계수 {c:.2f}', loc='left', fontsize=11.5, fontweight='bold'); ax.text(cx - Rm * 1.1, cy - Rm * 1.17, f'이웃 {k}곳 중 서로 연결된 쌍 {e}개 / 가능한 쌍 {k*(k-1)//2}개', fontsize=9.5, color=INK2)
    ax.legend(loc='lower left', frameon=False, fontsize=8.5, bbox_to_anchor=(0, 0))
    NUM.append(('fig09', f'{ttl} 점포 {a.loc[i, "관리번호"]}', f'k={k}, e={e}, C={c:.3f}'))
fig.suptitle('그림 9. 군집계수란?  이웃 점포끼리 서로 가까운 정도 (주황선 = 이웃끼리도 300m 이내). 이웃 수는 같고 모양만 다르다', x=.05, ha='left', fontsize=11.5, fontweight='bold', y=.99)
save(fig, 'fig09_clustering_example.png')

# ---------------- fig10 SD 구조 (개념도) ----------------
fig, ax = plt.subplots(figsize=(11, 5.2)); ax.set_xlim(0, 11); ax.set_ylim(0, 5.2); ax.axis('off'); ax.grid(False)
ax.add_patch(Circle((0.7, 2.0), .38, fc='#f3f3f0', ec=MUTED, lw=1.5)); ax.text(.7, 1.35, '신규 진입', ha='center', fontsize=9, color=INK2)
arrow(ax, (1.1, 2.0), (3.0, 2.0), BLUE, lw=3); box(ax, 1.55, 1.7, 1.0, .6, '개업 I', 'white', BLUE, fs=10, bold=True)
box(ax, 3.1, 1.2, 2.6, 1.6, '영업 점포 수 N\n(읍면동 × 업종)', '#eaf2fc', BLUE, fs=11.5, bold=True)
arrow(ax, (5.75, 2.0), (8.0, 2.0), ORANGE, lw=3); box(ax, 6.1, 1.7, 1.55, .6, '폐업 N·μ', 'white', ORANGE, fs=10, bold=True)
ax.add_patch(Circle((8.4, 2.0), .38, fc='#f3f3f0', ec=MUTED, lw=1.5)); ax.text(8.4, 1.35, '퇴출', ha='center', fontsize=9, color=INK2)
box(ax, 5.2, 3.9, 5.4, 1.05, 'μ = μ0 · exp( β_임대료·Δ임대료 + β_동종·Δ동종밀도 + … )\nβ는 Cox에서 추정 (HR = exp β)', '#fdeee7', ORANGE, fs=10)
arrow(ax, (6.9, 3.85), (6.9, 2.35), ORANGE, lw=1.6)
box(ax, 0.25, 3.9, 3.9, 1.05, '정책 레버\n임대료 지원·동종 밀집 완화·지원 강도/기간', '#e3f6ef', AQUA, fs=9.5); arrow(ax, (4.2, 4.4), (5.15, 4.4), AQUA, lw=1.6)
arrow(ax, (4.4, 2.85), (5.6, 3.85), VIOLET, rad=-.25, lw=1.8); ax.text(3.75, 3.45, '되먹임: N이 줄면 동종 밀도↓\n→ 폐업률 μ 변화', fontsize=9, color=VIOLET, ha='center')
box(ax, 0.25, 0.1, 10.35, .72, '검증: 2021-04~2026-09 월별 실제 영업 점포 수를 시뮬레이션이 재현하는지 비교  →  통과 후 시나리오 실행', '#f3f3f0', INK2, fs=10)
ax.set_title('그림 10. SD 연결 모델: 재고(점포 수) · 유입 · 유출 · 되먹임', loc='left', fontsize=12, fontweight='bold', pad=4)
save(fig, 'fig10_sd_structure.png')

# ---------------- fig11 월별 점포 수·개업·폐업 (SD 보정/검증 목표) ----------------
lic = rd(os.path.join(RAW, 'restaurant_licenses.csv')); lic['open'] = pd.to_datetime(lic['인허가일자'].astype(str).str.strip(), errors='coerce')
lic['close'] = pd.to_datetime(lic['폐업일자'].astype(str).str.strip().replace({'': None, 'nan': None}), errors='coerce'); lic = lic[lic.open.dt.year >= 1970]
me = pd.date_range('2021-03-31', '2026-08-31', freq='ME')
stock = [int(((lic.open <= t) & (lic.close.isna() | (lic.close > t))).sum()) for t in me]
per = pd.period_range('2021-04', '2026-08', freq='M'); ent = lic[lic.open > B].open.dt.to_period('M').value_counts().reindex(per, fill_value=0); ex = lic[lic.close > B].close.dt.to_period('M').value_counts().reindex(per, fill_value=0)
NUM += [('fig11', '2021-03 말 영업 점포', stock[0]), ('fig11', '2026-08 말 영업 점포', stock[-1]), ('fig11', '기준일 이후 개업 합계(~2026-08)', int(ent.sum())), ('fig11', '기준일 이후 폐업 합계(~2026-08)', int(ex.sum()))]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={'width_ratios': [1, 1.15]})
ax = axs[0]; ax.plot(me, stock, color=BLUE, lw=2.4); ax.set_ylim(3500, 4400); ax.set_ylabel('영업 점포 수 (세로축 3,500부터)'); ax.set_title('(a) 월말 영업 점포 수 (재고 N)', loc='left', fontsize=10.5, fontweight='bold', color=INK2)
ax.text(me[0], stock[0] - 70, f'{stock[0]:,}', color=INK2, fontsize=9.5); ax.text(me[-1], stock[-1] - 70, f'{stock[-1]:,}', color=INK2, fontsize=9.5, ha='right', va='top')
ax = axs[1]; x = per.to_timestamp(); ax.plot(x, ex.values, color=ORANGE, lw=2); ax.plot(x, ent.values, color=BLUE, lw=2); ax.set_ylabel('월간 건수'); ax.set_title('(b) 월간 폐업(유출)과 개업(유입)', loc='left', fontsize=10.5, fontweight='bold', color=INK2)
ax.text(x[3], ex.values[:12].max() + 4, '폐업', color='#b4441a', fontsize=10); ax.text(x[-8], ent.values[-18:].max() + 4, '개업', color=BLUE, fontsize=10); ax.set_ylim(0, max(ex.max(), ent.max()) * 1.2)
fig.suptitle('그림 11. SD가 재현해야 할 실제 경로: 성동구 일반음식점 인허가 전체 (좌표 미확보 점포 포함)', x=.06, ha='left', fontsize=11.5, fontweight='bold', y=1.0)
save(fig, 'fig11_monthly_stock_flows.png')

# ---------------- fig12 HR → 폐업률 배수 (매핑 식의 작동) ----------------
dz = np.linspace(-1.5, 1.5, 61); fig, ax = plt.subplots(figsize=(8.2, 4.6)); ax.axhline(1, color=INK2, lw=1); ends = []
for k, nm, c in (('log_rent', '임대료(읍면동)', VIOLET), ('Competitor_C', '동종 경쟁 점포 수', BLUE), ('Adjacent_BIZ', '인접 음식점 수', ORANGE)):
    r = s.loc['z_' + k]; b, l, u = np.log(r['exp(coef)']), np.log(r['exp(coef) lower 95%']), np.log(r['exp(coef) upper 95%'])
    ax.plot(dz, np.exp(b * dz), color=c, lw=2.4); ax.fill_between(dz, np.exp(l * dz), np.exp(u * dz), color=c, alpha=.12, lw=0)
    ends.append((np.exp(b * 1.5), nm)); NUM.append(('fig12', f'배수 {k} Δz=+1', round(float(np.exp(b)), 3)))
for (v, nm), yl in zip(ends, dodge([e[0] for e in ends], .04)): ax.text(1.55, yl, nm, color=INK2, fontsize=9.5, va='center')
ax.set_xlim(-1.5, 2.7); ax.set_ylim(.7, 1.4); ax.set_xlabel('변수 변화 Δz (표준편차 단위)'); ax.set_ylabel('폐업 위험 배수  exp(β·Δz)')
ax.text(.35, 1.33, '연관성 기반 배수이며 인과가 아님.\n음영은 95% 신뢰구간 → 정책 변수의 효과는 불확실', fontsize=9, color=MUTED, va='top')
ax.set_title('그림 12. 매핑: Cox 계수가 SD 폐업률의 배수가 되는 방식', loc='left', fontsize=11.5, fontweight='bold')
save(fig, 'fig12_mapping_multiplier.png')

pd.DataFrame(NUM, columns=['그림', '항목', '값']).to_csv(os.path.join(RES, 'fig_model_dev_numbers.csv'), index=False, encoding='utf-8-sig'); print('수치 저장 fig_model_dev_numbers.csv')


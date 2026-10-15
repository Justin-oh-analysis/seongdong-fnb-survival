"""확장 분석 시각화 — 입력: results/ext_*.csv (07_extended_analyses.py 산출물) → results/figures/"""
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding='utf-8')
if os.environ.get('NOFIGNUM', '1') != '0':    # 논문 본문에서 그림 번호를 따로 붙이므로 그림 안의 번호는 제거
    import re
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    _st, _su = Axes.set_title, Figure.suptitle
    Axes.set_title = lambda self, label, *a, **k: _st(self, re.sub(r'^그림 \d+\.\s*', '', label), *a, **k)
    Figure.suptitle = lambda self, t, *a, **k: _su(self, re.sub(r'^그림 \d+\.\s*', '', t), *a, **k)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(BASE, 'data', 'processed'); RES = os.path.join(BASE, 'results'); os.makedirs(RES, exist_ok=True); FIG = os.environ.get('FIGDIR') or os.path.join(RES, 'figures'); os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, AQUA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7'
INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'savefig.dpi': 200, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'axes.axisbelow': True, 'font.size': 10})
def save(fig, name): fig.savefig(os.path.join(FIG, name), bbox_inches='tight'); plt.close(fig); print('저장', name)

# ---- fig13: 강건성 (사양별 HR)
R = pd.read_csv(os.path.join(RES, 'robustness_by_spec.csv'))
SP = [('S0 기본 M2', '기본 모형(M2)'), ('S1 렌트 제외', '렌트 제외'), ('S3 M2+군집 고정효과', '군집 고정효과 추가'), ('S4 M2 군집 robust SE', '군집 보정 SE (16개 군집)'), ('S4b M2 군집(동종 270개) robust SE', '군집 보정 SE (270개 군집)'),
      ('S5 M2+확장 구조지표', '확장 구조지표 5개 추가'), ('S6 업력 시간축+좌측절단', '업력 시간축 + 좌측절단')]
PV = [('Competitor_C', '동종 경쟁'), ('Adjacent_BIZ', '인접 음식점'), ('Business_D', '업태 다양성'), ('Clust_Coef', '군집계수(구조)'), ('log_rent', '렌트'), ('log_age', '업력')]
fig, axs = plt.subplots(2, 3, figsize=(12, 7.2), sharey=True)
for ax, (v, nm) in zip(axs.ravel(), PV):
    ax.axvline(1, color=INK2, lw=1)
    for i, (sp, lab) in enumerate(SP):
        r = R[(R.spec == sp) & (R['var'] == v)]
        if r.empty: ax.text(1, i, '해당 없음', fontsize=8, color=MUTED, ha='center', va='center'); continue
        r = r.iloc[0]; sig = r.p < .05; ax.plot([r.lo, r.hi], [i, i], color=BLUE if sig else MUTED, lw=2); ax.plot(r.HR, i, 'o', ms=7, mfc=BLUE if sig else 'white', mec=BLUE if sig else MUTED, mew=1.8)
    ax.set_title(nm, loc='left', fontsize=10.5, fontweight='bold', color=INK2); ax.set_yticks(range(len(SP))); ax.set_yticklabels([s[1] for s in SP], fontsize=9); ax.grid(axis='y', visible=False); ax.set_xscale('log')
    lo = {'log_age': .6, 'log_rent': .75}.get(v, .7); hi = {'log_age': 1.0, 'log_rent': 1.1}.get(v, 1.25); ax.set_xlim(lo, hi); ticks = [t for t in (.6, .7, .8, .9, 1, 1.1, 1.2) if lo <= t <= hi]; ax.set_xticks(ticks); ax.set_xticklabels([f'{t:g}' for t in ticks]); ax.set_xlabel('HR (1 SD당)')
axs[0, 0].invert_yaxis()
fig.suptitle('그림 13. 강건성 검토: 사양을 바꿔도 업력·면적은 유지, 경쟁 효과는 보정 방식에 따라 달라지고 구조 지표는 일관되게 없음  (● p<0.05, ○ 아님)', x=.05, ha='left', fontsize=11.5, fontweight='bold', y=1.0); fig.tight_layout(rect=(0, 0, 1, .97))
save(fig, 'fig13_robustness_specs.png')

# ---- fig14: ML 비교 + 순열 중요도
E = pd.read_csv(os.path.join(RES, 'ext_ml_cv.csv')); I = pd.read_csv(os.path.join(RES, 'ext_ml_importance.csv'))
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.5), gridspec_kw={'width_ratios': [1.25, 1]})
ax = axs[0]; mods = ['로지스틱', '그래디언트부스팅(HGB)', '랜덤포레스트']; w = .26
for j, (fs, c, lab) in enumerate((('기본 8변수+업종', BLUE, '기본 8변수+업종'), ('+확장 구조지표', ORANGE, '+확장 구조지표 5개'))):
    for k, cvn in enumerate(['군집 단위 분할']):
        v = [E[(E.변수 == fs) & (E.모델 == m) & (E.검증 == cvn)].AUC.iloc[0] for m in mods]
        ax.bar(np.arange(3) + (j - .5) * w, v, w * .92, color=c, label=lab)
        for x, y_ in zip(np.arange(3) + (j - .5) * w, v): ax.text(x, y_ + .003, f'{y_:.3f}', ha='center', fontsize=9, color=INK2)
age = E[E.변수 == '업력만'].AUC.iloc[0]; ax.axhline(age, color=INK2, lw=1.2, ls='--', label=f'업력 1개 변수만 사용 (AUC {age:.3f})')
ax.axhline(.5, color=MUTED, lw=1); ax.set_ylim(.5, .66); ax.set_xticks(range(3)); ax.set_xticklabels(['로지스틱', '그래디언트\n부스팅', '랜덤\n포레스트']); ax.set_ylabel('AUC (3년 내 폐업, 군집 단위 5-fold)'); ax.legend(frameon=False, loc='upper right', fontsize=9); ax.grid(axis='x', visible=False)
ax.set_title('(a) 모델 비교: 복잡한 모델도 구분력이 오르지 않는다', loc='left', fontsize=10.5, fontweight='bold', color=INK2)
ax = axs[1]; I = I.sort_values('AUC_감소'); ax.barh(I['변수'], I['AUC_감소'], color=BLUE, height=.6)
for y_, v in enumerate(I['AUC_감소']): ax.text(v + .001, y_, f'{v:.4f}', va='center', fontsize=9, color=INK2)
ax.set_xlabel('순열 중요도 (AUC 감소)'); ax.grid(axis='y', visible=False); ax.set_title('(b) 변수 중요도(그래디언트부스팅)', loc='left', fontsize=10.5, fontweight='bold', color=INK2)
fig.suptitle('그림 14. 머신러닝 교차 확인: Cox와 같은 결론(업력·면적이 핵심, 구조 지표 기여 미미)', x=.05, ha='left', fontsize=11.5, fontweight='bold', y=1.0)
save(fig, 'fig14_ml_comparison.png')

# ---- fig16: 연구 모형 (가설 H1~H4)
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
def bx(ax, x, y, w, h, t, fc, ec, fs=10, bold=False): ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec=ec, lw=1.4)); ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', linespacing=1.5)
def ar(ax, p, q, c=INK2, ls='-', rad=0, lw=1.6): ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=14, color=c, lw=lw, linestyle=ls, connectionstyle=f'arc3,rad={rad}'))
fig, ax = plt.subplots(figsize=(11.5, 5.6)); ax.set_xlim(0, 11.5); ax.set_ylim(0, 5.6); ax.axis('off'); ax.grid(False)
bx(ax, .2, 4.0, 2.9, .95, '점포 특성\n업력 · 면적', '#eaf2fc', BLUE); bx(ax, .2, 2.7, 2.9, .95, '경쟁 밀도 (반경 100~500m)\n동종 · 인접 음식점 · 다양성', '#eaf2fc', BLUE)
bx(ax, .2, 1.4, 2.9, .95, '경쟁 구조 (연결 모양)\n군집계수 · ego 밀도 · 중심성', '#eaf2fc', BLUE); bx(ax, .2, .1, 2.9, .95, '입지 통제\n읍면동 임대료 · 업종 층화', '#f3f3f0', MUTED)
bx(ax, 4.6, 1.9, 2.5, 1.7, '폐업 위험 h(t)\n(Cox 비례위험모형)\n표준편차당 HR', '#fdeee7', ORANGE, bold=True)
bx(ax, 8.7, 1.9, 2.6, 1.7, 'SD 정책 시뮬레이션\n폐업률 μ = μ0·exp(β·Δz)\n점포 수 N(t)', '#e3f6ef', AQUA, bold=True); bx(ax, 8.7, 4.2, 2.6, .9, '정책 시나리오\n임대료 · 동종 밀집 · 지원 강도/기간', 'white', AQUA, fs=9)
ar(ax, (3.15, 4.45), (4.55, 3.4), BLUE); ax.text(3.55, 4.15, 'H1', fontsize=11, fontweight='bold', color=BLUE)
ar(ax, (3.15, 3.15), (4.55, 3.0), BLUE); ax.text(3.65, 3.2, 'H2', fontsize=11, fontweight='bold', color=BLUE)
ar(ax, (3.15, 1.9), (4.55, 2.4), BLUE); ax.text(3.6, 1.8, 'H3', fontsize=11, fontweight='bold', color=BLUE)
ar(ax, (3.15, .6), (4.55, 2.0), MUTED, ls='--'); ax.text(3.45, .75, '통제', fontsize=10, color=MUTED)
ar(ax, (7.15, 2.75), (8.65, 2.75), ORANGE, lw=2.2); ax.text(7.35, 3.0, 'H4', fontsize=11, fontweight='bold', color='#b4441a'); ar(ax, (10.0, 4.15), (10.0, 3.65), AQUA)
ar(ax, (3.15, 4.2), (3.95, 3.12), VIOLET, ls='--', rad=.3); ax.text(4.1, 3.75, '업력에 따라 달라짐', fontsize=9, color=VIOLET)
ax.text(5.85, 1.45, '5분위·이차항·업력 구간·초기/후기로 모양 확인', fontsize=8.5, color=MUTED, ha='center')
ax.set_title('그림 16. 연구 모형: 가설 H1~H4의 경로', loc='left', fontsize=12, fontweight='bold', pad=4)
save(fig, 'fig16_research_model.png')

# ---- fig15: 초기/후기 + 비례위험 검정
T = pd.read_csv(os.path.join(RES, 'ext_ph_timesplit.csv')); P = pd.read_csv(os.path.join(RES, 'ext_ph_test.csv')).set_index('index')
nm2 = ['log_age', 'log_area', 'log_rent', 'Competitor_C', 'Adjacent_BIZ', 'Business_D', 'Clust_Coef', 'Ego_Density']; php = [P.loc['z_' + c, 'p'] for c in nm2]
fig, ax = plt.subplots(figsize=(9, 4.9)); ax.axvline(1, color=INK2, lw=1)
for i, r in T.iterrows():
    ax.plot([r['HR_초기(0-2년)'], r['HR_후기(2년 이후)']], [i, i], color=AXIS, lw=2, zorder=1)
    ax.plot(r['HR_초기(0-2년)'], i, 'o', ms=9, mfc=ORANGE if r.p_초기 < .1 else 'white', mec=ORANGE, mew=2, zorder=2); ax.plot(r['HR_후기(2년 이후)'], i, 'o', ms=9, mfc=BLUE if r.p_후기 < .1 else 'white', mec=BLUE, mew=2, zorder=2)
    ax.text(1.06, i, f'PH 검정 p={php[i]:.3f}' + (' *' if php[i] < .05 else ''), va='center', fontsize=9, color=INK2)
ax.set_yticks(range(len(T))); ax.set_yticklabels(T['var']); ax.invert_yaxis(); ax.grid(axis='y', visible=False); ax.set_xlim(.7, 1.22); ax.set_xlabel('HR (1 SD당)')
ax.plot([], [], 'o', color=ORANGE, label='초기 0~2년'); ax.plot([], [], 'o', color=BLUE, label='후기 2년 이후'); ax.legend(frameon=False, loc='lower left', fontsize=9.5)
fig.text(.13, -.01, '채운 점: p<0.10   빈 점: 아님   * 비례위험 가정 위반 p<0.05', fontsize=8.5, color=MUTED, va='top')
ax.set_title('그림 15. 경쟁 효과는 초기 2년에 집중되고 시간이 지나면 약해진다 (비례위험 가정 위반의 원인)', loc='left', fontsize=11, fontweight='bold')
save(fig, 'fig15_early_late.png')


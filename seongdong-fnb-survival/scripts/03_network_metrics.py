"""① 경쟁 변수 + 네트워크 구조 지표 (시점별 계산 = method B, 누수 방지)
각 점포 i를 개업일(ref_date) 기준으로, 그 시점에 영업 중이던 이웃(반경 RADIUS)으로 계산.
  Competitor_C : 활성 이웃 중 동일 업태구분명 수       Adjacent_BIZ : 활성 이웃 전체 수
  Business_D   : 활성 이웃의 업태 가짓수              Competitor_R : Competitor_C / Adjacent_BIZ (분모 0이면 0)
  Clust_Coef   : 동일업종 ego 군집계수 (이웃 쌍 중 서로 반경 내 비율, k<2면 0)
  NbrDeg_Mean/Max : 동일업종 이웃들의 degree 평균/최대     Ego_Density : ego 네트워크(i 포함) 엣지 밀도
사용: python 03_network_metrics.py [radius_m] [base]    base = orig | expanded (기본 orig)
  orig     = 기존 Tier0 11,805건 (재현 검증: 기존 Tier0 변수와 일치해야 함)
  expanded = 좌표 복원 점포(지오코딩) 포함
"""
import os, sys, glob
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RADIUS = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
KIND = sys.argv[2] if len(sys.argv) > 2 else 'orig'

t = pd.read_csv(os.path.join(BASE, 'data', 'processed', f'tier0_{KIND}_base.csv'), encoding='utf-8-sig')
t['open'] = pd.to_datetime(t['인허가일자'].astype(str).str.strip(), errors='coerce')
t['close'] = pd.to_datetime(t['폐업일자'].astype(str).str.strip().replace({'': None, 'nan': None}), errors='coerce')
assert t['open'].notna().all(), '개업일 결측 존재'
FAR = pd.Timestamp('2262-01-01')
op = t['open'].values.astype('datetime64[ns]').astype(np.int64)
cl = t['close'].fillna(FAR).values.astype('datetime64[ns]').astype(np.int64)
typ = pd.factorize(t['업태구분명'])[0]
xy = t[['X', 'Y']].values
raw = cKDTree(xy).query_ball_point(xy, r=RADIUS)
allc = [np.array([j for j in nb if j != i], dtype=np.int64) for i, nb in enumerate(raw)]
cand = [a[typ[a] == typ[i]] for i, a in enumerate(allc)]
print(f'KIND={KIND} RADIUS={RADIUS}m n={len(t)} 평균 정적 동일업종 후보={np.mean([len(c) for c in cand]):.1f}')

def active(idx, ref):
    return idx[(op[idx] <= ref) & (cl[idx] >= ref)]

cols = ['Competitor_C', 'Adjacent_BIZ', 'Business_D', 'Competitor_R', 'Clust_Coef', 'NbrDeg_Mean', 'NbrDeg_Max', 'Ego_Density']
res = np.zeros((len(t), len(cols)))
for i in range(len(t)):
    ref = op[i]
    A = active(allc[i], ref)
    N = A[typ[A] == typ[i]]
    k = len(N)
    res[i, 0], res[i, 1] = k, len(A)
    res[i, 2] = len(set(typ[A].tolist()))
    res[i, 3] = k / len(A) if len(A) else 0.0
    if k == 0: continue
    Nset = set(N.tolist()); degs, e_among = [], 0
    for j in N:
        nj = active(cand[j], ref)
        degs.append(len(nj)); e_among += sum(1 for m in nj if m in Nset)
    e_among //= 2
    res[i, 4] = e_among / (k * (k - 1) / 2) if k >= 2 else 0.0
    res[i, 5], res[i, 6] = np.mean(degs), np.max(degs)
    res[i, 7] = (k + e_among) / ((k + 1) * k / 2)

out = pd.concat([t.drop(columns=['open', 'close']), pd.DataFrame(res, columns=cols)], axis=1)
if KIND == 'orig' and RADIUS == 300.0:
    t0 = pd.read_csv(os.path.join(BASE, 'data', 'raw', 'initial_competition_vars.csv'), encoding='utf-8-sig')[['관리번호', 'Competitor_C', 'Competitor_R', 'Adjacent_BIZ', 'Business_D']]
    m = out.merge(t0, on='관리번호', suffixes=('', '_t0'))
    for c in ['Competitor_C', 'Adjacent_BIZ', 'Business_D']:
        print(f'재현 검증 {c}: 불일치 {(m[c] != m[c + "_t0"]).sum()}건 / {len(m)}')
    print(f'재현 검증 Competitor_R: 최대 절대오차 {(m["Competitor_R"] - m["Competitor_R_t0"]).abs().max():.2e}')
print(out[cols].describe().round(3).T)
fn = os.path.join(BASE, 'data', 'processed', f'network_{KIND}_r{int(RADIUS)}.csv')
out.to_csv(fn, index=False, encoding='utf-8-sig'); print('saved', fn)

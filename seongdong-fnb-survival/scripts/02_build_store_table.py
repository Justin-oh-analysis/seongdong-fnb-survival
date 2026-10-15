"""복원 좌표를 합친 확장 점포 데이터셋 구축 + 복원/미복원 편향 점검"""
import os, sys, glob
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'data', 'processed'); RES = os.path.join(BASE, 'results'); os.makedirs(RES, exist_ok=True)
def rd(p):
    for e in ('utf-8-sig', 'cp949'):
        try: return pd.read_csv(p, encoding=e, low_memory=False)
        except Exception: pass
lic = rd(os.path.join(BASE, 'data', 'raw', 'restaurant_licenses.csv'))
t0 = rd(os.path.join(BASE, 'data', 'raw', 'initial_competition_vars.csv'))
g = rd(os.path.join(OUT, 'geocode_full.csv'))
lic['X'] = pd.to_numeric(lic['좌표정보(X)'], errors='coerce'); lic['Y'] = pd.to_numeric(lic['좌표정보(Y)'], errors='coerce')
lic['has_xy'] = lic.X.notna() & lic.Y.notna()
lic['closed'] = (lic['영업상태명'] == '폐업').astype(int)
lic['open_year'] = pd.to_datetime(lic['인허가일자'], errors='coerce').dt.year
miss = lic[~lic.has_xy].merge(g[['관리번호', 'success', 'X_5174', 'Y_5174', 'used']], on='관리번호', how='left')
print('결측 점포 중 복원 성공:', int(miss.success.sum()), '/', len(miss))
print('폐업률  복원 성공 vs 실패 vs 원래 좌표 보유')
print('  복원 성공:', miss[miss.success == 1].closed.mean().round(3), '| 복원 실패:', miss[miss.success == 0].closed.mean().round(3),
      '| 원래 보유:', lic[lic.has_xy].closed.mean().round(3))
miss['dec'] = miss.open_year // 10 * 10
print('연대별 복원 성공률:\n', miss.groupby('dec').success.mean().round(3).to_string())
miss['dong'] = miss['지번주소'].astype(str).str.extract(r'성동구\s+(\S+)')
dt = miss.groupby('dong').success.agg(['mean', 'size']); print('동별 복원 성공률(표본≥60):\n', dt[dt['size'] >= 60].sort_values('mean').round(3).to_string())

# 확장 데이터셋: Tier0과 동일 정제(1970 이전 제외, 좌표 필수) 규칙
orig = lic[lic.has_xy & lic['관리번호'].isin(t0['관리번호'])].copy(); orig['src'] = 'orig'
add = lic[~lic.has_xy].merge(g[['관리번호', 'success', 'X_5174', 'Y_5174']], on='관리번호')
add = add[add.success == 1].copy(); add['X'] = add.X_5174; add['Y'] = add.Y_5174; add['src'] = 'geocoded'
cols = ['관리번호', '사업장명', '업태구분명', '영업상태명', '인허가일자', '폐업일자', 'X', 'Y', '소재지면적', 'src']
ex = pd.concat([orig[cols], add[cols]]).rename(columns={'소재지면적': 'Area'})
ex = ex[pd.to_datetime(ex['인허가일자'], errors='coerce').dt.year >= 1970]
print('\n원래 Tier0과 일치:', len(orig), '(기대 11,805) | 복원 추가:', len(add), '| 확장 합계:', len(ex))
ex.to_csv(os.path.join(OUT, 'tier0_expanded_base.csv'), index=False, encoding='utf-8-sig')
orig[cols].rename(columns={'소재지면적': 'Area'}).to_csv(os.path.join(OUT, 'tier0_orig_base.csv'), index=False, encoding='utf-8-sig')

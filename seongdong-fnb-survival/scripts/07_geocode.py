"""좌표 결측 인허가 점포 주소 → VWorld 지오코딩 (캐시·재개·중단 안전장치)
사용: python 07_geocode.py pilot 100   |   python 07_geocode.py full
  - 도로명주소 우선, 실패/성동구 아님이면 지번주소 재시도
  - 응답 refined 텍스트에 '성동구'가 있을 때만 채택 (이전 시험에서 ~25%가 38km 오매칭)
  - 주소 중복 제거, 호출 상한(CAP), 인증/한도 오류 시 즉시 중단, 결과는 jsonl 캐시에 즉시 저장
키 값은 출력하지 않음 (환경변수 GEOCODE_API_KEY, 앞뒤 공백 제거)
"""
import os, sys, glob, json, time, urllib.parse, urllib.request
import numpy as np, pandas as pd
from pyproj import Transformer
sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'out'); os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, 'geocode_cache.jsonl')
mode = sys.argv[1] if len(sys.argv) > 1 else 'pilot'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
CAP = 4000          # 이번 실행 호출 상한
DELAY = 0.12
key = os.environ.get('GEOCODE_API_KEY', '').strip()
if not key: print('[안내] GEOCODE_API_KEY 없음: out/geocode_cache.jsonl 의 캐시만 사용합니다')
to5174 = Transformer.from_crs(4326, 5174, always_xy=True)

def rd(p):
    for e in ('utf-8-sig', 'cp949'):
        try: return pd.read_csv(p, encoding=e, low_memory=False)
        except Exception: pass
lic = rd(glob.glob(os.path.join(BASE, 'raw', '*일반음식점*'))[0])
x = pd.to_numeric(lic['좌표정보(X)'], errors='coerce'); y = pd.to_numeric(lic['좌표정보(Y)'], errors='coerce')
miss = lic[x.isna() | y.isna()].copy()
for c in ('도로명주소', '지번주소'):
    miss[c] = miss[c].astype(str).str.strip().replace({'nan': '', 'None': ''})
print(f'좌표 결측 {len(miss)}건 | 도로명 있음 {(miss.도로명주소 != "").sum()} | 지번 있음 {(miss.지번주소 != "").sum()} | 둘 다 없음 {((miss.도로명주소 == "") & (miss.지번주소 == "")).sum()}')

cache = {}
if os.path.exists(CACHE):
    for line in open(CACHE, encoding='utf-8'):
        r = json.loads(line); cache[(r['type'], r['addr'])] = r
print('캐시 기존', len(cache), '건')
calls = 0; consec_err = 0

def call(addr, typ):
    global calls, consec_err
    k = (typ, addr)
    if k in cache: return cache[k]
    if calls >= CAP: raise SystemExit(f'호출 상한 {CAP} 도달 — 중단(재실행하면 캐시 이어서 진행)')
    if not key: raise SystemExit('캐시에 없는 주소를 조회하려면 환경변수 GEOCODE_API_KEY(VWorld)가 필요합니다')
    q = urllib.parse.urlencode(dict(service='address', request='getcoord', version='2.0', crs='epsg:4326', address=addr,
                                    refine='true', simple='false', format='json', type=typ, key=key))
    rec = dict(type=typ, addr=addr, status='ERR', x=None, y=None, refined='')
    for attempt in range(3):
        try:
            j = json.loads(urllib.request.urlopen('https://api.vworld.kr/req/address?' + q, timeout=20).read().decode('utf-8'))
            calls += 1; consec_err = 0; resp = j.get('response', {}); st = resp.get('status')
            if st == 'OK':
                p = resp['result']['point']
                rec.update(status='OK', x=float(p['x']), y=float(p['y']), refined=resp.get('refined', {}).get('text', ''))
            else:
                txt = resp.get('error', {}).get('text', '')
                rec.update(status=st or 'ERR', refined=txt)
                if st == 'ERROR' and any(w in txt for w in ('인증', '키', '한도', '초과', '제한', 'limit', 'LIMIT')):
                    raise SystemExit(f'인증/한도 오류로 즉시 중단: {txt}')
            break
        except SystemExit: raise
        except Exception as e:
            consec_err += 1; time.sleep(1.5)
            if consec_err >= 10: raise SystemExit('연속 네트워크 오류 10회 — 중단')
    cache[k] = rec
    with open(CACHE, 'a', encoding='utf-8') as f: f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    time.sleep(DELAY)
    return rec

import re
def clean(a):
    a = re.sub(r'\([^)]*\)', ' ', a)                       # 괄호 표기 제거: (지상1층), (행당동) 등
    a = re.sub(r'\s+\d+\s*층.*$', '', a)                    # 뒤쪽 층/호 표기
    a = re.sub(r'(\d+)-0\b', r'\1', a)                      # '100-0' -> '100'
    return re.sub(r'\s+', ' ', a).strip()

def ok(rec): return rec['status'] == 'OK' and '성동구' in rec['refined']

targets = miss.sample(N, random_state=11) if mode == 'pilot' else miss
rows = []
for n, (_, r) in enumerate(targets.iterrows(), 1):
    got, used = None, ''
    for typ, addr0 in (('road', r['도로명주소']), ('parcel', r['지번주소'])):
        if not addr0: continue
        for addr in dict.fromkeys([addr0, clean(addr0)]):       # 원문 → 정제본 순서, 중복 제거
            if not addr: continue
            rec = call(addr, typ)
            if ok(rec): got, used = rec, typ + ('_clean' if addr != addr0 else ''); break
        if got: break
    X5174 = Y5174 = np.nan
    if got: X5174, Y5174 = to5174.transform(got['x'], got['y'])
    rows.append(dict(관리번호=r['관리번호'], used=used, success=int(got is not None), X_5174=X5174, Y_5174=Y5174,
                     refined=got['refined'] if got else ''))
    if n % 500 == 0: print(f'  진행 {n}/{len(targets)} | 이번 실행 호출 {calls}', flush=True)
df = pd.DataFrame(rows)
fn = os.path.join(OUT, f'geocode_{mode}.csv'); df.to_csv(fn, index=False, encoding='utf-8-sig')
print(f'\n[{mode}] 대상 {len(df)} | 성공 {df.success.sum()} ({df.success.mean():.1%}) | 사용: {df.used.value_counts().to_dict()} | 이번 실행 실제 호출 {calls}')
b = df[df.success == 1]
if len(b): print('좌표 범위 X_5174', b.X_5174.min().round(0), b.X_5174.max().round(0), '| Y_5174', b.Y_5174.min().round(0), b.Y_5174.max().round(0))
print('저장:', fn)

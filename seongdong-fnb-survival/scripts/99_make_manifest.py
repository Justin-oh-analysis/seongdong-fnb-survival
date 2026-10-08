"""원자료(raw/)·산출물(out/) 목록 + MD5 체크섬 생성 / 검증
  python 99_make_manifest.py            -> DATA_MANIFEST.csv, OUTPUT_MANIFEST.csv, requirements.txt 생성
  python 99_make_manifest.py --verify   -> 현재 raw/ 파일이 DATA_MANIFEST.csv의 MD5와 같은지 검증 (원자료 변조 확인)
원자료(raw/)는 절대 수정하지 않는다. 가공 결과는 모두 out/ 에 저장된다."""
import os, sys, hashlib, io, glob, csv
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW, OUT = os.path.join(BASE, 'raw'), os.path.join(BASE, 'out')
MAN = os.path.join(BASE, 'DATA_MANIFEST.csv')

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

# 출처 정보: 프로젝트 브리핑·팀 노션에 적힌 내용 그대로 (URL·다운로드일은 확인되지 않아 비워 둠 → 사람이 채워야 함)
SRC = [
    ('final_dataset', '팀원 구축 데이터셋(상권 코드·매출·렌트·리뷰 결합) — 구축 방식은 팀 확인 필요', '기준일 2021-04-01', '결과·업력·경쟁(8개 업종 기준)·렌트·리뷰 포함'),
    ('인허가_정보', '지방행정인허가데이터(브리핑) / 서울 열린데이터광장(팀 노션) — 출처 표기 확인 필요', '1970~2026', '성동구 일반음식점 인허가 원본 (15,200행)'),
    ('Tier0', '자체 산출(원자료: 일반음식점 인허가)', '-', '초기 산출본. 최신 변수는 out/network_*.csv (02_network_metrics.py)'),
    ('점포-상권', '서울 열린데이터광장 · 서울시 상권분석서비스', '2021~2025 분기별', '상권×업종 점포수·개폐업률 (2025년 파일은 영문 컬럼명)'),
    ('추정매출-상권', '서울 열린데이터광장 · 서울시 상권분석서비스', '2021~2025 분기별', '상권×업종 매출'),
    ('상주인구', '서울 열린데이터광장 · 서울시 상권분석서비스', '2021Q1~2026Q2', ''),
    ('직장인구', '서울 열린데이터광장 · 서울시 상권분석서비스', '2021Q1~2026Q2', ''),
    ('길단위인구', '서울 열린데이터광장 · 서울시 상권분석서비스', '2021Q1~2026Q2', '유동인구'),
    ('집객시설', '서울 열린데이터광장 · 서울시 상권분석서비스', '2021Q1~2026Q2', ''),
    ('영역-상권', '서울 열린데이터광장 · 서울시 상권분석서비스 (SHP, EPSG:5181)', '-', '상권 폴리곤 1,650개'),
    ('상가임대료', '공공데이터포털(서울특별시 성동구) — 팀 노션 표기', '2015.05~2021.11', '평당전세가, 단위 확인 필요'),
    ('임대동향', '한국부동산원 — 초안 표기', '2024Q3~2026Q2', '천원/㎡, 서울 64개 상권'),
    ('버스정류소', '서울 열린데이터광장 — 팀 노션 표기', '2026.09.02', 'WGS84'),
    ('도시철도', '공공데이터포털(국가철도공단) — 팀 노션 표기', '2026.06.30', 'WGS84'),
    ('naver_seongsu', '네이버 플레이스 (크롤링 파일럿 3곳 581건)', '2026', '교수님 방침: 확대 크롤링 안 함'),
]
def src_of(name):
    for k, s, p, n in SRC:
        if k in name: return s, p, n
    return '', '', ''

def enc_and_shape(p):
    if p.lower().endswith('.xlsx'):
        d = pd.read_excel(p); return 'xlsx', len(d), d.shape[1]
    if p.lower().endswith('.csv'):
        b = open(p, 'rb').read()
        for e in ('utf-8-sig', 'cp949'):
            try:
                n = b.decode(e).count('\n'); return e, max(n - 1, 0), '-'
            except UnicodeDecodeError: pass
    return '-', '-', '-'

def raw_files():
    return sorted(os.path.join(d, f) for d, _, fs in os.walk(RAW) for f in fs)

if '--verify' in sys.argv:
    saved = pd.read_csv(MAN, encoding='utf-8-sig')
    ok = True
    for _, r in saved.iterrows():
        p = os.path.join(BASE, r['경로'])
        if not os.path.exists(p): print('없음!', r['경로']); ok = False; continue
        if md5(p) != r['md5']: print('변경됨!', r['경로']); ok = False
    print('원자료 검증:', '모든 파일 일치 (변조 없음)' if ok else '불일치 있음'); sys.exit(0 if ok else 1)

rows = []
for p in raw_files():
    e, n, c = enc_and_shape(p); s, per, note = src_of(os.path.basename(p))
    rows.append(dict(경로=os.path.relpath(p, BASE), 크기_MB=round(os.path.getsize(p) / 1e6, 2), md5=md5(p), 인코딩=e, 행수_근사=n, 출처_기재=s, 기간=per, 비고=note,
                     다운로드_URL='(작성 필요)', 다운로드일='(작성 필요)'))
pd.DataFrame(rows).to_csv(MAN, index=False, encoding='utf-8-sig'); print('saved DATA_MANIFEST.csv', len(rows), '개 원자료')

GEN = {'final_dataset_baseline2021_r300.csv': '21_build_final_dataset.py', 'baseline2021_dataset.csv': '19_baseline2021_network_cox.py', 'baseline2021_cox_summary.csv': '19_baseline2021_network_cox.py', 'tier0_trdar.csv': '01_spatial_join.py', 'geocode_full.csv': '07_geocode.py full', 'geocode_pilot.csv': '07_geocode.py pilot 100', 'geocode_cache.jsonl': '07_geocode.py (VWorld 응답 캐시, 원 API 결과)',
       'tier0_expanded_base.csv': '08_build_expanded.py', 'tier0_orig_base.csv': '08_build_expanded.py', 'ego.json': '14_pick_ego_examples.py', 'cluster_map.png': '15_cluster_map.py',
       'radius_robustness.csv': '10_radius_summary.py', 'radius_robustness.png': '10_radius_summary.py', 'summary_chart.png': '05_chart.py', 'meeting_summary.png': '11_meeting_figure.py',
       'meeting_summary.pdf': '11_meeting_figure.py', 'tier0_network_r300.csv': '(폐기: 지오코딩 이전 구버전, out/_deprecated/로 이동)'}
orows = []
for p in sorted(glob.glob(os.path.join(OUT, '**', '*'), recursive=True)):
    if os.path.isdir(p) or os.sep + 'logs' + os.sep in p: continue
    n = os.path.basename(p)
    g = GEN.get(n) or ('02_network_metrics.py ' + n.split('_r')[-1].replace('.csv', '') + ' ' + n.split('_')[1] if n.startswith('network_') else ('19_baseline2021_network_cox.py' if n.startswith('baseline2021_network') else ''))
    orows.append(dict(파일=os.path.relpath(p, BASE), md5=md5(p), 크기_KB=round(os.path.getsize(p) / 1e3), 생성_스크립트=g))
pd.DataFrame(orows).to_csv(os.path.join(BASE, 'OUTPUT_MANIFEST.csv'), index=False, encoding='utf-8-sig'); print('saved OUTPUT_MANIFEST.csv', len(orows), '개 산출물')

import importlib.metadata as im
pk = ['pandas', 'numpy', 'scipy', 'geopandas', 'shapely', 'pyproj', 'pyogrio', 'networkx', 'lifelines', 'matplotlib', 'openpyxl', 'pypdf']
with open(os.path.join(BASE, 'requirements.txt'), 'w', encoding='utf-8') as f:
    f.write(f'# Python {sys.version.split()[0]} (Windows 11) 에서 실행·검증한 버전\n')
    for k in pk:
        try: f.write(f'{k}=={im.version(k)}\n')
        except Exception: pass
print('saved requirements.txt')

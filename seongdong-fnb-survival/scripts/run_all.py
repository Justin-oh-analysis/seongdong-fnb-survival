"""전체 재현: data/raw (원자료) -> data/processed (최종 데이터셋) -> results (표·그림).
  python scripts/run_all.py
지오코딩은 data/raw/geocode_cache.jsonl 의 저장된 응답을 쓰므로 외부 API 호출과 키가 필요 없다.
실행 후 이전 산출물과 MD5를 비교해 같은 결과인지 알려 준다(처음 실행이면 비교 대상이 없음)."""
import os, sys, subprocess, hashlib, time, glob
sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
OUT = os.path.join(BASE, 'data', 'processed'); RES = os.path.join(BASE, 'results'); LOG = os.path.join(OUT, 'logs'); os.makedirs(LOG, exist_ok=True)
STEPS = [
    ('01 좌표 복원(주소 지오코딩, 캐시 사용)', '01_restore_coordinates.py', ['full']),
    ('02 점포 테이블 구축(좌표 복원 포함)', '02_build_store_table.py', []),
    ('03 경쟁 네트워크 지표(개업 시점, 300m)', '03_network_metrics.py', ['300', 'expanded']),
    ('04 기준일 네트워크 지표 + 반경별 Cox', '04_baseline_cox.py', []),
    ('05 최종 데이터셋 구축', '05_build_final_dataset.py', []),
    ('06 업력 구간별 효과·상호작용', '06_age_heterogeneity.py', []),
    ('07 확장 분석(강건성·ML·리뷰·구조지표)', '07_extended_analyses.py', []),
    ('08 그림(기본)', '08_figures_main.py', []),
    ('09 그림(확장)', '09_figures_extended.py', []),
]
if not os.path.exists(os.path.join(BASE, 'data', 'raw', 'geocode_cache.jsonl')): sys.exit('data/raw/geocode_cache.jsonl 없음')
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
watch = lambda: {os.path.relpath(p, BASE): md5(p) for d in (OUT, RES) for p in glob.glob(os.path.join(d, '*.csv')) + glob.glob(os.path.join(d, '*.jsonl'))}
before = watch(); t0 = time.time(); failed = []
for name, script, args in STEPS:
    t1 = time.time()
    r = subprocess.run([sys.executable, '-W', 'ignore', os.path.join(HERE, script)] + args, capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=HERE)
    open(os.path.join(LOG, script[:-3] + '.log'), 'w', encoding='utf-8').write(r.stdout + '\n---STDERR---\n' + r.stderr)
    print(f'[{"OK  " if r.returncode == 0 else "FAIL"}] {name} ({time.time() - t1:.0f}s)')
    if r.returncode != 0: failed.append(name); print(r.stderr[-500:])
after = watch()
same = [k for k in before if k in after and before[k] == after[k]]; diff = [k for k in before if k in after and before[k] != after[k]]
print(f'\n총 {time.time() - t0:.0f}초 | 실패 {len(failed)} | [재현 대조] 이전 산출물과 동일 {len(same)}개 / 달라짐 {len(diff)}개')
for k in diff: print('  달라진 파일:', k)

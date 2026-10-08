"""저장소 전체 재현: raw/ (원자료) -> out/ (최종 데이터셋·결과). 실행 후 이전 산출물과 MD5를 비교해 같은 결과인지 확인한다.
  python scripts/run_all.py
지오코딩(07)은 out/geocode_cache.jsonl 의 VWorld 응답 캐시를 사용하므로 외부 API 호출·API 키가 필요 없다."""
import os, sys, subprocess, hashlib, time, glob
sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
OUT = os.path.join(BASE, 'out'); LOG = os.path.join(OUT, 'logs'); os.makedirs(LOG, exist_ok=True)
STEPS = [
    ('원자료 체크섬 검증', '99_make_manifest.py', ['--verify']),
    ('01 공간조인 (좌표계 5174->5181)', '01_spatial_join.py', []),
    ('07 지오코딩 복원 (캐시 사용)', '07_geocode.py', ['full']),
    ('08 확장 점포 데이터셋', '08_build_expanded.py', []),
    ('02 네트워크 지표 (orig, 300m)', '02_network_metrics.py', ['300', 'orig']),
    ('02 네트워크 지표 (expanded, 300m)', '02_network_metrics.py', ['300', 'expanded']),
    ('19 기준일 2021-04-01 네트워크 지표 + Cox', '19_baseline2021_network_cox.py', []),
    ('21 최종 데이터셋 구축', '21_build_final_dataset.py', []),
    ('22 Cox vs 3년 내 폐업 로지스틱', '22_cox_vs_logit.py', []),
]
if not os.path.exists(os.path.join(OUT, 'geocode_cache.jsonl')): sys.exit('out/geocode_cache.jsonl 없음')
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
watch = lambda: {os.path.basename(p): md5(p) for p in glob.glob(os.path.join(OUT, '*.csv')) + glob.glob(os.path.join(OUT, '*.jsonl'))}
before = watch(); t0 = time.time(); failed = []
for name, script, args in STEPS:
    t1 = time.time()
    r = subprocess.run([sys.executable, '-W', 'ignore', os.path.join(HERE, script)] + args, capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=HERE)
    open(os.path.join(LOG, 'run_all_' + (script[:-3] + '_' + '_'.join(args)).strip('_') + '.log'), 'w', encoding='utf-8').write(r.stdout + '\n---STDERR---\n' + r.stderr)
    print(f'[{"OK  " if r.returncode == 0 else "FAIL"}] {name} ({time.time() - t1:.0f}s)')
    if r.returncode != 0: failed.append(name); print(r.stderr[-500:])
after = watch()
same = [k for k in before if k in after and before[k] == after[k]]; diff = [k for k in before if k in after and before[k] != after[k]]
print(f'\n총 {time.time() - t0:.0f}초 | 실패 {len(failed)} | [재현 대조] 이전 산출물과 동일 {len(same)}개 / 달라짐 {len(diff)}개')
for k in diff: print('  달라진 파일:', k)

"""네이버 플레이스 성동구 음식점 목록 + 최신 리뷰 수집 (Windows / VS Code 용, 원본: naver_crawling2 (1).ipynb)

사용법
  1) 아래 GROUP_NUMBER 를 5, 6, 7, 8 중 하나로 바꾸고 실행 (VS Code 오른쪽 위 ▶ 또는 터미널에서 python crawl_naver_reviews.py)
  2) 또는 터미널에서 번호를 직접 지정:  python crawl_naver_reviews.py 6
  3) 중간에 멈췄다면 같은 그룹 번호로 다시 실행하면 이미 받은 매장은 건너뛰고 이어서 수집한다.

사전 준비 (1회):  pip install playwright pandas   →   playwright install chromium
결과 저장 위치:   ../raw/reviews/seongdong_group{번호}_restaurants.csv, ..._reviews.csv  (수집일 crawl_date 열 포함)
"""
import asyncio
import hashlib
import os
import re
import sys
from datetime import date

import pandas as pd
from playwright.async_api import async_playwright

sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows 콘솔 한글/특수문자 깨짐 방지

# ============================================================
# 그룹별 키워드 구성 (기존 1~4 유지 + 신규 5~8 추가)
# ============================================================
DONG_GROUPS = {
    # --- [기존 그룹] ---
    1: [
        "성수동 음식점", "성수동 카페"
    ],
    2: [
        "왕십리 음식점", "왕십리 카페",
        "행당동 음식점", "행당동 카페",
        "마장동 음식점", "마장동 카페"
    ],
    3: [
        "금호동 음식점", "금호동 카페",
        "옥수동 음식점", "옥수동 카페"
    ],
    4: [
        "사근동 음식점", "사근동 카페",
        "응봉동 음식점", "응봉동 카페",
        "용답동 음식점", "용답동 카페",
        "송정동 음식점", "송정동 카페"
    ],
    5: [
        "성수역 음식점", "성수역 카페",
        "뚝섬역 음식점", "뚝섬역 카페",
        "서울숲역 음식점", "서울숲역 카페",
        "서울숲 음식점", "서울숲 카페",
        "연무장길 음식점", "연무장길 카페"
    ],
    6: [
        "왕십리역 음식점", "왕십리역 카페",
        "상왕십리역 음식점", "상왕십리역 카페",
        "한양대 음식점", "한양대 카페",
        "한양대역 음식점", "한양대역 카페",
        "행당역 음식점", "마장역 음식점"
    ],
    7: [
        "금호역 음식점", "금호역 카페",
        "신금호역 음식점", "신금호역 카페",
        "옥수역 음식점", "옥수역 카페",
        "응봉역 음식점"
    ],
    8: [
        "용답역 음식점", "용답역 카페",
        "답십리역 음식점", "답십리역 카페"
    ]
}

# ------------------------------------------------------------
# 💡 돌리고 싶은 그룹 번호를 선택하세요 (1~8 중 선택)  ← 여기만 바꾸면 됩니다
# ------------------------------------------------------------
GROUP_NUMBER = 5
if len(sys.argv) > 1 and sys.argv[1].isdigit():
    GROUP_NUMBER = int(sys.argv[1])                  # 터미널에서 번호를 주면 그 값을 우선 사용

HEADLESS = True                    # 브라우저 창을 보면서 확인하려면 False (선택자가 깨졌을 때 디버깅용)
COLLECT_ADDRESS = True             # 매장 주소 수집 (인허가 점포와 매칭하려면 필요). 시간이 부담되면 False
MAX_PAGES_PER_QUERY = 5            # 검색어당 최대 5페이지
MAX_REVIEWS_PER_RESTAURANT = 10    # 매장당 최신 리뷰 개수 (교수님 지시: 업체당 최신순 10개)

TARGET_DONG_LIST = DONG_GROUPS[GROUP_NUMBER]
CRAWL_DATE = date.today().isoformat()                # 수집일 (원본 데이터 출처 기록용)

# 저장 경로: 이 파일(scripts/) 기준 ../raw/reviews/  (원자료는 raw/ 아래에 수정 없이 보관)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "raw", "reviews")
os.makedirs(OUT_DIR, exist_ok=True)
RESTAURANT_LIST_CSV = os.path.join(OUT_DIR, f"seongdong_group{GROUP_NUMBER}_restaurants.csv")
REVIEW_OUTPUT_CSV = os.path.join(OUT_DIR, f"seongdong_group{GROUP_NUMBER}_reviews.csv")


# ============================================================
# 헬퍼 함수
# ============================================================
def make_review_id(place_id, review_text, review_date):
    text = f"{place_id}|{review_date}|{review_text}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def extract_full_date(full_text):
    match = re.search(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*[월화수목금토일]요일", full_text)
    if not match: return None
    return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"

def extract_rating(full_text):
    match = re.search(r"별점\s*(\d(?:\.\d+)?)\s*점", full_text)
    return float(match.group(1)) if match else None

def extract_waiting_time(full_text):
    match = re.search(r"이용대기 시간\s*(바로 입장|\d+\s*분 이내|\d+\s*시간 이내)", full_text)
    return match.group(1) if match else None

def extract_reservation(full_text):
    if "예약 없이 이용" in full_text: return "예약 없이 이용"
    if "예약 후 이용" in full_text: return "예약 후 이용"
    return None

def extract_verification(full_text):
    if "인증 수단" not in full_text: return None
    part = full_text.split("인증 수단")[-1].strip()
    if part.startswith("영수증"): return "영수증"
    if part.startswith("예약"): return "예약"
    return None


# ============================================================
# Step 1: 목록 수집
# ============================================================
async def collect_restaurant_list(page, search_query):
    print(f"\n[검색 시작] 키워드: {search_query}")
    await page.goto("https://map.naver.com/p/", wait_until="domcontentloaded", timeout=30000)
    await page.wait_for_timeout(2500)

    search_inputs = page.locator("input.input_search")
    search_input = None
    for i in range(await search_inputs.count()):
        candidate = search_inputs.nth(i)
        if await candidate.is_visible():
            search_input = candidate
            break

    if not search_input: return []

    await search_input.fill(search_query)
    await search_input.press("Enter")
    await page.wait_for_timeout(3500)

    try:
        await page.wait_for_selector("iframe#searchIframe", timeout=15000)
    except Exception:
        return []

    iframe_element = await page.query_selector("iframe#searchIframe")
    frame = await iframe_element.content_frame()

    collected_data = []
    page_num = 1
    last_page_first_name = ""

    while page_num <= MAX_PAGES_PER_QUERY:
        print(f"  └ [{search_query}] {page_num}/{MAX_PAGES_PER_QUERY}페이지 스크롤 및 수집 중...")

        previous_count = 0
        no_change_count = 0
        for scroll in range(15):
            restaurants = frame.locator("li.UEzoS")
            current_count = await restaurants.count()
            if current_count == previous_count:
                no_change_count += 1
            else:
                no_change_count = 0

            if no_change_count >= 3: break
            previous_count = current_count

            await frame.evaluate("""
                () => {
                    const scrollContainer = document.querySelector('#_pcmap_list_scroll_container') || document.querySelector('.m2A33');
                    if (scrollContainer) scrollContainer.scrollTop = scrollContainer.scrollHeight;
                    else window.scrollTo(0, document.body.scrollHeight);
                }
            """)
            await frame.wait_for_timeout(600)

        restaurants = frame.locator("li.UEzoS")
        restaurant_count = await restaurants.count()
        if restaurant_count == 0: break

        first_name_el = restaurants.nth(0).locator("span.TYaxT")
        if await first_name_el.count() > 0:
            current_first_name = await first_name_el.first.inner_text()
            if current_first_name == last_page_first_name:
                print("  └ [안내] 마지막 페이지 도달.")
                break
            last_page_first_name = current_first_name

        for i in range(restaurant_count):
            try:
                restaurants = frame.locator("li.UEzoS")
                restaurant = restaurants.nth(i)
                name_el = restaurant.locator("span.TYaxT")
                if await name_el.count() == 0: continue
                name = await name_el.first.inner_text()

                cat_el = restaurant.locator("span.KCMnt")
                category = await cat_el.first.inner_text() if await cat_el.count() > 0 else None

                old_place_id = None
                old_match = re.search(r"/place/(\d+)", page.url)
                if old_match: old_place_id = old_match.group(1)

                await name_el.first.click()

                place_id = None
                for _ in range(8):
                    await page.wait_for_timeout(150)
                    match = re.search(r"/place/(\d+)", page.url)
                    if match and (old_place_id is None or match.group(1) != old_place_id):
                        place_id = match.group(1)
                        break

                if place_id is None:
                    entry_iframe = page.locator("iframe#entryIframe")
                    if await entry_iframe.count() > 0:
                        src = await entry_iframe.first.get_attribute("src")
                        if src:
                            match = re.search(r"/(?:restaurant|place)/(\d+)", src)
                            if match and (old_place_id is None or match.group(1) != old_place_id):
                                place_id = match.group(1)

                if place_id:
                    collected_data.append({"place_id": str(place_id), "name": name, "category": category})
            except Exception:
                continue

        page_num += 1
        if page_num > MAX_PAGES_PER_QUERY: break

        target_num_btn = frame.locator(f"a:has-text('{page_num}')")
        if await target_num_btn.count() > 0 and await target_num_btn.first.is_visible():
            await target_num_btn.first.click()
            await page.wait_for_timeout(2000)
        else:
            break

    print(f"  ⇒ [{search_query}] 수집 매장 수: {len(collected_data)}개")
    return collected_data


# ============================================================
# (추가) 매장 주소 수집 — 인허가 점포와 연결하기 위한 키
#   선택자(클래스명) 대신 홈 탭 본문에서 '서울 성동구 ...' 패턴을 찾는다. 실패하면 None (수집은 계속됨)
# ============================================================
ADDR_RE = re.compile(r"서울(?:특별시)?\s*성동구\s+[^\n\r]{2,60}")

async def fetch_address(page, place_id):
    try:
        await page.goto(f"https://pcmap.place.naver.com/restaurant/{place_id}/home", wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(1200)
        body = await page.locator("body").inner_text()
        m = ADDR_RE.search(body)
        return m.group(0).strip() if m else None
    except Exception:
        return None


# ============================================================
# Step 2: 리뷰 수집
# ============================================================
async def sort_by_latest(page):
    buttons = page.locator("a.place_btn_option")
    for i in range(await buttons.count()):
        button = buttons.nth(i)
        try:
            text = (await button.inner_text()).strip()
            if "최신순" in text:
                await button.click()
                await page.wait_for_timeout(1000)
                return True
        except Exception:
            continue
    return False

async def crawl_one_restaurant_reviews(page, place_id, restaurant_name, category, address, max_reviews):
    url = f"https://pcmap.place.naver.com/restaurant/{place_id}/review/visitor"
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_selector("ul#_review_list", timeout=12000)
        await page.wait_for_timeout(800)
    except Exception:
        return []

    await sort_by_latest(page)

    no_growth_count = 0
    while True:
        review_items = page.locator("ul#_review_list > li.EjjAW")
        loaded_count = await review_items.count()
        if loaded_count >= max_reviews: break

        more_button = page.locator("div.NSTUp a.fvwqf")
        if await more_button.count() == 0: break

        try:
            button = more_button.first
            await button.scroll_into_view_if_needed()
            old_count = loaded_count
            await button.click()

            try:
                await page.wait_for_function(
                    "oldCount => document.querySelectorAll('ul#_review_list > li.EjjAW').length > oldCount",
                    arg=old_count,
                    timeout=2500
                )
            except Exception:
                await page.wait_for_timeout(600)

            new_count = await page.locator("ul#_review_list > li.EjjAW").count()
            if new_count <= old_count: no_growth_count += 1
            else: no_growth_count = 0

            if no_growth_count >= 3: break
        except Exception:
            break

    review_items = page.locator("ul#_review_list > li.EjjAW")
    collect_count = min(await review_items.count(), max_reviews)

    collected_reviews = []
    seen_ids = set()

    for i in range(collect_count):
        review = review_items.nth(i)
        try:
            full_text = (await review.inner_text()).strip()
            review_date = extract_full_date(full_text)

            content_el = review.locator("div.pui__vn15t2 > a")
            review_text = (await content_el.first.inner_text()).strip() if await content_el.count() > 0 else ""
            if not review_text: continue

            author_el = review.locator("span.pui__NMi-Dp")
            author = (await author_el.first.inner_text()).strip() if await author_el.count() > 0 else None

            review_id = make_review_id(place_id, review_text, review_date)
            if review_id in seen_ids: continue
            seen_ids.add(review_id)

            collected_reviews.append({
                "review_id": review_id,
                "place_id": str(place_id),
                "restaurant_name": restaurant_name,
                "category": category,
                "address": address,
                "author": author,
                "review_date": review_date,
                "rating": extract_rating(full_text),
                "waiting_time": extract_waiting_time(full_text),
                "reservation": extract_reservation(full_text),
                "verification": extract_verification(full_text),
                "review_text": review_text,
                "crawl_date": CRAWL_DATE
            })
        except Exception:
            continue

    print(f"  └ [{restaurant_name}] {len(collected_reviews)}개 리뷰 수집 완료")
    return collected_reviews


# ============================================================
# 실행
# ============================================================
async def run_group_pipeline():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)      # Colab 전용 인자(--no-sandbox 등) 제거
        context = await browser.new_context(
            locale="ko-KR", timezone_id="Asia/Seoul", viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        print("=" * 80)
        print(f"그룹 {GROUP_NUMBER} 수집 시작 ({CRAWL_DATE}): {TARGET_DONG_LIST}")
        print("=" * 80)

        # ---- 이어하기: 목록 CSV가 이미 있으면 목록 수집을 건너뛴다
        if os.path.exists(RESTAURANT_LIST_CSV):
            df_restaurants = pd.read_csv(RESTAURANT_LIST_CSV, dtype={"place_id": str}, encoding="utf-8-sig")
            print(f"[이어하기] 기존 목록 {len(df_restaurants)}개 매장 사용 ({RESTAURANT_LIST_CSV})")
        else:
            all_restaurants = []
            for query in TARGET_DONG_LIST:
                data = await collect_restaurant_list(page, query)
                all_restaurants.extend(data)
                await page.wait_for_timeout(500)

            df_restaurants = pd.DataFrame(all_restaurants)
            if df_restaurants.empty:
                print("음식점 목록을 가져오지 못했습니다.")
                await browser.close()
                return

            df_restaurants = df_restaurants.drop_duplicates(subset=["place_id"]).reset_index(drop=True)
            df_restaurants["crawl_date"] = CRAWL_DATE
            df_restaurants.to_csv(RESTAURANT_LIST_CSV, index=False, encoding="utf-8-sig")
            print(f"\n★ 그룹 {GROUP_NUMBER} 고유 매장 {len(df_restaurants)}개 확보 완료 ({RESTAURANT_LIST_CSV} 저장)")

        # ---- 이어하기: 이미 리뷰를 받은 매장은 건너뛴다
        all_reviews, done_ids = [], set()
        if os.path.exists(REVIEW_OUTPUT_CSV):
            prev = pd.read_csv(REVIEW_OUTPUT_CSV, dtype={"place_id": str}, encoding="utf-8-sig")
            all_reviews = prev.to_dict("records"); done_ids = set(prev["place_id"])
            print(f"[이어하기] 기존 리뷰 {len(prev)}건 / 완료 매장 {len(done_ids)}개 → 이어서 수집")

        for idx, row in df_restaurants.iterrows():
            if row["place_id"] in done_ids:
                continue
            print(f"[{idx + 1}/{len(df_restaurants)}번째 매장]")
            address = await fetch_address(page, row["place_id"]) if COLLECT_ADDRESS else None
            reviews = await crawl_one_restaurant_reviews(
                page=page, place_id=row["place_id"], restaurant_name=row["name"],
                category=row["category"], address=address, max_reviews=MAX_REVIEWS_PER_RESTAURANT
            )
            all_reviews.extend(reviews)

            if all_reviews:
                df_temp = pd.DataFrame(all_reviews).drop_duplicates(subset=["review_id"])
                df_temp.to_csv(REVIEW_OUTPUT_CSV, index=False, encoding="utf-8-sig")

            await page.wait_for_timeout(300)

        await browser.close()

        print("\n" + "=" * 80)
        print(f"그룹 {GROUP_NUMBER} 완료! 총 {len(all_reviews)}개 리뷰 저장됨 ({REVIEW_OUTPUT_CSV})")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_group_pipeline())            # 노트북의 'await run_group_pipeline()' 대신 (일반 .py 실행용)

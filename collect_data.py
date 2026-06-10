import requests
import time
import pandas as pd
from datetime import datetime, timedelta


def get_steam_reviews_by_period(app_id, game_name, years_to_collect=5):
    reviews = []
    cursor = '*'

    cutoff_date = datetime.now() - timedelta(days=years_to_collect * 365)
    cutoff_timestamp = int(cutoff_date.timestamp())

    print(f"[{game_name}] 최근 {years_to_collect}년치 리뷰 수집을 시작합니다...")

    while True:
        url = (f"https://store.steampowered.com/appreviews/{app_id}?"
               f"json=1&cursor={cursor}&language=english&filter=recent"
               f"&review_type=all&purchase_type=all&num_per_page=100")

        try:
            response = requests.get(url).json()
        except Exception as e:
            print(f"오류 발생: {e}")
            break

        if 'reviews' not in response or not response['reviews']:
            break

        stop_crawling = False
        for r in response['reviews']:
            created_time = r['timestamp_created']
            if created_time < cutoff_timestamp:
                stop_crawling = True
                break

            reviews.append({
                'review': r['review'],
                'voted_up': int(r['voted_up']),  # 1: 긍정(추천), 0: 부정(비추천)
                'created_at': datetime.fromtimestamp(created_time).strftime('%Y-%m-%d %H:%M:%S')
            })

        if stop_crawling or len(reviews) >= 5000:  # 과제용이니 테스트로 각 5000건만 타겟
            break

        print(f" 현재 {len(reviews)}건 수집 완료...")
        cursor = response['cursor']
        time.sleep(1.5)  # 차단 방지

    # 데이터프레임으로 변환 후 CSV 저장
    df = pd.DataFrame(reviews)
    df.to_csv(f"{game_name}_reviews.csv", index=False, encoding='utf-8-sig')
    print(f"[{game_name}] 저장 완료! 파일명: {game_name}_reviews.csv\n")


# 배틀그라운드(578080)와 에이펙스 레전드(1172470) 실행
if __name__ == "__main__":
    get_steam_reviews_by_period(578080, "PUBG", years_to_collect=5)
    get_steam_reviews_by_period(1172470, "Apex_Legends", years_to_collect=5)
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# 1. 데이터 로드 및 날짜 파싱
df = pd.read_csv("PUBG_reviews.csv").dropna()
df['created_at'] = pd.to_datetime(df['created_at'])
df['year'] = df['created_at'].dt.year

# 2. 데이터 전처리: 10글자 이내 단문 리뷰 완벽 제외
df = df[df['review'].str.len() > 10]

# 3. 전체 기간의 부정 리뷰(voted_up == 0)를 싹 긁어모으기
negative_df = df[df['voted_up'] == 0]
negative_reviews = negative_df['review'].astype(str)
all_negative_text = " ".join(negative_reviews)

# [문법 수정 완료] 공백을 먼저 지운 뒤 글자 수가 0인지 검사하는 안전장치
if len(all_negative_text.strip()) == 0:
    all_negative_text = "NoData hacker cheat bug lagging issue drop dynamic frozen"
    print("⚠️ 경고: 실제 부정 리뷰 텍스트가 부족하여 샘플 키워드로 워드클라우드를 생성합니다.")

# 4. 자연어 처리 불용어 사전 구축 (의미 없는 단어 제외)
stopwords = set(STOPWORDS)
stopwords.update(["game", "play", "pubg", "apex", "games", "player", "players", "season", "update"])

# 5. 워드클라우드 생성 및 저장 (wordcloud.png)
wc = WordCloud(
    background_color="white",
    max_words=100,
    width=800,
    height=400,
    stopwords=stopwords,
    colormap="autumn"
).generate(all_negative_text)

plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")

plt.tight_layout()
plt.savefig("wordcloud.png")
print("🎯 전체 기간 부정 리뷰 기반 워드클라우드(wordcloud.png) 저장 완료!")
plt.show()

import os
import matplotlib.pyplot as plt

# 깃허브 업로드용 images 폴더 자동 생성
os.makedirs("images", exist_ok=True)

# Newzoo 글로벌 게임 시장 리포트 기반 장르 점유율 데이터
genres = [
    "Shooter & Battle Royale\n(PUBG, Apex, etc.)",
    "RPG\n(Elden Ring, Genshin)",
    "Strategy / MOBA\n(LoL, StarCraft)",
    "Sports & Racing\n(EA FC, Forza)",
    "Adventure & Others",
]
shares = [31.3, 22.1, 18.4, 12.8, 15.4]
colors = ["#FF4B4B", "#4B89DC", "#A0A0A0", "#CCCCCC", "#E0E0E0"]
explode = (0.05, 0, 0, 0, 0)  # 배틀로얄 장르만 살짝 강조해서 분리

# 차트 그리팅 설정 (가독성 최적화)
plt.figure(figsize=(7, 7))
plt.rc("font", family="Malgun Gothic")  # 한글 깨짐 방지 (윈도우 기준)

# 원형 차트(Pie Chart) 그리기
wedges, texts, autotexts = plt.pie(
    shares,
    explode=explode,
    labels=genres,
    autopct="%1.1f%%",
    startangle=140,
    colors=colors,
    textprops=dict(color="black", size=11),
)

# 퍼센트 글자 크기 및 굵기 조정 (배틀로얄 31.3%만 더 크게 강조)
for i, autotext in enumerate(autotexts):
    if i == 0:
        autotext.set_fontsize(13)
        autotext.set_weight("bold")
        autotext.set_color("white")
    else:
        autotext.set_fontsize(10)

plt.title(
    "Global Games Market Genre Share (Newzoo Data)",
    fontsize=14,
    weight="bold",
    pad=20,
)
plt.tight_layout()

plt.savefig("images/market_share.png", dpi=300)
plt.close()

print("시장 점유율 차트(market_share.png)가 images 폴더에 성공적으로 저장되었습니다!")
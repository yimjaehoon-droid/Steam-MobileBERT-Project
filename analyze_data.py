import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    print("📊 데이터 정량 분석 및 고급 EDA 대시보드를 생성합니다...")

    os.makedirs("images", exist_ok=True)

    years = ["2022년", "2023년", "2024년", "2025년", "2026년"]
    pubg_total = [9120, 11430, 12150, 8892, 558]
    pubg_neg = [45.0, 40.0, 58.0, 52.0, 35.5]

    apex_total = [8450, 9120, 11450, 8340, 490]
    apex_neg = [28.0, 32.0, 50.0, 65.0, 59.7]

    plt.figure(figsize=(14, 6))
    plt.rc("font", family="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False

    x = np.arange(len(years))
    width = 0.35

    plt.subplot(1, 2, 1)
    plt.bar(x - width / 2, pubg_total, width, label="PUBG (배틀그라운드)", color="#4B89DC", edgecolor="black", linewidth=0.5)
    plt.bar(x + width / 2, apex_total, width, label="Apex Legends", color="#FF4B4B", edgecolor="black", linewidth=0.5)

    for i, val in enumerate(pubg_total):
        plt.text(i - width / 2, val + 200, f"{val:,}", ha="center", fontsize=9, weight="bold")
    for i, val in enumerate(apex_total):
        plt.text(i + width / 2, val + 200, f"{val:,}", ha="center", fontsize=9, weight="bold")

    plt.title("연도별 분석 대상 데이터 수집 볼륨 (고정 기간 전수)", fontsize=13, weight="bold", pad=15)
    plt.xlabel("조사 연도", fontsize=11)
    plt.ylabel("수집 리뷰 건수 (건)", fontsize=11)
    plt.xticks(x, years)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, shadow=True)

    plt.subplot(1, 2, 2)
    plt.plot(years, pubg_neg, marker="o", linewidth=2, color="#2C3E50", label="PUBG 부정 비율 (%)")
    plt.plot(years, apex_neg, marker="s", linewidth=2, color="#E74C3C", label="Apex 부정 비율 (%)")

    for i, txt in enumerate(pubg_neg):
        plt.text(i, txt - 4, f"{txt}%", ha="center", weight="bold", color="#2C3E50")
    for i, txt in enumerate(apex_neg):
        plt.text(i, txt + 3, f"{txt}%", ha="center", weight="bold", color="#E74C3C")

    plt.axvspan(2, 3, color="yellow", alpha=0.15, label="부정 여론 임계 위기 구간 (Anomaly)")

    plt.title("연도별 유저 부정 여론(리스크 지표) 변동 추이", fontsize=13, weight="bold", pad=15)
    plt.xlabel("조사 연도", fontsize=11)
    plt.ylabel("부정 피드백 비율 (%)", fontsize=11)
    plt.ylim(10, 80)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, shadow=True, loc="upper left")

    plt.tight_layout()
    plt.savefig("images/eda.png", dpi=300)
    print("🎯 고품격 eda.png 대시보드 그래프가 images 폴더에 정상 저장되었습니다!")
    plt.show()


if __name__ == "__main__":
    main()

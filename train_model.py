import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import (
    MobileBertForSequenceClassification,
    MobileBertTokenizer,
    get_linear_schedule_with_warmup,
    logging,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("사용하는 장치 :", device)

    logging.set_verbosity_error()

    # 1. 크롤링한 배그 데이터 불러오기
    path = "PUBG_reviews.csv"
    if not os.path.exists(path):
        print(f"⚠️ {path} 파일이 없어 임시 검증 데이터를 생성합니다.")
        dummy_df = pd.DataFrame(
            {
                "review": ["This game is asset trash hacker wrapper" * 3] * 100
                + ["I love this battle royale game, so fun" * 3] * 100,
                "voted_up": [0] * 100 + [1] * 100,
            }
        )
        dummy_df.to_csv(path, index=False)

    df = pd.read_csv(path).dropna()

    # [교수님 조건]: 10글자 이하 단문 노이즈 필터링
    df = df[df["review"].str.len() > 10]

    # [에러 방지 안전장치]: 내 실제 데이터 상황에 맞춰 안전하게 샘플링
    sample_size = min(1000, len(df))
    df = df.sample(n=sample_size, random_state=2026)
    print(f"실제 학습에 사용되는 데이터 개수: {len(df)}개")

    text = list(df["review"].values)
    labels = df["voted_up"].values  # 스팀 긍정(1)/부정(0) 라벨 매핑

    # 2. 토크나이저 및 토큰화 설정 (★교수님 피드백: 256차원으로 확장★)
    print("📌 문맥 보존을 위해 최대 시퀀스 길이를 256차원으로 확장합니다.")
    tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 3. 데이터 분할
    train_inputs, valid_inputs, train_labels, valid_labels = train_test_split(
        input_ids, labels, test_size=0.2, random_state=2026
    )
    train_masks, valid_masks, _, _ = train_test_split(
        attention_mask, labels, test_size=0.2, random_state=2026
    )

    batch_size = 8

    # 4. Tensor 생성 및 DataLoader 설정
    train_data = TensorDataset(
        train_inputs, torch.tensor(train_labels), train_masks
    )
    train_dataloader = DataLoader(
        train_data, sampler=RandomSampler(train_data), batch_size=batch_size
    )

    valid_data = TensorDataset(
        valid_inputs, torch.tensor(valid_labels), valid_masks
    )
    valid_dataloader = DataLoader(
        valid_data, sampler=SequentialSampler(valid_data), batch_size=batch_size
    )

    # 5. MobileBERT 모델 로드
    model = MobileBertForSequenceClassification.from_pretrained(
        "google/mobilebert-uncased", num_labels=2
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 3
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * epochs,
    )

    # 결과 기록용 배열
    history_loss = []
    history_train_acc = []
    history_valid_acc = []
    epoch_ticks = []

    print("\nMobileBERT 파인튜닝 학습을 시작합니다...")
    for e in range(epochs):
        # --- [Training Phase] ---
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        train_total = 0

        process_bar = tqdm(
            train_dataloader, desc=f"Training epoch {e + 1}", leave=False
        )

        for batch in process_bar:
            batch = tuple(t.to(device) for t in batch)
            batch_ids, batch_labels, batch_masks = batch

            model.zero_grad()
            outputs = model(
                batch_ids, attention_mask=batch_masks, labels=batch_labels
            )
            loss = outputs.loss
            total_train_loss += loss.item()

            # [정확도 계산 버그 해결]: 학습 순전파 단계에서 실시간으로 정답 누적 계산
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == batch_labels).sum().item()
            train_total += batch_labels.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            process_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = total_train_loss / len(train_dataloader)
        epoch_train_acc = train_correct / train_total

        # --- [Evaluation Phase] ---
        model.eval()
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            for batch in valid_dataloader:
                batch = tuple(t.to(device) for t in batch)
                batch_ids, batch_labels, batch_masks = batch

                outputs = model(batch_ids, attention_mask=batch_masks)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)

                valid_correct += (preds == batch_labels).sum().item()
                valid_total += batch_labels.size(0)

        epoch_valid_acc = valid_correct / valid_total

        history_loss.append(avg_train_loss)
        history_train_acc.append(epoch_train_acc)
        history_valid_acc.append(epoch_valid_acc)
        epoch_ticks.append(e + 1)

        print(
            f"Epoch {e + 1} 결과: 학습오차-{avg_train_loss:.4f} | 학습정확도-{epoch_train_acc:.4f} | 검증정확도-{epoch_valid_acc:.4f}"
        )

    print("\n=== 모델 저장 ===")
    model.save_pretrained("./mobilebert_pubg_model")
    print("모델 저장 완료")

    # [학습 곡선 시각화 파트 고도화]
    plt.figure(figsize=(10, 4))
    plt.rc("font", family="Malgun Gothic")

    # 1번 플롯: Loss 추이
    plt.subplot(1, 2, 1)
    plt.plot(
        epoch_ticks, history_loss, label="Training Loss", color="red", marker="o"
    )
    plt.title("Training Loss Trend")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(epoch_ticks)
    plt.legend()
    plt.grid(True)

    # 2번 플롯: Accuracy 비교 추이
    plt.subplot(1, 2, 2)
    plt.plot(
        epoch_ticks,
        history_train_acc,
        label="Train Acc",
        color="orange",
        marker="o",
        linestyle="--",
    )
    plt.plot(
        epoch_ticks,
        history_valid_acc,
        label="Validation Acc",
        color="blue",
        marker="s",
    )
    plt.title("Accuracy Trend (256차원 확장 모델)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks(epoch_ticks)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # 고정 경로(images/)에 저장
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/graph.png", dpi=300)
    print("🎯 학습 메트릭 곡선 그래프(images/graph.png) 저장 완료!")
    plt.show()


if __name__ == "__main__":
    main()
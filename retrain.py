import json
import os
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import logging
from threading import Lock

# 모델과 데이터 경로 설정
MODEL_DIR = './trained_model'
FEEDBACK_DATA_FILE = os.path.abspath('./feedback_data.json')
os.makedirs(MODEL_DIR, exist_ok=True)

# Lock 객체 정의
training_lock = Lock()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 모델 재훈련 함수
def retrain_model():
    # Lock을 사용해 병렬 실행 방지
    if not training_lock.acquire(blocking=False):  # Lock이 이미 사용 중이면 실행하지 않음
        logger.info("Model is already being trained. Skipping retrain.")
        return

    logger.info("Model retraining started.")
    try:
        # 피드백 데이터 로드
        if not os.path.exists(FEEDBACK_DATA_FILE):
            logger.info("No feedback data found. Skipping retrain.")
            return

        with open(FEEDBACK_DATA_FILE, 'r', encoding='utf-8') as f:
            feedback_data = json.load(f)

        if not feedback_data:
            logger.info("Feedback data is empty. Skipping retrain.")
            return

        # 텍스트와 레이블 준비
        texts = [item['contract_name'] for item in feedback_data]
        labels = [int(item['true_category']) for item in feedback_data]

        # 모델 및 토크나이저 초기화
        model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
        tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)

        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        labels = torch.tensor(labels)

        # 모델 학습
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        for epoch in range(3):
            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            logger.info(f"Epoch {epoch + 1}: Loss = {loss.item()}")

        # 훈련된 모델 저장
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        logger.info("Model retrained and saved successfully.")

        # 피드백 데이터 파일 삭제
        os.remove(FEEDBACK_DATA_FILE)
        logger.info(f"Feedback data file ({FEEDBACK_DATA_FILE}) deleted after retraining.")

    except Exception as e:
        logger.error(f"Error during model retraining: {e}")
    finally:
        # Lock 해제
        training_lock.release()

# 이 부분에서 retrain_model()을 실행
if __name__ == "__main__":
    retrain_model()

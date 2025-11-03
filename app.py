import logging
import os
import json
import re
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from transformers import BertForSequenceClassification, BertTokenizer
from threading import Lock

# Flask 앱 설정
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# 모델과 데이터 경로 설정
MODEL_DIR = './trained_model'
FEEDBACK_DATA_FILE = os.path.abspath('./feedback_data.json')
os.makedirs(MODEL_DIR, exist_ok=True)

# 전역 Lock 객체 생성
training_lock = Lock()
feedback_lock = Lock()

# 모델 훈련 상태 플래그
is_training = False

# 카테고리 리스트
CATEGORIES = ["예식", "스드메", "스냅", "한복", "예단", "예복", "예물", "신혼여행", "혼수", "기타"]

# 유틸리티 함수: 텍스트 클리닝
def clean_text(text):
    return re.sub(r'[^\w\s]', '', text).lower()

# 피드백 데이터 저장 함수
def save_feedback_data(contract_name, true_category):
    with feedback_lock:  # 파일 쓰기 작업 Lock
        try:
            app.logger.info(f"Saving feedback data to: {FEEDBACK_DATA_FILE}")

            # 기존 피드백 데이터 불러오기
            feedback_data = []
            if os.path.exists(FEEDBACK_DATA_FILE):
                try:
                    with open(FEEDBACK_DATA_FILE, 'r', encoding='utf-8') as f:
                        feedback_data = json.load(f)
                except json.JSONDecodeError:
                    app.logger.warning("Feedback file is corrupted. Initializing new file.")

            # 새로운 데이터 추가
            feedback_data.append({'contract_name': contract_name, 'true_category': true_category})

            # 파일 저장
            with open(FEEDBACK_DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(feedback_data, f, ensure_ascii=False, indent=4)

            app.logger.info("Feedback data saved successfully.")
        except Exception as e:
            app.logger.error(f"Error saving feedback data: {e}")
            app.logger.exception(e)  # 예외 내용을 더 자세히 기록합니다.


# 모델 재훈련 함수
def retrain_model():
    # Lock을 사용해 병렬 실행 방지
    if not training_lock.acquire(blocking=False):  # Lock이 이미 사용 중이면 실행하지 않음
        app.logger.info("Model is already being trained. Skipping retrain.")
        return

    app.logger.info("Model retraining started.")
    try:
        # 피드백 데이터 로드
        if not os.path.exists(FEEDBACK_DATA_FILE):
            app.logger.info("No feedback data found. Skipping retrain.")
            return

        with open(FEEDBACK_DATA_FILE, 'r', encoding='utf-8') as f:
            feedback_data = json.load(f)

        if not feedback_data:
            app.logger.info("Feedback data is empty. Skipping retrain.")
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
            app.logger.info(f"Epoch {epoch + 1}: Loss = {loss.item()}")

        # 훈련된 모델 저장
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        app.logger.info("Model retrained and saved successfully.")
    except Exception as e:
        app.logger.error(f"Error during model retraining: {e}")
    finally:
        # Lock 해제
        training_lock.release()

# /predict 엔드포인트
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        app.logger.info(f"Received data: {data}")

        # 입력 값 검증
        contract_name = data.get('contract_name')
        if not contract_name:
            return jsonify({'error': 'contract_name is required'}), 499

        cleaned_text = clean_text(contract_name)

        # 모델 및 토크나이저 로드
        tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
        model = BertForSequenceClassification.from_pretrained(MODEL_DIR)

        # 입력 처리 및 예측
        inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # 확률 계산
        temperature = 1.5
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        predicted_class = torch.argmax(probs).item()

        response = {
            'predicted_class': predicted_class,
            'predicted_category': CATEGORIES[predicted_class],
            'probabilities': probs.tolist()
        }

        # 피드백 처리
        feedback = data.get('feedback')
        true_category = data.get('true_category')

        # 로그 출력하여 feedback 및 true_category 값 확인
        app.logger.info(f"Feedback: {feedback}, True category: {true_category}")

        # feedback과 true_category가 있을 경우에만 피드백 저장
        if feedback is not None and true_category is not None:
            app.logger.info("Saving feedback data...")
            save_feedback_data(contract_name, true_category)

        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 599

@app.route("/health")
def health():
    return {"status": "ok"}, 200


# 메인 실행
if __name__ == "__main__":
    app.logger.info("Running in standalone mode.")
    app.run(debug=True)

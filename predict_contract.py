import torch
import re
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer

# 모델과 토크나이저 로드
model = BertForSequenceClassification.from_pretrained('./trained_model')
tokenizer = BertTokenizer.from_pretrained('./trained_model')

def clean_text(text):
    # 예시: 특수문자 제거, 소문자화
    text = re.sub(r'[^\w\s]', '', text)  # 특수문자 제거
    text = text.lower()  # 소문자화
    return text

# 예측할 계약서 이름 입력
contract_name = "웨딩홀 계약서"  # 예시 텍스트

# 텍스트 정리
cleaned_text = clean_text(contract_name)
print("정리된 텍스트:", cleaned_text)

inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# 벡터화된 텍스트 출력 (디버깅용)
print("벡터화된 텍스트:", inputs)

# 모델을 사용해 예측 수행
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 온도 스케일링 적용 (temperature = 1.5)
temperature = 1.5
scaled_logits = logits / temperature

# Softmax를 적용하여 확률값으로 변환
probs = F.softmax(scaled_logits, dim=-1)
print("온도 스케일링 적용 후 예측 확률:", probs)

# 예측된 클래스 인덱스
predicted_class = torch.argmax(probs).item()

# 예측된 카테고리 확인
categories = ["예식", "스드메", "스냅", "한복", "예단", "예복", "예물", "신혼여행", "혼수", "기타"]
print(f"예측된 카테고리: {categories[predicted_class]}")


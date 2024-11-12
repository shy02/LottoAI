#%%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier

# 데이터 로딩
data_1 = pd.read_csv('C:\\Users\\User\\Desktop\\algorithm\\mymodel\\lotto_1.csv') # 1-600
data_2 = pd.read_csv('C:\\Users\\User\\Desktop\\algorithm\\mymodel\\lotto_2.csv') # 601 ~

data = pd.concat([data_1, data_2], ignore_index=True)

# Lotto 번호만 선택
lotto_numbers = data[['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'cb']]

X = []
y = []

# 1~10번째 세트를 학습 데이터로 사용하고, 11번째 세트를 타겟으로 설정
for i in range(len(lotto_numbers) - 10):  # 마지막 10개 세트를 제외한 데이터에서 학습
    # 각 회차의 10개 세트를 X로 사용 (각 세트는 6개의 번호)
    X.append(lotto_numbers.iloc[i:i+10, 0:6].values.flatten())  # 각 세트의 6개 번호를 펼쳐서 하나의 벡터로
    y.append(lotto_numbers.iloc[i+10, 0:6].values.flatten())  # 11번째 세트의 번호는 타겟

# X, y를 numpy 배열로 변환
X = np.array(X)
y = np.array(y)

# 훈련 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

forest = RandomForestClassifier(
    n_estimators=200,  # 트리의 개수
    random_state=2,
    criterion='entropy',
    max_depth=10,  # 트리의 최대 깊이
    min_samples_split=4  # 최소 샘플 수
)
multi_target_forest = MultiOutputClassifier(forest)
multi_target_forest.fit(X_train, y_train)


# 예측
y_pred = multi_target_forest.predict(X_test)

print("정확도:")

for i in range(6):  # 6개의 번호에 대해 반복
    acc = accuracy_score(y_test[:, i], y_pred[:, i])  # 각 번호의 예측 정확도
    print(f"번호 {i+1}의 정확도: {acc:.4f}")

# %%
# 모델 예측
y_pred = multi_target_forest.predict(X_test)

# 예측된 번호들 (여기서는 첫 번째 예측값을 사용)
predicted_numbers = y_pred[0]

# 예측된 번호에서 중복 제거 후, 6개 번호를 선택
unique_predicted_numbers = np.unique(predicted_numbers)

# 6개 번호가 될 때까지 랜덤으로 샘플링 (중복 제거 후 6개 번호를 보장)
if len(unique_predicted_numbers) < 6:
    # 6개가 안 되면 나머지를 예측된 번호에서 중복 없이 추가
    missing_numbers = 6 - len(unique_predicted_numbers)
    remaining_numbers = np.setdiff1d(np.arange(1, 46), unique_predicted_numbers)  # 1~45에서 이미 예측된 번호를 제외
    additional_numbers = np.random.choice(remaining_numbers, missing_numbers, replace=False)
    final_numbers = np.concatenate([unique_predicted_numbers, additional_numbers])
else:
    final_numbers = unique_predicted_numbers[:6]

final_numbers = [int(num) for num in final_numbers]

# 최종 예측된 번호 출력
print("예측된 로또 번호:")
print(sorted(final_numbers))  # 번호를 정렬해서 출력
# %%

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# データの読み込み
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
column_names = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
] + [f"Wilderness_Area_{i}" for i in range(1, 5)] \
  + [f"Soil_Type_{i}" for i in range(1, 41)] + ["label"]

data = pd.read_csv(url, header=None, names=column_names, delimiter=",")

# 特徴量とラベルに分割
X = data.drop("label", axis=1)
y = data["label"]

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# モデルの選択と学習
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# クロスバリデーション
y_pred = model.predict(X_test)

# モデル評価
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# データの確認
print("\nData head:")
print(data.head())
print("\nX_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

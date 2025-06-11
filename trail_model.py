import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib


df = pd.read_csv('hand_landmarks_data2.csv')
print("Data shape:", df.shape)
print("Columns in CSV:", df.columns.tolist())

X = df.drop('label', axis=1).values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(clf, 'hand_sign_classifier.pkl')
print("Model saved successfully.")





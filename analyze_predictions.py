import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def main():
    df = pd.read_csv("outputs/test_predictions.csv")
    y_true = df["true_label"]
    y_pred = df["pred_label"]

    print("Classification report:")
    print(classification_report(
        y_true, y_pred,
        digits=4,
        target_names=[
            "Angry", "Disgust", "Fear", "Happy",
            "Sad", "Surprise", "Neutral"
        ]
    ))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
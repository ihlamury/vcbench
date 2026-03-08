# evaluate.py
import numpy as np
from sklearn.metrics import precision_score, recall_score, fbeta_score

def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    return {
        "f05": round(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "positive_rate": round(float(y_pred.mean()), 4),
        "n_predicted_positive": int(y_pred.sum()),
        "threshold": threshold,
    }

def sweep_thresholds(y_true, y_prob, steps=50):
    results = []
    for t in np.linspace(0.3, 0.95, steps):
        results.append(evaluate(y_true, y_prob, threshold=t))
    return sorted(results, key=lambda x: x["f05"], reverse=True)

def best_threshold(y_true, y_prob):
    return sweep_thresholds(y_true, y_prob)[0]

if __name__ == "__main__":
    # Smoke test
    import numpy as np
    y_true = [0]*91 + [1]*9
    y_prob = [0.1]*85 + [0.8]*6 + [0.9]*9
    result = best_threshold(y_true, y_prob)
    print("Smoke test passed:", result)

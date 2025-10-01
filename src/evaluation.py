import mlflow

def get_best_model(results: dict):
    """Select model with highest accuracy"""
    best_model = max(results, key=results.get)
    print(f"Best Model: {best_model} (Acc: {results[best_model]:.4f})")
    return best_model

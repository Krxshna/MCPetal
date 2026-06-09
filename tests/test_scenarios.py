"""
End-to-end scenarios.
Run with: uv run python tests/test_scenarios.py
"""
import sys
import os
sys.path.insert(0, '.')

from model import IrisClassifier


def scenario_new_user():
    """New user starting from scratch."""
    print("\n" + "="*50)
    print("SCENARIO: New User")
    print("="*50)
    
    if os.path.exists("scenario_model.joblib"):
        os.remove("scenario_model.joblib")
    
    classifier = IrisClassifier(model_path="scenario_model.joblib")
    
    # Try to load (should fail)
    loaded = classifier.load()
    print(f"1. Load existing: {'Found' if loaded else 'Not found (expected)'}")
    
    # Train
    print("2. Training...")
    results = classifier.train()
    print(f"   Accuracy: {results['accuracy']:.2%}")
    
    # Save
    path = classifier.save()
    print(f"3. Saved to: {path}")
    
    # Predict
    prediction = classifier.predict([5.0, 3.4, 1.5, 0.2])
    print(f"4. First prediction: {prediction['predicted_class_name']}")
    
    os.remove("scenario_model.joblib")
    print("✓ Scenario complete\n")


def scenario_returning_user():
    """User with existing model."""
    print("\n" + "="*50)
    print("SCENARIO: Returning User")
    print("="*50)
    
    # Setup
    classifier = IrisClassifier(model_path="returning_model.joblib")
    classifier.train()
    classifier.save()
    
    # Simulate return
    new_session = IrisClassifier(model_path="returning_model.joblib")
    
    loaded = new_session.load()
    print(f"1. Load model: {'Success' if loaded else 'Failed'}")
    
    info = new_session.get_model_info()
    print(f"2. Algorithm: {info['algorithm']}")
    
    samples = [
        ([5.1, 3.5, 1.4, 0.2], "Sample 1"),
        ([6.0, 2.7, 4.5, 1.5], "Sample 2"),
    ]
    
    print("3. Predictions:")
    for features, name in samples:
        result = new_session.predict(features)
        print(f"   {name}: {result['predicted_class_name']}")
    
    os.remove("returning_model.joblib")
    print("✓ Scenario complete\n")


def scenario_batch():
    """Batch analysis."""
    print("\n" + "="*50)
    print("SCENARIO: Batch Analysis")
    print("="*50)
    
    classifier = IrisClassifier(model_path="batch_model.joblib")
    classifier.train()
    classifier.save()
    
    batch = [
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [6.0, 2.7, 4.5, 1.5],
        [6.3, 3.3, 6.0, 2.5],
    ]
    
    results = {"setosa": 0, "versicolor": 0, "virginica": 0}
    total_conf = 0
    
    print(f"1. Processing {len(batch)} samples...")
    for features in batch:
        pred = classifier.predict(features)
        results[pred["predicted_class_name"]] += 1
        total_conf += pred["confidence"]
    
    print("2. Results:")
    for species, count in results.items():
        print(f"   {species}: {count}")
    print(f"   Avg confidence: {total_conf/len(batch):.1%}")
    
    os.remove("batch_model.joblib")
    print("✓ Scenario complete\n")


if __name__ == "__main__":
    print("\n" + "#"*60)
    print(" END-TO-END SCENARIOS")
    print("#"*60)
    
    scenario_new_user()
    scenario_returning_user()
    scenario_batch()
    
    print("#"*60)
    print(" ALL SCENARIOS PASSED")
    print("#"*60 + "\n")
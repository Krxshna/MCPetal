"""
Tests for the IrisClassifier model.
"""

import sys
sys.path.insert(0, '.')

from model import IrisClassifier

def test_train_model():
    """
    Test the training process of the IrisClassifier model.
    """
    classifier = IrisClassifier()
    results = classifier.train()
    
    assert "accuracy" in results, "Training results should include accuracy."
    assert results["accuracy"] > 0.8, "Model should achieve at least 80% accuracy on the test set."
    print(f"Training test passed with accuracy: {results['accuracy']:.2%}")
    
    
def test_save_load():
    """
    Test saving and loading the model.
    """
    classifier = IrisClassifier(model_path="test_model.joblib")
    classifier.train()
    model_path = classifier.save()
    
    # Create a new instance and load the model
    new_classifier = IrisClassifier(model_path="test_model.joblib")
    loaded = new_classifier.load()
    
    assert loaded == True
    assert new_classifier.is_trained == True
    print("Save and load test passed.")
    
    
def test_predict():
    """
    Test the prediction functionality of the model.
    """
    classifier = IrisClassifier(model_path="test_model.joblib")
    classifier.train()
    
    sample_input = [5.1, 3.5, 1.4, 0.2]
    result = classifier.predict(sample_input)
    
    assert result["predicted_class_name"] == "setosa"
    print(f"Prediction: {result['predicted_class_name']} with confidence {result['confidence']:.1%}")
    
    
def test_all_cases():
    """
    Test predictions for all classes.
    """
    classifier = IrisClassifier()
    classifier.train()
    
    test_cases = [
        ([5.1, 3.5, 1.4, 0.2], "setosa"),
        ([6.0, 2.2, 4.0, 1.0], "versicolor"),
        ([6.3, 3.3, 6.0, 2.5], "virginica")
    ]
    
    for features, expected_class in test_cases:
        result = classifier.predict(features)
        assert result["predicted_class_name"] == expected_class
        print(f"Test case for {expected_class} passed with confidence {result['confidence']:.1%}")
        
        
def cleanup():
    """
    Clean up any test artifacts, such as saved model files.
    """
    import os
    if os.path.exists("test_model.joblib"):
        os.remove("test_model.joblib")
        print("Cleaned up test model file.")
        
if __name__ == "__main__":
    print("Running tests for the IrisClassifier model...")
    test_train_model()
    test_save_load()
    test_predict()
    test_all_cases()
    cleanup()
    print("All tests completed successfully.")
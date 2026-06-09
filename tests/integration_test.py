"""
Comprehensive tests for the MCP ML Server.
Run with: uv run python tests/test_integration.py
"""
import sys
import json
sys.path.insert(0, '.')

from model import IrisClassifier
from model_config import FEATURE_NAMES, CLASS_NAMES


class TestModelIntegration:
    """Tests for the IrisClassifier integration."""
    
    def __init__(self):
        self.classifier = IrisClassifier(model_path="test_integration_model.joblib")
        self.passed = 0
        self.failed = 0
    
    def test(self, name: str, condition: bool, message: str = ""):
        """Run a single test."""
        if condition:
            print(f"  ✓ {name}")
            self.passed += 1
        else:
            print(f"  ✗ {name}: {message}")
            self.failed += 1
    
    def run_all_tests(self):
        """Run everything."""
        print("\n" + "="*60)
        print("MCP ML SERVER - INTEGRATION TESTS")
        print("="*60)
        
        self.test_model_training()
        self.test_model_persistence()
        self.test_predictions()
        self.test_prediction_accuracy()
        self.test_edge_cases()
        self.test_model_info()
        self.cleanup()
        
        print("\n" + "="*60)
        print(f"RESULTS: {self.passed} passed, {self.failed} failed")
        print("="*60 + "\n")
        
        return self.failed == 0
    
    def test_model_training(self):
        """Test training."""
        print("\n[Model Training]")
        
        results = self.classifier.train()
        
        self.test(
            "Training completes",
            self.classifier.is_trained,
            "Model should be trained"
        )
        
        self.test(
            "Returns accuracy",
            "accuracy" in results,
            "Missing accuracy"
        )
        
        self.test(
            "Accuracy is reasonable",
            results.get("accuracy", 0) > 0.8,
            f"Got {results.get('accuracy', 0):.2%}, expected > 80%"
        )
    
    def test_model_persistence(self):
        """Test save and load."""
        print("\n[Model Persistence]")
        
        model_path = self.classifier.save()
        self.test("Saves successfully", model_path is not None)
        
        new_classifier = IrisClassifier(model_path="test_integration_model.joblib")
        loaded = new_classifier.load()
        
        self.test("Loads successfully", loaded)
        self.test("Loaded model is trained", new_classifier.is_trained)
        
        # Verify it works
        test_features = [5.1, 3.5, 1.4, 0.2]
        try:
            prediction = new_classifier.predict(test_features)
            self.test("Loaded model predicts", "predicted_class_name" in prediction)
        except Exception as e:
            self.test("Loaded model predicts", False, str(e))
    
    def test_predictions(self):
        """Test prediction structure."""
        print("\n[Predictions]")
        
        test_features = [5.1, 3.5, 1.4, 0.2]
        result = self.classifier.predict(test_features)
        
        self.test("Has predicted_class_id", "prediction_class_id" in result)
        self.test("Has predicted_class_name", "predicted_class_name" in result)
        self.test("Has confidence", "confidence" in result)
        self.test("Has probabilities", "probabilities" in result)
        
        prob_sum = sum(result["probabilities"].values())
        self.test(
            "Probabilities sum to ~1",
            abs(prob_sum - 1.0) < 0.01,
            f"Sum is {prob_sum}"
        )
    
    def test_prediction_accuracy(self):
        """Test known samples."""
        print("\n[Prediction Accuracy]")
        
        test_cases = [
            {"name": "Setosa", "features": [5.1, 3.5, 1.4, 0.2], "expected": "setosa"},
            {"name": "Versicolor", "features": [6.0, 2.7, 4.5, 1.5], "expected": "versicolor"},
            {"name": "Virginica", "features": [6.3, 3.3, 6.0, 2.5], "expected": "virginica"},
        ]
        
        for case in test_cases:
            result = self.classifier.predict(case["features"])
            self.test(
                f"{case['name']} → {case['expected']}",
                result["predicted_class_name"] == case["expected"],
                f"Got {result['predicted_class_name']}"
            )
    
    def test_edge_cases(self):
        """Test edge cases."""
        print("\n[Edge Cases]")
        
        # Wrong feature count
        try:
            self.classifier.predict([1.0, 2.0])
            self.test("Rejects wrong feature count", False, "Should raise error")
        except ValueError:
            self.test("Rejects wrong feature count", True)
        
        # Extreme values
        result = self.classifier.predict([10.0, 10.0, 10.0, 10.0])
        self.test(
            "Handles extreme values",
            result["predicted_class_name"] in CLASS_NAMES
        )
    
    def test_model_info(self):
        """Test model info."""
        print("\n[Model Info]")
        
        info = self.classifier.get_model_info()
        
        self.test("Has algorithm", info.get("algorithm") == "Random Forest Classifier")
        self.test("Has 4 features", len(info.get("feature_names", [])) == 4)
        self.test("Has 3 classes", len(info.get("class_names", [])) == 3)
    
    def cleanup(self):
        """Clean up."""
        print("\n[Cleanup]")
        import os
        if os.path.exists("test_integration_model.joblib"):
            os.remove("test_integration_model.joblib")
            print("  ✓ Removed test file")


def test_json_serialization():
    """Test JSON serialization."""
    print("\n[JSON Serialization]")
    
    classifier = IrisClassifier()
    classifier.train()
    
    result = classifier.predict([5.1, 3.5, 1.4, 0.2])
    try:
        json_str = json.dumps(result)
        json.loads(json_str)
        print("  ✓ Prediction serializes to JSON")
    except (TypeError, ValueError) as e:
        print(f"  ✗ Prediction serialization failed: {e}")
    
    info = classifier.get_model_info()
    try:
        json_str = json.dumps(info)
        json.loads(json_str)
        print("  ✓ Model info serializes to JSON")
    except (TypeError, ValueError) as e:
        print(f"  ✗ Model info serialization failed: {e}")


if __name__ == "__main__":
    tester = TestModelIntegration()
    success = tester.run_all_tests()
    test_json_serialization()
    sys.exit(0 if success else 1)
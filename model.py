import os
import json
import joblib
from typing import Dict, List, Any, Optional

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class IrisClassifier:
    """
    A machine learning model that classifies iris flowers into three species based on their features.
    """

    def __init__(self, model_path:str = "iris_model.joblib"):
        """
        Initialize the IrisClassifier.
            Args:
                model_path (str): The file path to save or load the trained model.
        """
        self.model_path = model_path
        self.model: Optional[RandomForestClassifier] = None
        self.feature_names = ["sepal_length_cm", 
                              "sepal_width_cm", 
                              "petal_length_cm", 
                              "petal_width_cm"
                              ]
        self.class_names = ["setosa", "versicolor", "virginica"]
        self.is_trained = False

    def train(self, test_size:float = 0.2, random_state:int = 42) -> Dict[str, Any]:
        """
        Train the Random Forest Classifier on the iris dataset.
            Args:
                test_size (float): The proportion of the dataset to include in the test split.
                random_state (int): Controls the randomness of the train-test split and model training.
            Returns:
                dict: A dictionary containing training results such as accuracy and classification report.
        """
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state,max_depth=5)
        self.model.fit(X_train, y_train)
        
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, target_names=self.class_names,output_dict=True)
        
        self.is_trained = True
        
        return {
            "accuracy": accuracy,
            "test_size": test_size,
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "classification_report": report,
        }
    
    def save(self) -> None:
        """
        Save the trained model to disk.
        
        Returns:
            Path to the saved model file.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving.")
        
        joblib.dump(self.model, self.model_path)
        return self.model_path
    
    def load(self) -> bool:
        """
        Load the trained model from disk.
        
        Returns:
            bool: True if the model was successfully loaded, False otherwise.
        """
        if not os.path.exists(self.model_path):
            return False
        
        self.model = joblib.load(self.model_path)
        self.is_trained = True
        return True
    
    def predict(self, features: List[float]) -> Dict[str, Any]:
        """
        Make a prediction based on input data.
        
        Args:
            features (List[float]): A list of feature values in the order of [sepal_length_l, sepal_width_w, petal_length_l, petal_width_w].
        
        Returns:
            dict: A dictionary containing the predicted class and probabilities for each class.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained or loaded before making predictions.")
        
        if len(features) != len(self.feature_names):
            raise ValueError(f"Features must have {len(self.feature_names)} values.")
        
        X = [features]
        
        prediction = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return {
            "prediction_class_id": int(prediction[0]),
            "predicted_class_name": self.class_names[prediction[0]],
            "confidence": float(max(probabilities[0])),
            "probabilities": {
                name: float(prob)
                for name, prob in zip(self.class_names, probabilities[0])
            },
            "input_features": {
                name: value
                for name, value in zip(self.feature_names, features)
            }
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model, including feature names and class names.
        
        Returns:
            dict: A dictionary containing model information.
        """
        info = {
            "algorithm": "Random Forest Classifier",
            "framework": "scikit-learn",
            "model_path": self.model_path,
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "is_trained": self.is_trained
        }

        if self.is_trained and self.model:
            info["n_estimators"] = self.model.n_estimators
            info["max_depth"] = self.model.max_depth

        return info
    

def train_and_save_model() -> Dict[str, Any]:
    """
    Train the model and save it to disk.
        
    Returns:
        dict: A dictionary containing training results and the path to the saved model.
    """
    classifier = IrisClassifier()
    results = classifier.train()
    model_path = classifier.save()
    results["model_saved_to"] = model_path
        
    return results
    

if __name__ == "__main__":
    print("Training the Iris Classifier model...")
    results = train_and_save_model()
    print(f"Model trained and saved to {results['model_saved_to']}"
            f"\nAccuracy: {results['accuracy']:.2%}")
        
    print("Testing prediction on a new sample...")
    classifier = IrisClassifier()
    classifier.load()

    sample_input = [5.1, 3.5, 1.4, 0.2]
    prediction = classifier.predict(sample_input)
    print(f"Predicted class for the sample input {sample_input}: {prediction['predicted_class_name']} with confidence {prediction['confidence']:.2f}")
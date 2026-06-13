"""
MCP ML server with Resources.
"""

from mcp.server.fastmcp import FastMCP
from typing import Annotated
from pydantic import BeforeValidator
import json
import sys

def _parse_json_dict(v: object) -> dict:
    if isinstance(v, str):
        parsed = json.loads(v)
        if not isinstance(parsed, dict):
            raise ValueError("Input must be a JSON object, not a list or primitive")
        return parsed
    if isinstance(v, dict):
        return v
    raise ValueError("Input must be a JSON object string or dict")

JsonDict = Annotated[dict, BeforeValidator(_parse_json_dict)]

from model import IrisClassifier
from model_config import (
    MODEL_INFO,
    FEATURE_NAMES,
    FEATURE_DESCRIPTION,
    CLASS_NAMES,
    CLASS_DESCRIPTION,
    INPUT_SCHEMA
)

mcp = FastMCP("ml-prediction-server")

classifier = IrisClassifier()

# ===== Resources =====
@mcp.resource("model://info")
def get_model_info() -> dict:
    """Get information about the loaded machine learning model.
     Returns:
        dict: A dictionary containing model information."""
    return json.dumps(MODEL_INFO, indent=2)

@mcp.resource("model://schema")
def get_input_schema() -> dict:
    """Get the input schema for the machine learning model.
     Returns:
        dict: A dictionary containing the input schema."""
    return json.dumps(INPUT_SCHEMA, indent=2)

@mcp.resource("model://features")
def get_feature_info() -> dict:
    """Get information about the features used by the machine learning model.
     Returns:
        dict: A dictionary containing feature names and descriptions."""
    lines = ["# Model Features\n"]
    for name in FEATURE_NAMES:
        description = FEATURE_DESCRIPTION.get(name, "No description available.")
        lines.append(f"## {name}")
        lines.append(f"{description}\n")
    return "\n".join(lines)

@mcp.resource("model://classes")
def get_class_info() -> dict:
    """Get information about the classes predicted by the machine learning model.
     Returns:
        dict: A dictionary containing class names and descriptions."""
    lines = ["# Model Classes\n"]
    for i, name in enumerate(CLASS_NAMES):
        description = CLASS_DESCRIPTION.get(name, "No description available.")
        lines.append(f"## Class {i}: {name}")
        lines.append(f"{description}\n")
    return "\n".join(lines)

@mcp.resource("model://sample-input")
def get_sample_input() -> dict:
    """Get a sample input for the machine learning model.
     Returns:
        example input values that can be used for prediction."""
    sample = {
        "sepal_length_cm": 5.1,
        "sepal_width_cm": 3.5,
        "petal_length_cm": 1.4,
        "petal_width_cm": 0.2,
        "expected_class": "setosa",
        "note": "This is typical setosa specimen"
    }
    return json.dumps(sample, indent=2)

# ===== Tools =====

@mcp.tool()
def predict(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float
) -> str:
    """Make a prediction using the trained machine learning model.
     Args:
        sepal_length (float): The length of the sepal in centimeters.
        sepal_width (float): The width of the sepal in centimeters.
        petal_length (float): The length of the petal in centimeters.
        petal_width (float): The width of the petal in centimeters.
     Returns:
        str: A JSON string containing the prediction results, confidence and probabilities."""
    if not classifier.is_trained:
        if not classifier.load():
            print("Training a new model", file=sys.stderr)
            classifier.train()
            classifier.save()
            
    features = [sepal_length, sepal_width, petal_length, petal_width]
    result = classifier.predict(features)
    
    response = {
        "prediction": result["predicted_class_name"],
        "confidence": f"{result['confidence']:.1%}",
        "confidence_score": result["confidence"],
        "probabilities": {
            name: f"{prob:.1%}"
            for name, prob in result["probabilities"].items()
        },
        "input": {
            "sepal_length_cm": sepal_length,
            "sepal_width_cm": sepal_width,
            "petal_length_cm": petal_length,
            "petal_width_cm": petal_width
        }
    }
    
    return json.dumps(response, indent=2)
        
        
@mcp.tool()
def get_prediction_explanation(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float
) -> str:
    """Get an explanation for the prediction made by the machine learning model.
     Args:
        sepal_length (float): The length of the sepal in centimeters.
        sepal_width (float): The width of the sepal in centimeters.
        petal_length (float): The length of the petal in centimeters.
        petal_width (float): The width of the petal in centimeters.
     Returns:
        str: A JSON string containing the prediction explanation."""
    if not classifier.is_trained:
        if not classifier.load():
            print("Training a new model", file=sys.stderr)
            classifier.train()
            classifier.save()
            
    features = [sepal_length, sepal_width, petal_length, petal_width]
    result = classifier.predict(features)
    
    explanation = f"""
    # Prediction Explanation
    
    ## Input Features
    - Sepal Length: {sepal_length} cm
    - Sepal Width: {sepal_width} cm
    - Petal Length: {petal_length} cm
    - Petal Width: {petal_width} cm
    
    # Results
    **Predicted species**: {result['predicted_class_name'].upper()}
    
    **Confidence**: {result['confidence']:.1%}
    
    ## Probability distribution:
    """
    
    for name, prob in sorted(result['probabilities'].items(), key=lambda x: x, reverse=True):
        bar_length = int(prob * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        explanation += f"- {name}: {prob:.1%} {bar}\n"
        
    explanation += f"""
    ## Interpretation
    The model is {result['confidence']:.0%} confident this is **{result['predicted_class_name'].upper()}** based on the input features.
    """
    
    return explanation

@mcp.tool()
def batch_predict(samples_json: str) -> str:
    """Make predictions for a batch of samples using the trained machine learning model.
     Args:
        samples_json (str): A JSON string containing a list of samples, where each sample is a dictionary with feature values.
     Returns:
        str: A JSON string containing the prediction results for each sample."""
    if not classifier.is_trained:
        if not classifier.load():
            print("Training a new model", file=sys.stderr)
            classifier.train()
            classifier.save()
    
    try:
        samples = json.loads(samples_json)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON input. Please provide a valid JSON string.")
    
    results = []
    for i, sample in enumerate(samples):
        try:
            features = [
                sample["sepal_length"],
                sample["sepal_width"],
                sample["petal_length"],
                sample["petal_width"]
            ]
            prediction = classifier.predict(features)
            results.append({
                "input": sample,
                "sample_index": i,
                "prediction": prediction["predicted_class_name"],
                "confidence": prediction['confidence'],
                "confidence_score": prediction["confidence"],
            })
        except (KeyError, TypeError) as e:
            results.append({
                "input": sample,
                "sample_index": i,
                "error": str(e)
            })
            
    return json.dumps(results, indent=2)

@mcp.tool()
def train_model() -> str:
    """Train or retrain the machine learning model and return the training results.
     Returns:
        str: A JSON string containing the training results, including accuracy and other metrics."""
    print("Training the model...")
    global classifier
    
    results = classifier.train()
    classifier.save()
    
    response = {
        "accuracy": f"{results['accuracy']:.2%}",
        "n_train_samples": results["n_train_samples"],
        "n_test_samples": results["n_test_samples"],
        "model_saved_to": classifier.model_path,
        "message": "Model trained and saved successfully.",
        "training_details": {
            "n_estimators": classifier.model.n_estimators,
            "max_depth": classifier.model.max_depth
        }
    }
    
    return json.dumps(response, indent=2)

# ===== Prompts =====

@mcp.prompt()
def analyze_prediction(species: str, confidence: str) -> str:
    """Generate a prompt for analyzing a prediction.
     Args:
        species (str): The predicted species name.
        confidence (str): The confidence level of the prediction.
     Returns:
        str: A natural language analysis of the prediction results."""
    return f"""
            please analyze the Iris classification:
            - Predicted species: {species}
            - Confidence: {confidence}
            
            Tell me:
            1. what characteristics define this species?
            2. how reliable is this prediction and confidence level?
            3. what else might help confirm this classification?  
            """

@mcp.prompt()
def compare_samples(sample1: JsonDict, sample2: JsonDict) -> str:
    """Generate a prompt for comparing two samples.
     Args:
        sample1 (dict): The first sample, containing feature values and predicted class.
        sample2 (dict): The second sample, containing feature values and predicted class.
     Returns:
        str: A natural language analysis of the similarities and differences between the two samples."""
    return f"""
            please compare these two Iris samples:
            
            Sample 1:{sample1}
            Sample 2:{sample2}
            
            Analyze:
            1. key differences in measurements
            2. how differences affect classification
            3.which species each likely belong to
           """
           
@mcp.prompt()
def explain_features() -> str:
    """ Generate a prompt asking for feature explanation"""
    return """
            Explain the four Iris classification features:
            
            1. Sepal Length
            2. Sepal Width
            3. Petal Length
            4. Petal Width
            
            For each: what it measures, typical ranges, and how it helps differentiate between species."""
    

# ===== Main Server Loop =====

def main():
    """Run the server"""
    if classifier.load():
        print("Loaded existing model from disk.", file=sys.stderr)
    else:
        print("No existing model found. Training a new model.", file=sys.stderr)
        
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
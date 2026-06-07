"""
Model configuration and metadata.
"""

MODEL_INFO = {
    "name": "Iris Classifier",
    "version": "1.0.0",
    "algorithm": "Random Forest Classifier",
    "framework": "scikit-learn",
    "description": "A machine learning model that classifies iris flowers into three species based on their features.",
    "trained_date": "2024-06-01",
}

FEATURE_NAMES = [
    "sepal_length_cm", 
    "sepal_width_cm", 
    "petal_length_cm", 
    "petal_width_cm"
    ]
FEATURE_DESCRIPTION = {
    "sepal_length_cm": "The length of the sepal in centimeters.",
    "sepal_width_cm": "The width of the sepal in centimeters.",
    "petal_length_cm": "The length of the petal in centimeters.",
    "petal_width_cm": "The width of the petal in centimeters."
}

CLASS_NAMES = ["setosa", "versicolor", "virginica"]

CLASS_DESCRIPTION = {
    "setosa": "Iris Setosa is characterized by small petals",
    "versicolor": "Iris Versicolor has medium-sized petals and is often found in wetlands.",
    "virginica": "Iris Virginica has large petals"
}

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "sepal_length_cm": {"type": "number","description": "The length of the sepal in centimeters.","minimum": 4.0,"maximum": 8.0},
        "sepal_width_cm": {"type": "number","description": "The width of the sepal in centimeters.","minimum": 2.0,"maximum": 5.0},
        "petal_length_cm": {"type": "number","description": "The length of the petal in centimeters.","minimum": 1.0,"maximum": 7.0},
        "petal_width_cm": {"type": "number","description": "The width of the petal in centimeters.","minimum": 0.1,"maximum": 2.5}
    },
    "required": ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
}


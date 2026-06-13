# MCPetal

A lightweight Model Context Protocol (MCP) server for the classic Iris dataset. This repository demonstrates how to build an interactive model service with resources, tools, and prompt templates using `mcp[cli]` and a trained scikit-learn classifier.

## Suggested repository names

- `IrisMCP`
- `ContextIris`
- `MCP-IrisHub`
- `IrisBridge`
- `IrisScope`

## What this project does

This repository exposes an Iris flower classifier through MCP resources and tools. It includes:

- A trained Iris classification model backed by `scikit-learn`
- MCP resources for model metadata, schema, feature descriptions, and sample inputs
- MCP tools for prediction, explanation, batch prediction, and model training
- MCP prompts for generating analysis and comparison responses

## Key features

- `model://info` — model metadata and version details
- `model://schema` — input schema for prediction requests
- `model://features` — feature descriptions for the Iris dataset
- `model://classes` — class information for each Iris species
- `predict(...)` — single-sample prediction endpoint
- `get_prediction_explanation(...)` — explanation and confidence summary
- `batch_predict(...)` — predict multiple samples from JSON
- `train_model()` — retrain and save the model
- prompt templates for deeper analysis and comparison

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uv run python server.py
```

Or run with the MCP inspector:

```bash
npx @modelcontextprotocol/inspector uv run python server.py
```

## Recommended workflow

1. Activate the virtual environment.
2. Install dependencies.
3. Launch `server.py`.
4. Use MCP-compatible tools or the inspector to call resources, tools, and prompts.

## Files to know

- `server.py` — MCP resource/tool definitions and server startup
- `model.py` — Iris classifier wrapper and persistence logic
- `model_config.py` — dataset schema, feature names, and descriptions
- `iris_model.joblib` — serialized trained model file
- `main.ipynb` — interactive exploration and development notebook

## Why this repo is useful

This project is a great starting point for any MCP-driven machine learning deployment. It shows how to combine a classic model with structured MCP interfaces, making the model accessible, explainable, and easy to extend.


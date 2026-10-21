# Getting Started with Auraflux-Core

This guide helps you set up your development environment, install dependencies, and run your first validation steps for **Auraflux-Core**.

## 1. Prerequisites

- Python 3.10+
- Conda or virtual environment tool (as required by project environment governance rules)

## 2. Installation

Clone the repository and install dependencies within your designated virtual environment:

```bash
# Example using conda or venv
pip install -r requirements.txt
```

## 3. Basic Configuration

Ensure environment variables or configuration files under `src/auraflux_core/core/configs/` are properly set for your LLM providers (e.g., Gemini, OpenAI, vLLM).

## 4. Running Tests

Verify your installation by running tests through pytest in your isolated environment:

```bash
pytest
```

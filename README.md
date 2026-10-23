# fypBackend

Backend for the final-year project.

## Requirements

- Python 3.11.9
- pip

The project dependencies are listed in
[requirements.txt](C:/Users/User/Desktop/fypBackend/requirements.txt).

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements include Django, Django REST Framework, PostgreSQL support,
data-science libraries, NLP libraries, TensorFlow, PyTorch, and Hugging Face
Transformers.

## Verify the installation

```powershell
python -m pip check
```

To verify the main data-science and NLP libraries:

```powershell
python -c "import sklearn, nltk, spacy, tensorflow, torch, transformers; print('All libraries imported successfully.')"
```

## Environment variables

Create a local `.env` file for environment-specific settings and  secrets.
Do not commit `.env`; it is excluded by `.gitignore`.

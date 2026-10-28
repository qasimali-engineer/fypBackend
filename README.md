# fypBackend

Backend for the final-year project.

## Requirements

- Python 3.11.9
- pip

The project dependencies are listed in
[requirements.txt](C:/Users/User/Desktop/fypBackend/requirements.txt).

## Setup from a fresh clone

The Git repository contains the application source code only. The dataset and
the fine-tuned model are intentionally excluded from Git because together they
are several gigabytes and exceed normal GitHub file limits. A new clone must
therefore obtain the local artifacts before the API can make predictions.

Clone the repository and enter it:

```powershell
git clone https://github.com/qasimali-engineer/fypBackend.git
cd fypBackend
```

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

Apply the Django migrations:

```powershell
python manage.py migrate
```

The requirements include Django, Django REST Framework, PostgreSQL support,
data-science libraries, NLP libraries, TensorFlow, PyTorch, and Hugging Face
Transformers.

## Obtain the dataset and trained model

The following paths and files are required:

```text
Data_set/
  nonsuicide_english.csv
  nonsuicide_urdu.csv
  nonsuicide_roman_urdu.csv
  suicide_english.csv
  suicide_urdu.csv
  suicide_roman_urdu.csv

fyp-xlmr-model/
  config.json
  model.safetensors
  tokenizer.json
  tokenizer_config.json
  training_args.bin
```

These files are not downloaded by `git clone`. At present, this repository
does not publish a public download URL for the custom dataset or trained
model. Copy them from the project owner or download them from the project's
approved private storage, preserving the folder names above. For example, if
an archive is supplied, extract it at the repository root:

```powershell
Expand-Archive .\fypBackend-artifacts.zip -DestinationPath . -Force
```

After copying or extracting the artifacts, verify that the model can be
loaded:

```powershell
python -c "from pathlib import Path; from transformers import AutoTokenizer, AutoModelForSequenceClassification; p=Path('fyp-xlmr-model'); AutoTokenizer.from_pretrained(p); AutoModelForSequenceClassification.from_pretrained(p); print('Model loaded successfully.')"
```

If a trained model is not available, generate one locally by following the
training instructions below. Training requires the six language-specific CSV
files in `Data_set/`.

## Verify the installation

```powershell
python -m pip check
```

To verify the main data-science and NLP libraries:

```powershell
python -c "import sklearn, nltk, spacy, tensorflow, torch, transformers; print('All libraries imported successfully.')"
```

## Fine-tune XLM-RoBERTa

The training data is kept locally in `Data_set/` and is excluded from Git
because the exports are too large for a normal GitHub repository. The training
script uses the six language-specific CSV files and intentionally excludes the
combined/merged exports to prevent duplicate examples:

```powershell
python scripts/train_xlmr.py --epochs 3 --rows-per-file 20
```

The trained model and checkpoint files in `fyp-xlmr-model/` are also local
artifacts and are excluded from Git. Generate them locally before starting the
API. By default this test run reads only the first 20 rows from each of the six
language/label CSV files (up to 120 examples). It creates `fyp-xlmr-model/`,
selects the checkpoint with the highest suicide-label recall, and saves
evaluation metrics alongside the model. Use `--rows-per-file` to change the
sample size. For production training, use the full files with
`--rows-per-file 60000` and review `recall_suicide` before deploying.

For a complete local training run using all available rows, use:

```powershell
python scripts/train_xlmr.py `
  --data-dir .\Data_set `
  --output-dir .\fyp-xlmr-model `
  --epochs 3 `
  --rows-per-file 60000
```

Training creates the required `fyp-xlmr-model/` directory. It can take
substantial time and disk space, especially while checkpoints are being
written.

To generate a SHAP token-attribution visualization after training:

```powershell
python scripts/train_xlmr.py `
  --explain-text "Example text to explain" `
  --explanation-output shap-explanation.html
```

The generated HTML shows which tokens contributed to each class prediction.

## Self-harm classification API

After `fyp-xlmr-model/` has been copied, downloaded, or created by training,
start Django:

```powershell
python manage.py runserver
```

Send English, Urdu, or Roman Urdu text to:

```text
POST http://127.0.0.1:8000/api/classify/
Content-Type: application/json
```

Example request:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/classify/ `
  -H "Content-Type: application/json" `
  -d '{"text":"آپ کیسے ہیں؟"}'
```

Response:

```json
{
  "self_harm": "Yes",
  "class": "suicide",
  "confidence": 0.9231
}
```

`self_harm` follows the model prediction, with a narrow explicit-phrase
override for clear phrases such as “خود کو نقصان”, “khud ko nuqsan”, “kill
myself”, and “hurt myself”. This prevents the small test model from missing
an unambiguous self-harm statement when it is written in another script. The
For non-explicit text, the test model uses a 0.60 probability threshold to
avoid turning its borderline predictions into false positives. The response
includes the raw `suicide_probability` and `detection_method`.
The model is loaded once and reused for subsequent requests. The 120-example
test model is not a production-quality classifier; train with substantially
more data before deployment.

## Environment variables

Create a local `.env` file for environment-specific settings and  secrets.
Do not commit `.env`; it is excluded by `.gitignore`.

install:
	pip install -r requirements.txt

data:
	python scripts/download_data.py

run:
	python scripts/run_pipeline.py

app:
	streamlit run app.py

test:
	pytest -q

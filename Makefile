.PHONY: clean backend frontend

clean:
	rm -rf __pycache__
	rm -rf app/__pycache__
	rm -rf frontend/__pycache__

backend:
	uv run fastapi dev app/main.py --port 8000 --reload

frontend:
	uv run streamlit run frontend/app.py --server.port 8501
"""Entry point.

    python main.py                          # start the app at http://127.0.0.1:8000
    uvicorn app.main:app --reload           # development: restart on code changes
"""

from app.main import app  # noqa: F401  (re-exported so `uvicorn main:app` also works)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)

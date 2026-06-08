import json
import os
import threading
from functools import lru_cache
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from pipeline import QQPPipeline


ROOT = Path(__file__).parent
PUBLIC_DIR = ROOT / "public"
DATA_PATH = ROOT / "data" / "questions.csv"

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8501"))

MODEL_OPTIONS = {
    "model5": "Lexical + Semantic Ensemble (Recommended)",
    "model4": "Sentence-BERT Supervised Classifier",
    "model3": "Sentence-BERT Cosine (Calibrated)",
    "model2": "TF-IDF + Logistic Regression",
    "model1": "TF-IDF Cosine (Calibrated)",
}

_pipeline_lock = threading.Lock()


@lru_cache(maxsize=2)
def load_pipeline(with_sbert: bool) -> QQPPipeline:
    with _pipeline_lock:
        pipeline = QQPPipeline(data_path=str(DATA_PATH))
        pipeline.load_and_split_data()
        
        # Load serialized models if they exist, otherwise train once
        models_dir = ROOT / "outputs"
        if not pipeline.load_models(directory=str(models_dir)):
            print("Pre-trained models not found in outputs/. Training once on startup...")
            pipeline.train_models()
            pipeline.save_models(directory=str(models_dir))
        else:
            print("Pre-trained models loaded successfully.")
            
        if with_sbert and pipeline.sbert_model is None:
            pipeline.load_sbert()
            
        return pipeline


def predict_pair(q1: str, q2: str, model_key: str) -> dict:
    if model_key not in MODEL_OPTIONS:
        raise ValueError(f"Unknown model '{model_key}'")
    
    needs_sbert = model_key in ["model3", "model4", "model5"]
    pipeline = load_pipeline(with_sbert=needs_sbert)
    result = pipeline.predict(q1, q2, model_type=model_key)
    result["model_key"] = model_key
    return result


def compare_models(q1: str, q2: str, include_sbert: bool) -> list[dict]:
    keys = ["model1", "model2"]
    if include_sbert:
        keys = ["model1", "model2", "model3", "model4", "model5"]
    return [predict_pair(q1, q2, key) for key in keys]


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PUBLIC_DIR), **kwargs)

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/summary":
            self.send_json({"models": MODEL_OPTIONS})
            return
            
        if parsed.path == "/api/metrics":
            metrics_path = PUBLIC_DIR / "outputs" / "metrics.csv"
            if not metrics_path.exists():
                # Fallback to outputs/metrics.csv
                metrics_path = ROOT / "outputs" / "metrics.csv"
                
            if metrics_path.exists():
                try:
                    df = pd.read_csv(metrics_path)
                    data = df.to_dict(orient="records")
                    self.send_json({"metrics": data})
                except Exception as e:
                    self.send_json({"error": f"Failed to read metrics: {str(e)}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            else:
                self.send_json({"error": "Metrics file not found. Please run evaluate.py first."}, status=HTTPStatus.NOT_FOUND)
            return

        if parsed.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.path = "/index.html"
        return super().do_HEAD()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            payload = self.read_json()
            q1 = str(payload.get("question1", "")).strip()
            q2 = str(payload.get("question2", "")).strip()
            model = str(payload.get("model", "model5"))
            include_sbert = bool(payload.get("include_sbert", True))
            compare = bool(payload.get("compare", True))

            if not q1 or not q2:
                self.send_json(
                    {"error": "Enter both questions before running the model."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            response = {"result": predict_pair(q1, q2, model)}
            if compare:
                response["comparison"] = compare_models(q1, q2, include_sbert)
            self.send_json(response)
        except Exception as exc:
            self.send_json(
                {"error": str(exc)},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw or "{}")

    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main() -> None:
    PUBLIC_DIR.mkdir(exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print(f"QQP dashboard running at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()

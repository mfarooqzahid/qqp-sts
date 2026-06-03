import argparse

from pipeline import QQPPipeline


MODEL_NAMES = {
    "model1": "TF-IDF Cosine Similarity",
    "model2": "TF-IDF + Logistic Regression",
    "model3": "Sentence-BERT (MiniLM)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict whether two questions are duplicates."
    )
    parser.add_argument("question1", nargs="?", help="First question")
    parser.add_argument("question2", nargs="?", help="Second question")
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        default="model2",
        help="Model to use. Defaults to the offline-capable Logistic Regression model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    q1 = args.question1 or input("Enter Question 1: ")
    q2 = args.question2 or input("Enter Question 2: ")

    if not q1.strip() or not q2.strip():
        print("Both questions are required.")
        return

    pipeline = QQPPipeline()

    print("Initializing pipeline...")
    try:
        pipeline.load_and_split_data()
        pipeline.train_tfidf_models()
        if args.model == "model3":
            pipeline.load_sbert()
    except Exception as exc:
        print(f"Error initializing pipeline: {exc}")
        if args.model == "model3":
            print(
                "Sentence-BERT requires the MiniLM model to be cached locally or "
                "network access to download it."
            )
        return

    result = pipeline.predict(q1, q2, model_type=args.model)

    print("\n--- Result ---")
    print(f"Model: {result['model']}")
    print(f"Similarity Score: {result['score']:.4f}")
    print(f"Verdict: {result['verdict']}")
    print("--------------\n")


if __name__ == "__main__":
    main()

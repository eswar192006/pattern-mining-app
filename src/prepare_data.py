"""Run once:  python -m src.prepare_data   (from the project root)"""
from src.mining.data_loader import prepare_and_save, summarize

if __name__ == "__main__":
    df = prepare_and_save()
    print("Saved cleaned data.")
    for k, v in summarize(df).items():
        print(f"  {k}: {v}")
    print(df.head())

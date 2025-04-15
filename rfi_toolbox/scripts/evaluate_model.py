# rfi_toolbox/scripts/evaluate_model.py
import argparse
import torch
from rfi_toolbox.utils.evaluation import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained RFI masking model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model checkpoint")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the validation dataset directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for evaluation (cuda or cpu)")
    parser.add_argument("--in_channels", type=int, default=8, help="Number of input channels the model expects")
    args = parser.parse_args()

    results = evaluate_model(args.model_path, args.dataset_dir, args.batch_size, args.device, args.in_channels)

    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()

import argparse
import logging
import os

import numpy as np

from rfi_toolbox.core.simulator import RFISimulator
from rfi_toolbox.datasets import RFIMaskDataset


def save_example_pair_npy(tf_plane, mask, index, out_dir, generate_mask=True):
    """
    Saves input and mask numpy arrays.

    Args:
        tf_plane (dict): Dictionary containing the visibility data.
        mask (ndarray): The RFI mask.
        index (int): Index of the sample.
        out_dir (str): Output directory.
        generate_mask (bool): Whether to save the mask.
    """
    sample_dir = os.path.join(out_dir, f"{index:04d}")
    os.makedirs(sample_dir, exist_ok=True)
    input_data = np.stack(
        [
            tf_plane["RR"].real,
            tf_plane["RR"].imag,
            tf_plane["RL"].real,
            tf_plane["RL"].imag,
            tf_plane["LR"].real,
            tf_plane["LR"].imag,
            tf_plane["LL"].real,
            tf_plane["LL"].imag,
        ],
        axis=0,
    )  # shape: (8, time_bins, freq_bins)

    input_path = os.path.join(sample_dir, "input.npy")
    np.save(input_path, input_data)

    if generate_mask:
        mask_path = os.path.join(sample_dir, "rfi_mask.npy")
        np.save(mask_path, mask)


def main():
    """
    Main function to generate synthetic RFI dataset or read from an MS.
    """
    parser = argparse.ArgumentParser(
        description="Generate or load RFI dataset as numpy files."
    )
    parser.add_argument(
        "--samples_training", type=int, default=1000, help="Number of training samples."
    )
    parser.add_argument(
        "--samples_validation",
        type=int,
        default=200,
        help="Number of validation samples.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="rfi_dataset", help="Output directory."
    )
    parser.add_argument(
        "--only_clean",
        action="store_true",
        help="Generate only clean data without RFI (incompatible with --use_ms).",
    )
    parser.add_argument(
        "--time_bins",
        type=int,
        default=1024,
        help="Number of time bins in the TF plane.",
    )
    parser.add_argument(
        "--frequency_bins",
        type=int,
        default=1024,
        help="Number of frequency bins in the TF plane.",
    )
    parser.add_argument(
        "--generate_mask", action="store_true", default=True, help="Generate RFI masks."
    )
    parser.add_argument(
        "--no_generate_mask",
        action="store_false",
        dest="generate_mask",
        help="Disable mask generation.",
    )
    parser.add_argument(
        "--use_ms",
        action="store_true",
        default=False,
        help="Load data from a Measurement Set.",
    )
    parser.add_argument(
        "--ms_name", type=str, default=None, help="Path to the Measurement Set."
    )
    parser.add_argument(
        "--train_field", type=int, help="FIELD_ID to use for training set."
    )
    parser.add_argument(
        "--val_field", type=int, help="FIELD_ID to use for validation set."
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if args.use_ms:
        if not args.ms_name:
            logging.error("Error: --ms_name must be specified when --use_ms is used.")
            return

        if args.only_clean:
            logging.error(
                "Error: --only_clean is incompatible with --use_ms. --only_clean will be ignored."
            )
            return

        logging.info(f"Loading data from Measurement Set: {args.ms_name}")
        # Create a single directory to hold all MS samples
        ms_output_dir = os.path.join(args.output_dir, "ms_data")
        os.makedirs(ms_output_dir, exist_ok=True)

        # Create training and validation datasets, selecting fields if specified
        train_dataset = RFIMaskDataset(
            data_dir=ms_output_dir,
            use_ms=True,
            ms_name=args.ms_name,
            field_selection=args.train_field,
        )
        val_dataset = RFIMaskDataset(
            data_dir=ms_output_dir,
            use_ms=True,
            ms_name=args.ms_name,
            field_selection=args.val_field,
        )

        logging.info(f"Number of training samples from MS: {len(train_dataset)}")
        logging.info(f"Number of validation samples from MS: {len(val_dataset)}")

    else:
        simulator = RFISimulator(
            time_bins=args.time_bins, freq_bins=args.frequency_bins
        )

        if args.only_clean:
            logging.info("Generating only clean data without RFI.")
            train_dir = os.path.join(args.output_dir, "train")
            os.makedirs(train_dir, exist_ok=True)
            logging.info(
                f"Generating {args.samples_training} clean samples in '{train_dir}' (mask generation: {args.generate_mask})"
            )

            for i in tqdm(range(args.samples_training), desc="Clean"):
                tf_plane, mask = simulator.generate_clean_data()
                save_example_pair_npy(
                    tf_plane,
                    mask,
                    index=i,
                    out_dir=train_dir,
                    generate_mask=args.generate_mask,
                )

            tf_plane = simulator.tf_plane
            mask = np.zeros_like(tf_plane["RR"], dtype=bool)
        else:
            # Train samples
            train_dir = os.path.join(args.output_dir, "train")
            os.makedirs(train_dir, exist_ok=True)
            logging.info(
                f"Generating {args.samples_training} training samples in '{train_dir}' (mask generation: {args.generate_mask})"
            )
            for i in tqdm(range(args.samples_training), desc="Training"):
                tf_plane, mask = simulator.generate_rfi()
                save_example_pair_npy(
                    tf_plane,
                    mask,
                    index=i,
                    out_dir=train_dir,
                    generate_mask=args.generate_mask,
                )

            # Validation samples
            val_dir = os.path.join(args.output_dir, "val")
            os.makedirs(val_dir, exist_ok=True)
            logging.info(
                f"Generating {args.samples_validation} validation samples in '{val_dir}' (mask generation: {args.generate_mask})"
            )
            for i in tqdm(range(args.samples_validation), desc="Validation"):
                tf_plane, mask = simulator.generate_rfi()
                save_example_pair_npy(
                    tf_plane,
                    mask,
                    index=i,
                    out_dir=val_dir,
                    generate_mask=args.generate_mask,
                )
        logging.info("Dataset generation complete.")


if __name__ == "__main__":
    main()

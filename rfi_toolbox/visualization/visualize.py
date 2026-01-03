# rfi_toolbox/visualization/visualize.py
import numpy as np
import torch
from rfi_toolbox.datasets import RFIMaskDataset
from rfi_toolbox.models.unet import UNet
from bokeh.plotting import figure, show
from bokeh.layouts import column, row
from bokeh.models import Slider, ColumnDataSource
from bokeh.palettes import Viridis256, Gray256
import random
import argparse

def create_image_plot(image_data, title, width=250, height=250, palette=Viridis256):
    p = figure(width=width, height=height, title=title, x_range=(0, image_data.shape[1]), y_range=(0, image_data.shape[0]))
    p.image(image=[image_data], x=0, y=0, dw=image_data.shape[1], dh=image_data.shape[0], palette=palette)
    return p

def create_interactive_viewer(dataset_dir, model_path=None, device="cpu", in_channels=8, num_samples=100, seed=42):
    val_dataset = RFIMaskDataset(dataset_dir)
    random.seed(seed)
    indices = random.sample(range(len(val_dataset)), min(num_samples, len(val_dataset)))
    selected_samples = [val_dataset[i] for i in sorted(indices)]

    source = ColumnDataSource(data=dict(
        input_ch0=[np.zeros((1, 1))],
        input_ch1=[np.zeros((1, 1))],
        input_ch2=[np.zeros((1, 1))],
        input_ch3=[np.zeros((1, 1))],
        mask=[np.zeros((1, 1))],
        prediction=[np.zeros((1, 1))]
    ))

    plot_ch0 = create_image_plot(source.data['input_ch0'][0], "Input RR Amp")
    plot_ch1 = create_image_plot(source.data['input_ch1'][0], "Input RL Amp")
    plot_ch2 = create_image_plot(source.data['input_ch2'][0], "Input LR Amp")
    plot_ch3 = create_image_plot(source.data['input_ch3'][0], "Input LL Amp")
    plot_mask = create_image_plot(source.data['mask'][0], "Ground Truth Mask", palette=Gray256)
    plot_prediction = create_image_plot(source.data['prediction'][0], "Model Prediction", palette=Gray256)

    if model_path:
        model = UNet(in_channels=in_channels, out_channels=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

    slider = Slider(start=0, end=len(selected_samples) - 1, value=0, step=1, title="Sample Index")

    def update(attr, old, new):
        index = new
        if 0 <= index < len(selected_samples):
            input_tensor, mask = selected_samples[index]
            input_np = input_tensor.numpy()

            source.data = dict(
                input_ch0=[input_np[0]],
                input_ch1=[input_np[1]],
                input_ch2=[input_np[2]],
                input_ch3=[input_np[3]],
                mask=[mask.numpy()[0]],
                prediction=[np.zeros_like(mask.numpy()[0])]
            )

            if model_path:
                with torch.no_grad():
                    model_input = input_tensor.unsqueeze(0).to(device)
                    prediction = torch.sigmoid(model(model_input)).cpu().numpy()[0, 0]
                    source.patch({'prediction': [(index, prediction)]})

    slider.on_change('value', update)
    update(None, None, 0) # Initialize with the first sample

    layout = column(slider,
                    row(plot_ch0, plot_ch1),
                    row(plot_ch2, plot_ch3),
                    row(plot_mask, plot_prediction))
    return layout

def main():
    parser = argparse.ArgumentParser(description="Interactive visualization of RFI masking validation data and model predictions.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the validation dataset directory")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model checkpoint (optional)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for model inference (cuda or cpu)")
    parser.add_argument("--in_channels", type=int, default=8, help="Number of input channels the model expects")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of random samples to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    args = parser.parse_args()

    dashboard = create_interactive_viewer(args.dataset_dir, args.model_path, args.device, args.in_channels, args.num_samples, args.seed)
    show(dashboard)

if __name__ == "__main__":
    main()

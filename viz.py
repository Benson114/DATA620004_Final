import argparse
from torchviz import make_dot

from config.Config import *
from src.Models import SimpleCNN, VisionTransformer


def main(model_type):
    if model_type == "SimpleCNN":
        model = SimpleCNN(SimpleCNN_kwargs)
    elif model_type == "ViT":
        model = VisionTransformer(**ViT_kwargs)
    else:
        raise ValueError("model_type must be one of ['SimpleCNN', 'ViT']")

    # model.load("models", f"{model_type}.ckpt")
    x = torch.randn(1, 3, 32, 32)

    y = model(x)
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.render(f"img/{model_type}_graph", format="png")
    print(f"Graph saved as {model_type}_graph.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, required=True, choices=["SimpleCNN", "ViT"])
    args = parser.parse_args()

    main(args.model_type)

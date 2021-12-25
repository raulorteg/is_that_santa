import torch
import numpy as np
from torch.nn.modules.loss import BCELoss
import torchvision.transforms as transforms
from PIL import Image
from utils_dataset import SantaDataset
from utils_plotting import image_grid, compute_accuracy
from models import SantaFinder


def is_that_santa(resize, filename):

    transform = transforms.Compose(
                    [
                    transforms.Resize((resize,resize),interpolation=Image.NEAREST),
                    transforms.ToTensor(),
                    ])

    # load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SantaFinder(input_shape=[0,3,resize,resize]).to(device)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load("model_savepoints/model.pt")["model_state_dict"])
    else:
        net.load_state_dict(torch.load("model_savepoints/model.pt", map_location=torch.device('cpu'))["model_state_dict"])

    # cast the image into the needed formats
    image = Image.open(filename)
    image = image.convert('RGB')
    image.show()
    image = transform(image)
    image = image.unsqueeze(0)

    # predict
    net.eval()
    pred = net(image.to(device))["preds"]
    pred = pred.item()

    percentage_not_santa = (1-pred)*100
    percentage_santa = (pred)*100

    if percentage_santa >= percentage_not_santa:
        print(f"Hello Santa! (Santa: {percentage_santa:.2f}%, Not-santa: {percentage_not_santa:.2f}%)")
    else:
        print(f"Santa not there :(. (Santa: {percentage_santa:.2f}%, Not-santa: {percentage_not_santa:.2f}%)")

if __name__ == "__main__":

    import argparse
    # example: python is_that_santa.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--resize", help="Pixels of the resized image (defaults to 128)", default=128, type=int)
    parser.add_argument("--filename", help="filename to the image to classify (defaults to external/example.jpg)", default="external/example.jpg", type=str)
    args = parser.parse_args()

    is_that_santa(resize=args.resize, filename=args.filename)

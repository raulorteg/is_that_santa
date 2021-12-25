import torch
import numpy as np
from torch.nn.modules.loss import BCELoss
import torchvision.transforms as transforms
from PIL import Image
from utils_dataset import SantaDataset
from utils_plotting import image_grid, compute_accuracy
from models import SantaFinder


def main(n_epochs, lr, batch_size, resize, val_freq):
    
    # load datasets
    transform = transforms.Compose(
                    [
                    transforms.Resize((resize,resize),interpolation=Image.NEAREST),
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(),
                    ])
    
    transform_eval = transforms.Compose(
                    [
                    transforms.Resize((resize,resize),interpolation=Image.NEAREST),
                    transforms.ToTensor(),
                    ])

    dataset = SantaDataset(txt_file="datasets/train_dataset.txt", root_dir="datasets/train/", transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset_eval = SantaDataset(txt_file="datasets/test_dataset.txt", root_dir="datasets/test/", transform=transform_eval)
    dataloader_eval = torch.utils.data.DataLoader(dataset, batch_size=32)
    
    for sample in dataloader:
        input_shape = sample["image"].shape
        break

    with open("results/results.txt", "w") as f:
        print("epoch,train_loss,train_acc,test_loss,test_acc", file=f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SantaFinder(input_shape=input_shape).to(device)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    for epoch in range(1,n_epochs+1):
        
        # training loop
        running_loss = 0.0
        acc_buffer = []
        net.train()
        for sample in dataloader:
            batch_size = sample["image"].shape[1]

            images, labels = sample["image"].to(device), sample["label"].to(device)

            optimizer.zero_grad()

            predictions = net(images)
            preds = predictions["preds"]

            loss = criterion(preds.double(), labels.unsqueeze(1).double())

            loss.backward()
            optimizer.step()

            running_loss += loss.cpu().detach().item()/batch_size
            acc_buffer.append(compute_accuracy(preds.squeeze().detach().cpu().numpy(), labels.squeeze().detach().cpu().numpy()))
        
        train_loss.append(running_loss/len(dataset))
        train_acc.append(np.array(acc_buffer).mean())

        # validation loop
        if epoch % val_freq == 0:
            acc_buffer = []
            running_loss = 0.0
            net.eval()
            for sample in dataloader_eval:
                images, labels = sample["image"].to(device), sample["label"].to(device)

                predictions = net(images)
                preds = predictions["preds"]

                loss = criterion(preds.double(), labels.unsqueeze(1).double())

                running_loss += loss.cpu().detach().item()/batch_size
                acc_buffer.append(compute_accuracy(preds.squeeze().detach().cpu().numpy(), labels.squeeze().detach().cpu().numpy()))
            
            test_loss.append(running_loss/len(dataset_eval))
            test_acc.append(np.array(acc_buffer).mean())

            with open("results/results.txt", "a+") as f:
                print(f"{epoch},{train_loss[-1]},{train_acc[-1]},{test_loss[-1]},{test_acc[-1]}", file=f)
                print(f"epoch: {epoch}/{n_epochs},train_loss: {train_loss[-1]} ({train_acc[-1]*100:.2f}%), test_loss: {test_loss[-1]} ({test_acc[-1]*100:.2f}%)")

    # save last iteration
    torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                }, "model_savepoints/model.pt")
    print("Done!")

if __name__ == "__main__":

    import argparse
    # parsing user input
    # example: python main.py --n_epochs=100 --lr=0.0001 --batch_size=32 --latent_size=100 --resize=128 --val_freq=1
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", help="Number of epochs (defaults 100)", default=10, type=int)
    parser.add_argument("--lr", help="Learning rate (defaults 0.002)", default=0.002, type=float)
    parser.add_argument("--batch_size", help="Batch size (defaults 32)", default=32, type=int)
    parser.add_argument("--resize", help="Pixels of the resized image (defaults to 128)", default=128, type=int)
    parser.add_argument("--val_freq", help="Validation frequency (defaults to every 5 epochs)", default=1, type=int)
    args = parser.parse_args()

    main(n_epochs=args.n_epochs, lr=args.lr, batch_size=args.batch_size, resize=args.resize, val_freq=args.val_freq)

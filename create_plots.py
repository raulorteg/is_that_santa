import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_plots(path:str="results/results.txt"):
    results_training = pd.read_csv(path)

    fig, axs = plt.subplots(1,2, figsize=(12,6))
    axs[0].plot(results_training["epoch"], results_training["train_loss"], label="Training")
    axs[0].plot(results_training["epoch"], results_training["test_loss"], label="Testing")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("BCELoss")

    axs[1].plot(results_training["epoch"], results_training["train_acc"], label="Training")
    axs[1].plot(results_training["epoch"], results_training["test_acc"], label="Testing")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Acccuracy")

    plt.legend()
    plt.savefig("results/training_plots.png")
    print("Done!")

if __name__ == "__main__":
    import argparse
    # parsing user input
    # example: python create_plots.py --path="results/results.txt"
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to find filename containing results (defaults results/results.txt)", default="results/results.txt", type=str)
    args = parser.parse_args()

    create_plots(args.path)
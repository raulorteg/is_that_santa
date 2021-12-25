from pathlib import Path, PurePath

def create_db(path_imgs, name_db):
    with open(f"datasets/{name_db}", "w+") as f:
        print("filename,class_label", file=f)
        path_imgs = Path(path_imgs)
        for file in path_imgs.iterdir():
            filename = PurePath(file).parts[-1]
            if "not" in filename:
                class_label = 0.0
            else:
                class_label = 1.0
            
            print(f"{filename}, {class_label}", file=f)

if __name__ == "__main__":

    import argparse
    # parsing user input
    # example: python create_db.py --path_imgs="datasets/train" --name_db="train_dataset.txt"
    # example: python create_db.py --path_imgs="datasets/test" --name_db="test_dataset.txt"
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_imgs", help="Path to find the images (defaults datasets/train)", default="datasets/train", type=str)
    parser.add_argument("--name_db", help="Name to use in the resulting txt file", default="train_dataset.txt", type=str)
    args = parser.parse_args()

    create_db(path_imgs=args.path_imgs, name_db=args.name_db)
import torch
from torchvision import datasets, transforms

import argparse


def calculate_mean_std(path_to_dataset):
    """Given the root of an image dataset, it calculates and saves the mean and std of the dataset

    Parameters
    ----------
    path_to_dataset : str
        Root directory of dataset
    """
    transform = transforms.Compose([transforms.ToTensor(),])

    dataset = datasets.ImageFolder(path_to_dataset, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
    print("There are total of", len(dataset), "images")

    mean = torch.zeros(3)
    std = torch.zeros(3)

    for i, (data, _) in enumerate(dataloader):
        data = data[0].squeeze(0)
        if (i == 0): size = data.size(1) * data.size(2)
        mean += data.sum((1, 2)) / size

    mean /= len(dataset)
    print("Mean is")
    print(mean)
    mean = mean.unsqueeze(1).unsqueeze(2)

    for i, (data, _) in enumerate(dataloader):
        data = data[0].squeeze(0)
        std += ((data - mean) ** 2).sum((1, 2)) / size

    std /= len(dataset)
    std = std.sqrt()

    print("Mean is")
    print(mean)
    print("std is")
    print(std)

    torch.save(mean, 'mean.pt')
    torch.save(std, 'std.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate mean and std of Image dataset')
    parser.add_argument('source_slides_folder', type=str, default=None,
                        help='Root folder containing slides')
    args = parser.parse_args()
    calculate_mean_std(args.source_slides_folder)

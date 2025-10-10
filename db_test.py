from src.datasets.custom_dataset import CustomDataset

FILE_PATH = "custom_dataset"

ds = CustomDataset(FILE_PATH)
for i in range(len(ds)):
    print(ds[i])

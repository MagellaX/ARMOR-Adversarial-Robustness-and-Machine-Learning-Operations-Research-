from src.utils import DataLoader

loader = DataLoader()
data = loader.load_data('mnist')
print('Data loaded successfully!')
print(f'Training data shape: {data["x_train"].shape}')

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm


movies_df = pd.read_csv("ml-latest-small/movies.csv")
ratings_df = pd.read_csv("ml-latest-small/ratings.csv")

"""Data preprocessing"""

movies_df["production_year"] = movies_df.title.str.extract(
    r"\((\d{4})\)", expand=False
).astype(float)  # extracting years from titles
movies_df["title"] = movies_df.title.str.replace(
    pat=r"\(\d{4}\)", repl="", regex=True
)  # deleteing years of production from titles
movies_df["genres"] = movies_df.genres.str.split("|")

print(
    movies_df["production_year"].min(), movies_df["production_year"].max()
)  # range of production years

mlb = MultiLabelBinarizer()

movies_genres = pd.DataFrame(
    mlb.fit_transform(movies_df.genres), columns=mlb.classes_, index=movies_df.index
)

movies_df = pd.concat(
    [movies_df, movies_genres], axis=1
)  # join movies_df and movies_genres along columns

mms = MinMaxScaler(feature_range=(0, 1))

ratings_df[["userId_scaled", "movieId_scaled"]] = mms.fit_transform(
    ratings_df[["userId", "movieId"]]
)
movies_df["years_scaled"] = mms.fit_transform(movies_df[["production_year"]])

data = ratings_df.merge(movies_df, on="movieId")

data["years_scaled"].fillna(
    data["years_scaled"].median(), inplace=True
)  # filling missing years with years median


def user_train_test_split(df, train_size=0.8, seed=12):
    np.random.seed(seed)

    train_list = []
    test_list = []

    for userId, user_df in df.groupby("userId"):
        number_of_ratings = len(user_df)
        ratings_number_to_train = int(number_of_ratings * train_size)

        train_indices = np.random.choice(
            user_df.index, size=ratings_number_to_train, replace=False
        )
        test_indices = user_df.index.difference(train_indices)

        train_list.append(df.iloc[train_indices])
        test_list.append(df.iloc[test_indices])

    train_df = pd.concat(train_list, axis=0)
    test_df = pd.concat(test_list, axis=0)

    return train_df, test_df


train_data, test_data = user_train_test_split(data)
print(train_data.head())


"""Model definition"""

features_list = [
    "userId_scaled",
    "movieId_scaled",
    "years_scaled",
] + mlb.classes_.tolist()


class MovieDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.values).float()
        self.y = torch.from_numpy(y.values).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class Net(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x):
        y = self.net(x)

        return y


device = torch.device("cuda:0")

net = Net(len(features_list), 1, 256)
net.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters())

train_dataset = MovieDataset(X=train_data[features_list], y=train_data["rating"])
test_dataset = MovieDataset(X=test_data[features_list], y=test_data["rating"])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

epochs_bar = tqdm.tqdm(range(30))

for epoch in epochs_bar:
    total_loss = 0
    total_rmse = 0

    for X_train, y_train in train_dataloader:
        X_train = X_train.to(device)
        y_train = y_train.to(device)

        optimizer.zero_grad()
        y_predicted = net(X_train).squeeze()

        loss = criterion(y_predicted, y_train)
        loss.backward()

        optimizer.step()

        total_loss += loss.cpu().item()
        total_rmse += torch.sqrt(loss.detach().cpu()).item()

    # print(total_loss)

    epochs_bar.set_description(f"total rmse: {total_rmse / (len(train_data) / 64)}")

# mlpreg = MLPRegressor(
#     hidden_layer_sizes=256,
#     activation="relu",
#     batch_size=64,
#     random_state=12,
#     verbose=True,
# )


# # print(data[features_list].isna().sum()) #checking if some features are NaN

# mlpreg.fit(X=train_data[features_list], y=train_data["rating"])

# predicted = mlpreg.predict(test_data[features_list])
# expected = test_data["rating"]
# print(predicted, expected)


# def calculate_rmse_mae(true_values, estimated_values):
#     rmse = root_mean_squared_error(true_values, estimated_values)
#     mae = mean_absolute_error(true_values, estimated_values)

#     print(f"RMSE of the model: {rmse:.3f}")  # root mean squared error
#     print(f"MAE of the model: {mae:.3f}")  # mean absolute error


# calculate_rmse_mae(expected, predicted)

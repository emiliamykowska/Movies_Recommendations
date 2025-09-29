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

# ratings_df[["userId_scaled", "movieId_scaled"]] = mms.fit_transform(
#     ratings_df[["userId", "movieId"]]
# )


movies_df["years_scaled"] = mms.fit_transform(movies_df[["production_year"]])

data = ratings_df.merge(movies_df, on="movieId")

data["years_scaled"].fillna(
    data["years_scaled"].median(), inplace=True
)  # filling missing years with years median


movie_counts = data.groupby("movieId").size()  # get counts of movie ratings
movies_to_keep = movie_counts[
    movie_counts >= 3
].index  # get indices of movies with ratings >= 3

data = data[
    data["movieId"].isin(movies_to_keep)
]  # only leaves rows with movieId which are in movies_to_keep


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

        train_list.append(df.loc[train_indices])
        test_list.append(df.loc[test_indices])

    train_df = pd.concat(train_list, axis=0)
    test_df = pd.concat(test_list, axis=0)

    user2idx = {user: i for i, user in enumerate(df["userId"].unique())}
    movie2idx = {movie: i for i, movie in enumerate(df["movieId"].unique())}

    # Apply mappings to train and test, so each original userId is replaced with a 0-based integer
    train_df["userId_cat"] = train_df["userId"].map(user2idx)
    train_df["movieId_cat"] = train_df["movieId"].map(movie2idx)
    test_df["userId_cat"] = test_df["userId"].map(user2idx)
    test_df["movieId_cat"] = test_df["movieId"].map(movie2idx)

    return train_df, test_df


train_data, test_data = user_train_test_split(data)

print(train_data.head())


"""Model definition"""

features_list = [
    # "userId_scaled",
    # "movieId_scaled",
    "userId_cat",
    "movieId_cat",
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
    def __init__(
        self,
        num_users,
        num_movies,
        embedding_dim,
        features_dim,
        out_channels,
        hidden_channels,
    ):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        in_channels = 2 * embedding_dim + features_dim  # 2 * coz 2x embedding

        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(self, user_ids, movie_ids, features):
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)

        x = torch.cat([user_emb, movie_emb, features], dim=1)

        return self.net(x).squeeze(-1)


device = torch.device("cuda:0")

net = Net(
    num_users=data.userId.nunique(),
    num_movies=data.movieId.nunique(),
    embedding_dim=16,
    features_dim=len(features_list) - 2,
    out_channels=1,
    hidden_channels=64,
)
net.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), weight_decay=0.0005)

train_dataset = MovieDataset(X=train_data[features_list], y=train_data["rating"])
test_dataset = MovieDataset(X=test_data[features_list], y=test_data["rating"])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

epochs_bar = tqdm.tqdm(range(50))

for epoch in epochs_bar:
    total_loss = 0
    total_rmse = 0
    total_mae = 0

    for X_train, y_train in train_dataloader:
        X_train = X_train.to(device)
        y_train = y_train.to(device)

        user_ids = X_train[:, 0].long()
        movie_ids = X_train[:, 1].long()
        side_features = X_train[:, 2:]

        optimizer.zero_grad()
        y_predicted = net(user_ids, movie_ids, side_features)

        loss = criterion(y_predicted, y_train)
        loss.backward()

        optimizer.step()

        total_loss += loss.cpu().item()
        total_rmse += torch.sqrt(loss.detach().cpu()).item()
        total_mae += torch.abs(y_predicted.detach() - y_train).sum().cpu().item()

    # print(total_loss)
    batch_rmse = total_rmse / (len(train_data) / 64)  # divided by number of batches
    batch_mae = total_mae / len(train_data)  # divided by number of samples

    net.eval()
    total_rmse = 0
    total_mae = 0
    with torch.no_grad():
        for X_test, y_test in test_dataloader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            user_ids = X_test[:, 0].long()
            movie_ids = X_test[:, 1].long()
            side_features = X_test[:, 2:]
            y_pred = net(user_ids, movie_ids, side_features)
            total_rmse += torch.sqrt(criterion(y_pred, y_test)).cpu().item()
            total_mae += torch.abs(y_pred - y_test).sum().cpu().item()

    test_rmse = total_rmse / (len(test_data) / 64)
    test_mae = total_mae / len(test_data)

    epochs_bar.set_description(
        f"train_rmse: {batch_rmse:.4f}, train_mae: {batch_mae:.4f}, "
        f"test_rmse: {test_rmse:.4f}, test_mae: {test_mae:.4f}"
    )

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

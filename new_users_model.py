import pandas as pd
from surprise import SVD, Dataset, Reader
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

ratings_df = pd.read_csv("ml-latest-small/ratings.csv")
ratings_df = ratings_df.sort_values(
    by="timestamp", ascending=True
)  # sorted by timestamp so testset is built from most recent users

train_size = 0.8

cut_index = int(len(ratings_df) * train_size)
train_df = ratings_df.iloc[:cut_index]
test_df = ratings_df.iloc[cut_index:]

reader = Reader(rating_scale=((0.5, 5)))
trainset = Dataset.load_from_df(
    df=train_df[["userId", "movieId", "rating"]], reader=reader
).build_full_trainset()

testset = [
    (row.userId, row.movieId, 0)
    for row in test_df[["userId", "movieId"]].itertuples(index=False)
]  # setting rating as 0 (only to fill it, won't be used)

target_ratings = test_df["rating"].values

algo = SVD()

algo.fit(trainset)

predictions = algo.test(
    testset
)  # testset only used to generate estimated values, due to use 0 as "true rating" rmse and mae calculated using sklearn, not surprises
# users are completely new so sth close to global mean is used
predicted_ratings = [prediction.est for prediction in predictions]

global_avg = ratings_df["rating"].mean()

global_average_ratings = [
    global_avg for _ in predictions
]  # filling predicted ratings with global average rating for comparison

movie_avg_dict = train_df.groupby("movieId")["rating"].mean().to_dict()
movie_average_ratings = [
    movie_avg_dict.get(prediction.iid, global_avg) for prediction in predictions
]  # filling predicted ratings with average rating for this movie, if the movie was not rated yet, fill with global average


def calculate_rmse_mae(true_values, estimated_values):
    rmse = root_mean_squared_error(true_values, estimated_values)
    mae = mean_absolute_error(true_values, estimated_values)

    print(f"RMSE of the model for new users: {rmse:.3f}")  # root mean squared error
    print(f"MAE of the model for new users: {mae:.3f}")  # mean absolute error


print("SVD estimated model:")
calculate_rmse_mae(target_ratings, predicted_ratings)
print("Targets filled with global average of all movies:")
calculate_rmse_mae(target_ratings, global_average_ratings)
print("Targets filled with average for each movie:")
calculate_rmse_mae(target_ratings, movie_average_ratings)

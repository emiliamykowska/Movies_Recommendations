import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split, LeaveOneOut
from surprise import accuracy
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


ratings_df = pd.read_csv("ml-latest-small/ratings.csv")
user_movie_matrix = ratings_df.pivot(index="userId", columns="movieId", values="rating")

all_matrix_cells = user_movie_matrix.size
filled_cells = user_movie_matrix.count().sum()
sparse = filled_cells * 100 / all_matrix_cells  # checking sparsity of the matrix
print(f"Percentage of filled cells: {sparse:.2f}")

data = Dataset.load_from_df(
    df=ratings_df[["userId", "movieId", "rating"]], reader=Reader(rating_scale=(0.5, 5))
)

"""Evaluation using default train test split in surprise"""
trainset, testset = train_test_split(data)

algo = SVD()
algo.fit(trainset)

predictions = algo.test(testset)

print(
    f"RMSE of the default train_test_split model: {accuracy.rmse(predictions, False):.3f}"
)  # root mean squared error
print(
    f"MAE of the default train_test_split model: {accuracy.mae(predictions, False):.3f}"
)  # mean absolute error


def round_multiplied(number):
    number = round(
        number * 2
    )  # rounding number multiplied by 2 to get integers for confusion matrix

    return min(10, max(1, number))  # getting number between 1 and 10


predicted = [round_multiplied(prediction.est) for prediction in predictions]
expected = [round_multiplied(prediction.r_ui) for prediction in predictions]


confusion_mx = confusion_matrix(expected, predicted)

axes = sns.heatmap(
    confusion_mx,
    cmap="Blues",
    xticklabels=[x / 2 for x in range(1, 11)],
    yticklabels=[x / 2 for x in range(1, 11)],
)
plt.xlabel("Predicted ratings")
plt.ylabel("Expected ratings")
plt.savefig("conf_matrix.png")


"""Evaluation using LeaveOneOut"""
splits_number = 5
lvo = LeaveOneOut(n_splits=splits_number)
algo = SVD()

all_predictions = []

for trainset, testset in lvo.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)

    all_predictions.extend(predictions)

print(
    f"RMSE of the LeaveOneOut model: {accuracy.rmse(all_predictions, False):.3f}"
)  # root mean squared error
print(
    f"MAE of the default LeaveOneOut model: {accuracy.mae(all_predictions, False):.3f}"
)  # mean absolute error

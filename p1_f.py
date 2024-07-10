from sklearn.ensemble import GradientBoostingClassifier
from src import load_data, SEED
import matplotlib.pyplot as plt

def train_with_depth(depth:int):
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data()
    gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=depth, random_state=SEED)
    gb_clf.fit(train_x, train_y)

    train_accuracy = gb_clf.score(train_x, train_y)
    valid_accuracy = gb_clf.score(valid_x, valid_y)
    test_accuracy = gb_clf.score(test_x, test_y)

    return train_accuracy, valid_accuracy, test_accuracy

train_accuracies = []
valid_accuracies = []
test_accuracies = []

max_depth = 20
for depth in range(1, 20):
    train_accuracy, valid_accuracy, test_accuracy = train_with_depth(depth)
    train_accuracies.append(train_accuracy)
    valid_accuracies.append(valid_accuracy)
    test_accuracies.append(test_accuracy)

plt.figure(dpi=400)
plt.plot(range(1, max_depth), train_accuracies, label='Train')
plt.plot(range(1, max_depth), valid_accuracies, label='Validation')
plt.plot(range(1, max_depth), test_accuracies, label='Test')
plt.xlabel('Max depth')
plt.xticks(range(1, max_depth))
plt.ylabel('Accuracy')
plt.title('Gradient Boosting')
plt.legend()
plt.savefig('figures/P1F_GradientBoosting.png')
plt.show()

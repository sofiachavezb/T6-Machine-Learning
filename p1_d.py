from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from src import load_data, SEED, N_SEEDS_TO_REPEAT
import random
import matplotlib.pyplot as plt
import numpy as np
random.seed(SEED)

def run_experiment_with_seed(seed:int):
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data()

    # Bagging
    bagging_clf = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=seed)
    bagging_clf.fit(train_x, train_y)
    bagging_train_accuracy = accuracy_score(train_y, bagging_clf.predict(train_x))
    bagging_valid_accuracy = accuracy_score(valid_y, bagging_clf.predict(valid_x))
    bagging_test_accuracy = accuracy_score(test_y, bagging_clf.predict(test_x))

    # Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    rf_clf.fit(train_x, train_y)
    rf_train_accuracy = accuracy_score(train_y, rf_clf.predict(train_x))
    rf_valid_accuracy = accuracy_score(valid_y, rf_clf.predict(valid_x))
    rf_test_accuracy = accuracy_score(test_y, rf_clf.predict(test_x))   

    bagging_accuracies = (bagging_train_accuracy, bagging_valid_accuracy, bagging_test_accuracy)
    rf_accuracies = (rf_train_accuracy, rf_valid_accuracy, rf_test_accuracy)

    return bagging_accuracies, rf_accuracies
    
    
bagging_accuracies, rf_accuracies = run_experiment_with_seed(SEED)
bagging_train_accuracy, bagging_valid_accuracy, bagging_test_accuracy = bagging_accuracies
rf_train_accuracy, rf_valid_accuracy, rf_test_accuracy = rf_accuracies

print(f"Bagging train accuracy: {bagging_train_accuracy}")
print(f"Bagging validation accuracy: {bagging_valid_accuracy}")
print(f"Bagging test accuracy: {bagging_test_accuracy}")
print(f"Random Forest train accuracy: {rf_train_accuracy}")
print(f"Random Forest validation accuracy: {rf_valid_accuracy}")
print(f"Random Forest test accuracy: {rf_test_accuracy}")

if abs(bagging_test_accuracy - rf_test_accuracy) < 0.01:
    print(f"The models have similar performance, trying with {N_SEEDS_TO_REPEAT} random seeds")
    seeds = [random.randint(0, 2**32) for _ in range(N_SEEDS_TO_REPEAT)]
    
    bagging_train_accuracies = []
    bagging_valid_accuracies = []
    bagging_test_accuracies = []
    rf_train_accuracies = []
    rf_valid_accuracies = []
    rf_test_accuracies = []

    print("Loading ...")

    for i, seed in enumerate(seeds):
        if i in range(0, N_SEEDS_TO_REPEAT, N_SEEDS_TO_REPEAT//10):
            print(f"    {i}/{N_SEEDS_TO_REPEAT} seeds tested")
        bagging_accuracies, rf_accuracies = run_experiment_with_seed(seed)
        bagging_train_accuracies.append(bagging_accuracies[0])
        bagging_valid_accuracies.append(bagging_accuracies[1])
        bagging_test_accuracies.append(bagging_accuracies[2])
        rf_train_accuracies.append(rf_accuracies[0])
        rf_valid_accuracies.append(rf_accuracies[1])
        rf_test_accuracies.append(rf_accuracies[2])

    print("Done")   

    # plot density of accuracies for train, validation and test
    fig, ax = plt.subplots(1,3, dpi=400, figsize=(9,4))

    ax[0].hist(bagging_train_accuracies, bins=20, alpha=0.5, label='Bagging', color='tab:blue')
    ax[0].hist(rf_train_accuracies, bins=20, alpha=0.5, label='Random Forest', color='tab:orange')
    ax[0].set_xlabel('Accuracy')
    ax[0].set_ylabel('Density')
    ax[0].set_title('Train accuracies')
    ax[0].legend()

    
    ax[1].hist(bagging_valid_accuracies, bins=20, alpha=0.5, label='Bagging', color='tab:blue')
    ax[1].hist(rf_valid_accuracies, bins=20, alpha=0.5, label='Random Forest', color='tab:orange')
    ax[1].set_xlabel('Accuracy')
    ax[1].set_ylabel('Density')
    ax[1].set_title('Validation accuracies')


    ax[2].hist(bagging_test_accuracies, bins=20, alpha=0.5, label='Bagging', color='tab:blue')
    ax[2].hist(rf_test_accuracies, bins=20, alpha=0.5, label='Random Forest', color='tab:orange')
    ax[2].set_xlabel('Accuracy')
    ax[2].set_ylabel('Density')
    ax[2].set_title('Test accuracies')

    plt.tight_layout()
    plt.savefig('figures/P1D_accuracies_density.png')
    plt.show()

    # Print mean and std of accuracies
    print("Bagging")
    print(f"Train accuracy: {np.mean(bagging_train_accuracies)}")
    print(f"Validation accuracy: {np.mean(bagging_valid_accuracies)}")
    print(f"Test accuracy: {np.mean(bagging_test_accuracies)}")
    print(f"Train accuracy std: {np.std(bagging_train_accuracies)}")
    print(f"Validation accuracy std: {np.std(bagging_valid_accuracies)}")
    print(f"Test accuracy std: {np.std(bagging_test_accuracies)}")

    print("Random Forest")
    print(f"Train accuracy: {np.mean(rf_train_accuracies)}")
    print(f"Validation accuracy: {np.mean(rf_valid_accuracies)}")
    print(f"Test accuracy: {np.mean(rf_test_accuracies)}")
    print(f"Train accuracy std: {np.std(rf_train_accuracies)}")
    print(f"Validation accuracy std: {np.std(rf_valid_accuracies)}")
    print(f"Test accuracy std: {np.std(rf_test_accuracies)}")
    
    
    





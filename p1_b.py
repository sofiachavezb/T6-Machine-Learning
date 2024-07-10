from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
from src import load_data, SEED

(train_x, train_y), (val_x, val_y), (test_x, test_y) = load_data(   verbose=True,
                                                                    selected_columns=['perimeter_worst','perimeter_mean'])

trained_tree = DecisionTreeClassifier(random_state=SEED)
trained_tree.fit(train_x, train_y)

if __name__ == "__main__":
    train_accuracy = accuracy_score(train_y, trained_tree.predict(train_x))
    val_accuracy = accuracy_score(val_y, trained_tree.predict(val_x))
    test_accuracy = accuracy_score(test_y, trained_tree.predict(test_x))

    print(f"Train accuracy: {train_accuracy}")
    print(f"Validation accuracy: {val_accuracy}")
    print(f"Test accuracy: {test_accuracy}")
    
    plt.figure(figsize=(17, 9), dpi=400)
    ax = plt.axes()
    plt.suptitle('Default sklearn tree')
    tree.plot_tree(trained_tree, filled=True, ax=ax)
    plt.savefig('figures/P1B.png')
    plt.show()
    plt.close()
    
    print(f"Fitted tree's depth: {trained_tree.tree_.max_depth}")

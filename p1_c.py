from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
from src import load_data, SEED
from p1_b import trained_tree

(train_x, train_y),(val_x, val_y),(test_x, test_y) = load_data( selected_columns=['perimeter_worst','perimeter_mean'],
                                                                verbose=False)

# Get the cost-complexity pruning path
path = trained_tree.cost_complexity_pruning_path(train_x, train_y)
# ccp_alphas: cost-complexity pruning alphas 
ccp_alphas = path.ccp_alphas
ccp_alphas = sorted(ccp_alphas)

# Train a DecisionTreeClassifier for each alpha
decision_trees = []
for ccp_alpha in ccp_alphas:
    trained_tree = DecisionTreeClassifier(random_state=SEED, ccp_alpha=ccp_alpha)
    trained_tree.fit(train_x, train_y)
    decision_trees.append(trained_tree)

# Get the depth and number of nodes for each tree
depths = [clf.tree_.max_depth for clf in decision_trees]
nodes = [clf.tree_.node_count for clf in decision_trees]

# Plot depth and n° nodes vs alpha
fig, ax = plt.subplots(1, 2, figsize=(9, 4), dpi=400)
ax[0].plot(ccp_alphas, depths, marker='o')
ax[0].set_xlabel(r'$\alpha$')
ax[0].set_ylabel('depth')
ax[0].set_title(r'Profundidad de árbol vs $\alpha$')
ax[1].plot(ccp_alphas, nodes, marker='o')
ax[1].set_xlabel(r'$\alpha$')
ax[1].set_ylabel('n° nodes')
ax[1].set_title(r'Número de nodos vs $\alpha$')
plt.savefig('figures/P1C_nodes_and_depth_vs_alpha.png') 
#plt.show()
plt.close()

#Plot accuracies vs alpha
train_accuracies = [accuracy_score(train_y, clf.predict(train_x)) for clf in decision_trees]
val_accuracies = [accuracy_score(val_y, clf.predict(val_x)) for clf in decision_trees]
plt.figure(dpi=400)
plt.plot(ccp_alphas, train_accuracies, marker='o', label='Train')
plt.plot(ccp_alphas, val_accuracies, marker='o', label='Valid')

plt.xlabel(r'$\alpha$')
plt.ylabel('Accuracy')
plt.legend()
plt.title(r'Accuracies vs $\alpha$')
plt.savefig('figures/P1C_accuracies_vs_alpha.png')
#plt.show()
plt.close()

# Choose the best alpha:
# Plot accuracies and nodes vs alpha
# use two y scales in the same plot
# drop the values for the max alpha
ccp_alphas = ccp_alphas[:-1]
train_accuracies = train_accuracies[:-1]
val_accuracies = val_accuracies[:-1]
nodes = nodes[:-1]

max_val_accuracy = max(val_accuracies)
selected_alpha = ccp_alphas[val_accuracies.index(max_val_accuracy)]
print(f'Selected alpha value:{selected_alpha}')

fig, ax1 = plt.subplots(dpi=400)
ax1.set_xlabel(r'$\alpha$')
ax1.set_ylabel('Accuracy', color='tab:blue')
line_1, = ax1.plot(ccp_alphas, train_accuracies, marker='o', label='Train', color='tab:blue')
line_2, = ax1.plot(ccp_alphas, val_accuracies, marker='o', label='Valid', color='tab:purple')
ax1.tick_params(axis='y', labelcolor='tab:blue')
line_3 = ax1.axvline(selected_alpha, color='tab:green', linestyle=(0, (4, 12)), label=r'selected $\alpha$')
line_4 = ax1.axhline(max_val_accuracy, color='black', linestyle=(0, (4, 12)), label='max valid accuracy')

ax2 = ax1.twinx()
ax2.set_ylabel('Número de nodos', color='tab:orange')
line_5, = ax2.plot(ccp_alphas, nodes, marker='o', label='N° of nodes', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

# Unificar las leyendas
lines = [line_1, line_2, line_5,line_3, line_4]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper right')

plt.title(r'Accuracies and nodes vs $\alpha$')
plt.savefig('figures/P1C_accuracies_and_nodes_vs_alpha.png')
#plt.show()
plt.close()

# Train the final tree
final_tree = DecisionTreeClassifier(random_state=SEED, ccp_alpha=selected_alpha)
final_tree.fit(train_x, train_y)

# Plot the final tree
plt.figure(figsize=(5, 2), dpi=400)
ax = plt.axes()
plt.suptitle(r'$\text{Árbol para } \alpha \text{ seleccionado}$')
tree.plot_tree(final_tree, filled=True, ax=ax)
plt.savefig('figures/P1C_final_tree.png')
#plt.show()

# Print the final accuracies
print('Selected alpha tree results')
train_accuracy = accuracy_score(train_y, final_tree.predict(train_x))
val_accuracy = accuracy_score(val_y, final_tree.predict(val_x))
test_accuracy = accuracy_score(test_y, final_tree.predict(test_x))
print(f"Train accuracy: {train_accuracy}")
print(f"Validation accuracy: {val_accuracy}")
print(f"Test accuracy: {test_accuracy}")
print(f"Depth: {final_tree.tree_.max_depth}")

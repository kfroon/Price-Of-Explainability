from imports import *

#Read in data
path = ""
data = pd.read_csv(path)
target_variable = 'Classification'

#Rename predicted variable to "Classification" 
data['Classification'] = data['Classification'].map({1:0, 2:1})

#preprocessing function 
def preprocess(data):
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1:]#.squeeze()
    feature_names = X.columns
    scaler = StandardScaler()
    X = scaler.fit_transform(data.drop(columns = [target_variable], axis = 1))
    X = pd.DataFrame(X, columns = feature_names)
    data = pd.concat([X,y] ,axis = 1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 390, test_size=0.3)
    return x_train, x_test, y_train, y_test, feature_names

#Split into train and test set
x_train, x_test, y_train, y_test, feature_names = preprocess(data)
train_data = pd.concat([x_train, y_train], axis = 1)
test_data = pd.concat([x_test, y_test], axis = 1)


#Gini index function 
def calculate_gini(y):
    """Calculate the Gini index for a given array of labels."""
    p = y.value_counts() / len(y)
    gini = 1 - np.sum(p ** 2)
    return gini
    
def gini_index(decision_function, y):
    """Calculate the Gini index for a given decision function and labels."""
    indices_less_zero = np.where(decision_function < 0)[0]
    indices_greater_zero = np.where(decision_function > 0)[0]

    g1 = y.iloc[indices_less_zero]
    g2 = y.iloc[indices_greater_zero]

    gini1 = calculate_gini(g1)
    gini2 = calculate_gini(g2)

    n = len(y)
    gini_score = (len(g1) / n) * gini1 + (len(g2) / n) * gini2

    return gini_score


def test_split(decision_function, data):
    """Split the data into left and right data frames based on the decision function."""
    indices_less_zero = np.where(decision_function < 0)[0]
    indices_greater_zero = np.where(decision_function > 0)[0]

    left_data = data.iloc[indices_less_zero]
    right_data = data.iloc[indices_greater_zero]
    
    return left_data, right_data

#Split the data using the given feature set
def get_split(data):
    """Find the best Gini split for the given data."""
    best_gini = float('inf')
    best_split = None

    features = data.iloc[:, :-1]
    target = data.iloc[:, -1]

    #Generate all possible combinations of feature pairs, stipulate number of features in set 
    feature_pairs = list(combinations(features.columns, 2))
    #decision_functions_array = []

    for pair in feature_pairs:
  	# Select the pair of features
        feature1, feature2 = pair
        selected_features = features[[feature1, feature2]]
        #print(selected_features)

        # Train the SVM
        svm = SVC(kernel='linear', C = 100 )
        svm.fit(selected_features, target)

        # Use the decision function to split the data
        decision_function = svm.decision_function(selected_features)
    
        gini = gini_index(decision_function, target)
 
	#This must be adjusted when more featured are added 
        if gini < best_gini:
            best_gini = gini
            best_df = decision_function
            best_svm = svm
            best_split = {'feat1': selected_features.columns[0], 
                          'feat2': selected_features.columns[1],
                          'x1': np.around((best_svm.coef_[0][0]), 5), 
                          'x2': np.around((best_svm.coef_[0][1]), 5),
                          'intercept': np.around(best_svm.intercept_[0], 5)}
            rule = f"{best_split['x1']}*{best_split['feat1']} + {best_split['x2']}*{best_split['feat2']} + {best_split['intercept']} < 0"
            #f"{np.around((best_svm.coef_[0][0]), 5)}*{selected_features.columns[0]} + {np.around((best_svm.coef_[0][1]), 5)}*{selected_features.columns[1]} + {np.around(best_svm.intercept_[0], 5)} < 0 "
        
        groups = test_split(best_df, data)

    return {'best_split':best_split, 'split_rule': rule, 'best_df':best_df, 'best_svm':best_svm, 'groups':groups}


# Create a terminal node value
def to_terminal(subset):
    outcomes = list((subset.iloc[:,-1]))
    return max(set(outcomes), key=outcomes.count)

#Rules on when to split the data
def split(node, max_depth, min_samples, depth):
    left, right = node['groups']
    del(node['groups'])

    #Check for no-split
    if left.empty or right.empty:
        node['left']  = node['right'] = to_terminal(pd.concat([left, right]))
        return
    
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    
    # process left child
    if len(left) <= min_samples:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_samples, depth+1)

     # process right child
    if len(right) <= min_samples:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_samples, depth+1)

# Build tree 
def build_tree(train, max_depth, min_size):
   root = get_split(train)
   split(root, max_depth, min_size, 1)
   return root

#Function to make predications, must be adjusted with increase of features 
def predict(node, row):
    feat1 = node['best_split']['feat1']
    feat2 = node['best_split']['feat2']
    x1 = node['best_split']['x1']
    x2 = node['best_split']['x2']
    intercept = node['best_split']['intercept']

    score = x1 * row[feat1] + x2 * row[feat2] + intercept 
    if score < 0:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# Print tree
def print_tree(node, depth=1):
 if isinstance(node, dict):
    print(depth, depth*'  ', f'Split: {node["split_rule"]}', f" Points: {len(node['best_df'])}")
    print_tree(node['left'], depth+1)
    print_tree(node['right'], depth+1)
 else:
   print('%s class: [%s]' %( (depth*'  ', node)))

#Create and instance of the tree and test 

#Stipulate depth and minimum number of features to required to split
#Create and instance of the tree
tree = build_tree(train_data, depth, min_features_to_split)
print_tree(tree)

y_true = list(test_data.Classification)
predictions_saved = []
for row in range(len(test_data)):
    #print(test_data.iloc[row])
    p = predict(tree, test_data.iloc[row])
    predictions_saved.append(p)
y_pred = list(predictions_saved)
from sklearn.metrics import accuracy_score
print(f"accuracy: {accuracy_score(y_true, y_pred)}") 
print(f"f1_score: {f1_score(y_true, y_pred)}")



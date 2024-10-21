import argparse
import numpy as np
import pandas as pd

"""
To run script, use the command: 

python decision_tree.py <train input> <test
input> <max depth> <train out> <test out> <metrics out> <print out>

For example: python decision_tree.py heart_train.tsv heart_test.tsv 2 \
heart_2_train.txt heart_2_test.txt heart_2_metrics.txt
\ heart_2_print.txt

"""

class Node:

    def __init__(self, depth: int = 0, parent_attr: str = None, parent_value: int = None):
        self.left = None
        self.right = None
        self.parent_attr = parent_attr
        self.attr = None
        self.parent_value = parent_value
        self.vote = None
        self.depth = depth
        self.zeros = None
        self.ones = None

    # Create a tree
    def train(self, data, max_depth):
        # Base case
        if (
            any(num <= 0 for num in self.calculate_mutual_information(data)) or
            self.depth >= max_depth or
            len(data.columns) == 1 or
            len(data.iloc[:, -1].unique()) == 1
        ):
            self.vote = self.majority_vote(data)
        
        # Recursively train data
        else:
            split = data.columns[np.argmax(self.calculate_mutual_information(data))]
            self.attr = split
            # Left Node
            self.left = Node(depth=self.depth+1, parent_attr=split, parent_value=0)
            left_data = data[data[split] == 0]
            left_data = left_data.drop(split, axis=1)
            self.left.train(left_data, max_depth)
            # Right Node
            self.right = Node(depth=self.depth+1, parent_attr=split, parent_value=1)
            right_data = data[data[split] == 1]
            right_data = right_data.drop(split, axis=1)
            self.right.train(right_data, max_depth)
    
    def calculate_entropy(self, data):
        labels = data.iloc[:, -1]
        total_labels = len(labels)
        self.ones = np.sum(labels)
        self.zeros = total_labels - self.ones
        if self.ones == 0 or self.zeros == 0:
            entropy = 0
        else:
            entropy = -((self.ones/total_labels) * np.log2(self.ones/total_labels)) \
                - ((self.zeros/total_labels) * np.log2(self.zeros/total_labels))
            
        return entropy
    
    def calculate_mutual_information(self, data):
        mutual_informations = []
        label_name = data.columns[-1]
        label_count= len(data)
        entropy = self.calculate_entropy(data)
        for each_attr in data.columns[:-1]:
            conditional = []

            for i in [0,1]:
                condition = data[data[each_attr] == i]
                total_condition = len(condition)
                label_cond_one = len(condition[condition[label_name] == 1])
                label_cond_zero = len(condition[condition[label_name] == 0])

                # Calculate H(Y: X = x)
                if total_condition > 0:
                    if label_cond_one == 0 :
                        H_cond = - ((label_cond_zero/total_condition) * np.log2(label_cond_zero/total_condition))
                    elif label_cond_zero == 0:
                        H_cond = -((label_cond_one/total_condition) * np.log2(label_cond_one/total_condition))
                    else:
                        H_cond = -((label_cond_one/total_condition) * np.log2(label_cond_one/total_condition)) \
                        -((label_cond_zero/total_condition) * np.log2(label_cond_zero/total_condition))
                
                else: 
                    H_cond = 0

                conditional.append(H_cond)

            # Calculate H(Y:X)
            zeros = len(data[data[each_attr] == 0])
            ones = len(data[data[each_attr] == 1])
            H = ((zeros/label_count) * conditional[0]) + ((ones/label_count) * conditional[1])

            # Calulate I(Y;X)
            mutual_info = entropy - H
            mutual_informations.append(mutual_info)

        return mutual_informations
    
    def majority_vote(self, data):
        labels = data.iloc[:, -1]
        half = len(labels) / 2
        trues = np.sum(labels)
        if trues >= half:
            return 1
        else:
            return 0

def predict(tree: Node, test_input):
    """
    Takes in data frame and makes prediction of labels
    """
    predictions = []
    def predict_one(tree, single_input):
        if tree.vote is not None:
            return tree.vote
        else:
            if single_input[tree.attr] == 0:
                return predict_one(tree.left, single_input)
            elif single_input[tree.attr] == 1:
                return predict_one(tree.right, single_input)

    for idx, row in test_input.iterrows():
        predicted = predict_one(tree, row)
        predictions.append(predicted)
    
    return predictions
        

def load_data(filepath):
    """
    Loads data frame
    """
    return pd.read_csv(filepath, sep="\t")

def print_tree(root_node: Node):
    """
    Prints a trained Decision Tree Binary Classifier.
    """
    def print_node_info(node: Node):
        """
        Prints training-set majority vote of the given `node`.
        """
        out_str = '| ' * (node.depth) + f'{node.parent_attr} = {node.parent_value}: [{node.zeros} 0/{node.ones} 1]'
        
        if node.left is None and node.right is None and node.depth > 0:
            print(out_str)
        else:
            if node.depth > 0:
                print(out_str)
            print_node_info(node.left)
            print_node_info(node.right)

    # Root-node information
    print(f"[{root_node.zeros} 0/{root_node.ones} 1]")
    
    # Recursively traverse tree and print node information
    print_node_info(root_node)

def calculate_error(predicted, real):
    """
    Calculates the error between the predicted labels and real labels
    """
    real_labels = list(real.iloc[:, -1])
    total = len(real_labels)
    error_count = 0
    for idx, value in enumerate(predicted):
        if value != real_labels[idx]:
            error_count += 1
        else: continue
    error = error_count/total

    return error

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the test input .tsv file')
    parser.add_argument("max_depth", type=int, 
                        help='maximum depth to which the tree should be built')
    parser.add_argument("train_out", type=str, 
                        help='path to output .txt file to which the feature extractions on the training data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .txt file to which the feature extractions on the test data should be written')
    parser.add_argument("metrics_out", type=str, 
                        help='path of the output .txt file to which metrics such as train and test error should be written')
    parser.add_argument("print_out", type=str,
                        help='path of the output .txt file to which the printed tree should be written')
    args = parser.parse_args()
    
    # Train data and predict labels
    train_data = load_data(args.train_input)
    tree = Node()
    tree.train(train_data, args.max_depth)
    predicted_train_labels = predict(tree, train_data)
    with open(args.train_out, "w") as file:
        for each in predicted_train_labels:
            file.write(str(each) + "\n")

    # Test data and predict labels
    test_data = load_data(args.test_input)
    predicted_test_labels = predict(tree, test_data)
    with open(args.test_out, "w") as file:
        for each in predicted_test_labels:
            file.write(str(each) + "\n")
    
    # Find error
    train_error = calculate_error(predicted_train_labels, train_data)
    test_error = calculate_error(predicted_test_labels, test_data)
    with open(args.metrics_out, 'w') as outfile:
        outfile.write("error(train):" + str(train_error) + "\n")
        outfile.write("error(test):" + str(test_error) + "\n")

    # Print trained tree
    print_tree(tree)

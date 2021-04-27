import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse as ap

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def parse_arguments():
    parser = ap.ArgumentParser(description="Pass your arguments here")
    parser.add_argument('-t', '--train_file',
                        type=str,
                        required=True,
                        help='csv file with data')
    parser.add_argument('-s', '--train_test_split',
                        type=float,
                        default=0.2,
                        help='How many of specimens will be used for tests (default and min 0.2  while max 0.8)')

    parser.add_argument('-c', '--classification_column',
                        default=-1,
                        type=str,
                        help='Name of column in dataset with classification data')
    parser.add_argument('--max_depth',
                        type=int,
                        default=5,
                        help='Maximum depth of tree, dafault value is 5 and this is a ineteger')
    parser.add_argument('--acceptable_impurity',
                        type=float,
                        default=0.0,
                        help='Level of impurity at which we no longer split nodes, default value 0')
    args_result = parser.parse_args()
    if args_result.train_test_split < 0.2 or args_result.train_test_split > 0.8:
        raise ap.ArgumentError("-s must be more or equal 0.2 and less or equal 0.8")

    return args_result


if __name__ == '__main__':
    args = parse_arguments()
    dataset = pd.read_csv(args.train_file)
    target = dataset[dataset.columns[args.classification_column]]
    data = dataset[dataset.columns[:7]]
    data_train, data_test, target_train, target_test = \
        train_test_split(data, target, test_size=args.train_test_split)

    classifier = DecisionTreeClassifier(max_depth=args.max_depth, min_impurity_decrease=args.acceptable_impurity)
    classifier.fit(data_train, target_train)

    class_nms = target.unique().astype(str)

    export_graphviz(classifier, 'classifier.dot', feature_names=dataset.columns[:7], class_names=class_nms)
    os.system('dot -Tpng classifier.dot -o classifier.png')

    img = cv2.imread('classifier.png')
    plt.figure(figsize=(13, 13))
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(img)
    # plt.show()

    answers = classifier.predict(data_test.loc[:])
    print(40 * '--')
    print(answers)
    print(40 * '--')
    print('Answers:')
    print(answers)

    print('Values:')
    print(class_nms[answers])

    classifier = RandomForestClassifier(min_impurity_decrease=args.acceptable_impurity,
                                        max_depth=args.max_depth, n_estimators=9)
    classifier.fit(data_train, target_train)

    print(len(classifier.estimators_))

    for tree_id, tree in enumerate(classifier.estimators_):
        export_graphviz(tree, f'pngs/tree{tree_id:02d}.dot', feature_names=dataset.columns[:7], class_names=class_nms)
        os.system(f'dot -Tpng pngs/tree{tree_id:02d}.dot -o pngs/tree{tree_id:02d}.png')

    fig = plt.figure(figsize=(18, 18))
    fig.tight_layout(pad=0.8)

    for tree_id, tree in enumerate(classifier.estimators_):
        ax = fig.add_subplot(3, 3, tree_id + 1)
        ax.title.set_text(f'tree{tree_id:02d}')
        img = cv2.imread(f'pngs/tree{tree_id:02d}.png')
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    sample = data_test.sample()
    print(sample)
    print(target_test[sample.index])
    print(classifier.predict(sample))

    print('Stats')
    responses = classifier.predict(data_test)

    print(confusion_matrix(target_test, responses))
    print(classification_report(target_test, responses))
    print(accuracy_score(target_test, responses))
    plt.show()

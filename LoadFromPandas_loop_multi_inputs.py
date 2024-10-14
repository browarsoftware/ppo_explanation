# Generate CSV for Algorithm 2
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

def evaluate(file_path1, file_path2, file_path3):
    df = pd.read_csv(file_path1, sep=',')
    my_len = len(df.columns.to_list())
    if my_len > 35:
        [cmfb, foo, foo] = evaluate3x3(file_path1)
        [foo, cmlr, foo] = evaluate3x3(file_path2)
        [foo, foo, cmgj] = evaluate3x3(file_path3)
        return [cmfb, cmlr, cmgj]
    else:
        [cmfb, foo, foo] = evaluate2x2(file_path1)
        [foo, cmlr, foo] = evaluate2x2(file_path2)
        [foo, foo, cmgj] = evaluate2x2(file_path3)
        return [cmfb, cmlr, cmgj]

def evaluate3x3(file_path):
    df = pd.read_csv(file_path, sep=',')
    feature_names = df.columns[1:37].to_list()
    target_names = df.columns[39]
    X = df.iloc[:, 1:37].to_numpy()
    y = df.iloc[:, 39].to_numpy().astype('int32')
    target_names = ['Ground', 'Jump']

    # Fit the classifier with default hyper-parameters
    # clf = DecisionTreeClassifier(random_state=1234, max_depth=4)

    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree
    conf_matrix = np.zeros([2, 2])
    X_full = np.copy(X)
    y_full = np.copy(y)
    for id_to_delete in range(y_full.shape[0]):
        X = np.delete(X_full, (id_to_delete), axis=0)
        y = np.delete(y_full, (id_to_delete), axis=0)
        clf = DecisionTreeClassifier(random_state=1234)  # , min_samples_split=30, min_samples_leaf=30)
        model = clf.fit(X, y)
        y_pred = clf.predict(X_full[id_to_delete:(id_to_delete + 1), :])
        conf_matrix[y_pred, y_full[id_to_delete]] += 1
    cmgj = conf_matrix

    feature_names = df.columns[1:37].to_list()
    target_names = df.columns[37]
    X = df.iloc[:, 1:37].to_numpy()
    y = df.iloc[:, 37].to_numpy().astype('int32')
    # target_names = ['Ground','Jump']

    conf_matrix = np.zeros([3, 3])
    X_full = np.copy(X)
    y_full = np.copy(y)
    for id_to_delete in range(y_full.shape[0]):
        X = np.delete(X_full, (id_to_delete), axis=0)
        y = np.delete(y_full, (id_to_delete), axis=0)
        clf = DecisionTreeClassifier(random_state=1234)  # , min_samples_split=30, min_samples_leaf=30)
        model = clf.fit(X, y)
        y_pred = clf.predict(X_full[id_to_delete:(id_to_delete + 1), :])
        conf_matrix[y_pred, y_full[id_to_delete]] += 1

    cmfb = conf_matrix

    feature_names = df.columns[1:37].to_list()
    target_names = df.columns[38]
    X = df.iloc[:, 1:37].to_numpy()
    y = df.iloc[:, 38].to_numpy().astype('int32')
    # target_names = ['Ground','Jump']

    conf_matrix = np.zeros([3, 3])
    X_full = np.copy(X)
    y_full = np.copy(y)
    for id_to_delete in range(y_full.shape[0]):
        X = np.delete(X_full, (id_to_delete), axis=0)
        y = np.delete(y_full, (id_to_delete), axis=0)
        clf = DecisionTreeClassifier(random_state=1234)  # , min_samples_split=30, min_samples_leaf=30)
        model = clf.fit(X, y)
        y_pred = clf.predict(X_full[id_to_delete:(id_to_delete + 1), :])
        conf_matrix[y_pred, y_full[id_to_delete]] += 1

    cmlr = conf_matrix
    return [cmfb, cmlr, cmgj]


def evaluate2x2(file_path):
    df = pd.read_csv(file_path, sep=',')
    feature_names = df.columns[1:17].to_list()
    target_names = df.columns[19]
    X = df.iloc[:, 1:17].to_numpy()
    y = df.iloc[:, 19].to_numpy().astype('int32')
    target_names = ['Ground', 'Jump']

    # Fit the classifier with default hyper-parameters
    # clf = DecisionTreeClassifier(random_state=1234, max_depth=4)

    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree
    conf_matrix = np.zeros([2, 2])
    X_full = np.copy(X)
    y_full = np.copy(y)
    for id_to_delete in range(y_full.shape[0]):
        X = np.delete(X_full, (id_to_delete), axis=0)
        y = np.delete(y_full, (id_to_delete), axis=0)
        clf = DecisionTreeClassifier(random_state=1234)  # , min_samples_split=30, min_samples_leaf=30)
        model = clf.fit(X, y)
        y_pred = clf.predict(X_full[id_to_delete:(id_to_delete + 1), :])
        conf_matrix[y_pred, y_full[id_to_delete]] += 1
    cmgj = conf_matrix

    feature_names = df.columns[1:17].to_list()
    target_names = df.columns[17]
    X = df.iloc[:, 1:17].to_numpy()
    y = df.iloc[:, 17].to_numpy().astype('int32')
    # target_names = ['Ground','Jump']

    conf_matrix = np.zeros([3, 3])
    X_full = np.copy(X)
    y_full = np.copy(y)
    for id_to_delete in range(y_full.shape[0]):
        X = np.delete(X_full, (id_to_delete), axis=0)
        y = np.delete(y_full, (id_to_delete), axis=0)
        clf = DecisionTreeClassifier(random_state=1234)  # , min_samples_split=30, min_samples_leaf=30)
        model = clf.fit(X, y)
        y_pred = clf.predict(X_full[id_to_delete:(id_to_delete + 1), :])
        conf_matrix[y_pred, y_full[id_to_delete]] += 1

    cmfb = conf_matrix

    feature_names = df.columns[1:17].to_list()
    target_names = df.columns[18]
    X = df.iloc[:, 1:17].to_numpy()
    y = df.iloc[:, 18].to_numpy().astype('int32')
    # target_names = ['Ground','Jump']

    conf_matrix = np.zeros([3, 3])
    X_full = np.copy(X)
    y_full = np.copy(y)
    for id_to_delete in range(y_full.shape[0]):
        X = np.delete(X_full, (id_to_delete), axis=0)
        y = np.delete(y_full, (id_to_delete), axis=0)
        clf = DecisionTreeClassifier(random_state=1234)  # , min_samples_split=30, min_samples_leaf=30)
        model = clf.fit(X, y)
        y_pred = clf.predict(X_full[id_to_delete:(id_to_delete + 1), :])
        conf_matrix[y_pred, y_full[id_to_delete]] += 1

    cmlr = conf_matrix
    return [cmfb, cmlr, cmgj]


def eval_eval(file_path1, file_path2, file_path3):
    [cmfb, cmlr, cmgj] = evaluate(file_path1, file_path2, file_path3)
    ss = cmfbss = cmlrss = cmgjss = 0
    for a in range(3):
        ss += cmfb[a, a]
        cmfbss += cmfb[a, a]
        ss += cmlr[a, a]
        cmlrss += cmlr[a, a]
    for a in range(2):
        ss += cmgj[a, a]
        cmgjss += cmgj[a, a]
    return [cmfb, cmlr, cmgj, cmfbss, cmlrss, cmgjss, ss]

"""
x = 48
file_path = 'c:/data/AgentData/Nature/' + str(x) + 'x' + str(x) + '_hidden16_power=1_threshold=0_1000_naive.txt'
print(file_path)
eval_eval(file_path)
"""

file_path = 'c:/data/AgentData/Simple/192x192_hidden16_power=3_threshold=0.5_1000.txt'
file_path = 'c:/data/AgentData/Simple/192x192_hidden16_power=3_threshold=0.5_1000_naive.txt'
file_path = 'c:/data/AgentData/Simple/192x192_hidden16_power=1_threshold=0_1000.txt'

file_path = 'c:/data/AgentData/Simple/64x64_hidden16_power=1_threshold=0.15_1000_naive.txt'


file_name = 'c:/data/AgentData/Simple/evaluate_all.txt'
with open(file_name, "a") as myfile:
    myfile.write("x,power,threshold,cm1,cm2,cm3,acc\n")
for x in [48, 56, 64, 96, 128, 192]:
    file_path = 'c:/data/AgentData/Simple/' + str(x) + 'x' + str(x) + '_hidden16_power=1_threshold=0_1000_naive.txt'
    #file_path = 'c:/data/AgentData/Nature/' + str(x) + 'x' + str(x) + '_hidden16_power=1_threshold=0_1000_naive.txtfb.txt'
    print(file_path)
    [cm1, cm2, cm3, cm1ss, cm2ss, cm3ss, ss] = eval_eval(file_path, file_path, file_path)
    with open(file_name, "a") as myfile:
        myfile.write(str(x) + ", -1, -1," + str(cm1ss) + "," + str(cm2ss) + "," + str(cm3ss) + "," + str(ss) + "\n")
    for power in [1,2,3,5]:
        for threshold in [0,0.1,0.2,0.3,0.4,0.5]:
            file_path = 'c:/data/AgentData/Simple/' + str(x) + 'x' + str(x) + '_hidden16_power=' + str(
                power) + '_threshold=' + str(threshold) + '_1000.txtfb.txt'
            file_path1 = 'c:/data/AgentData/Simple/' + str(x) + 'x' + str(x) + '_hidden16_power=' + str(
                power) + '_threshold=' + str(threshold) + '_1000.txtlr.txt'
            file_path2 = 'c:/data/AgentData/Simple/' + str(x) + 'x' + str(x) + '_hidden16_power=' + str(
                power) + '_threshold=' + str(threshold) + '_1000.txtgj.txt'
            print(file_path)
            [cm1, cm2, cm3, cm1ss, cm2ss, cm3ss, ss] = eval_eval(file_path, file_path1, file_path2)
            with open(file_name, "a") as myfile:
                myfile.write(str(x) + "," + str(power) + "," + str(threshold) + "," + str(cm1ss) + "," + str(cm2ss) + "," + str(cm3ss) + "," + str(ss) + "\n")


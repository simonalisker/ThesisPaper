# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Simona Lisker
20-Jan-23
@how to:
This file should be executed as stand alone
It uses FiFTy npz files as input
@purpose:
Generate statistical features based on FiFTy dataset and save them in files

model_100_train
model_100_train
model_100_train
Statistical features are:
# Hamming weight,
# Low ASCII frequency,
# Medium ASCII frequency,
# High ASCII frequency
# Shannon entropy
# Length of longest streak of repeating bytes,
# Standard deviation
# Arithmetic mean
# Geometric mean
# Harmonic mean
-------------------------------------------------------------------------------
    Variables:
        path =  fragments location on which the model will be built
        additional code - XGBoost model and prediction using statistical features
@author: Simona Lisker
@inistitute: HIT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
import os
from nltk.tokenize import word_tokenize
import gensim
import time

from sklearn import metrics

import fragment_creation
import numpy as np
from enum import Enum
import collections
import statistics
from scipy.stats import gmean
from scipy.stats import entropy
from src.LoadData import load_dataset, train_base_path
import math
import re
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, make_scorer
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier
import seaborn as sns
from sklearn.metrics import precision_recall_curve
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import f1_score


def create_categories(data, labels):
    n_elements = len(data)
    print('Elements in dataset:', n_elements)
    categories = sorted(list(set(labels)))  # set will return the unique different entries
    n_categories = len(categories)
    # print("{} categories found:".format(n_categories))
    # for category in categories:
    # print(category)
    return n_categories, categories


def indicize_labels(data, labels):
    """Transforms string labels into indices"""
    n_categories, categories = create_categories(data, labels)
    indices = []
    for j in range(len(labels)):
        for i in range(n_categories):
            if labels[j] == categories[i]:
                indices.append(i)
    return indices
#############################################################
# Create statistical features file int stat_features directory
# for train, test, val based on file fragments
# Hamming weight,
# Low ASCII frequency,
# Medium ASCII frequency,
# High ASCII frequency
# Shannon entropy
# Length of longest streak of repeating bytes,
# Standard deviation
# Arithmetic mean
# Geometric mean
# Harmonic mean
############################################
def create_statistical_features(X_data, save_to_path, what_data_str, labels):
    vocab = []
    count = 0
    f_model = open(save_to_path + 'model_' + str(100) + what_data_str, 'a')
    block_size = len(X_data[0])
    f_model.write(f"hamming_weight, Shannon2, lowAsciiAv, meduimAsciiAv, highAsciiAv,"
                  f"block_ar_mean, block_std, block_geo_mean, block_harmonic_mean, longestStreakRepBytes, block_sum\n")
    print(len(X_data))
    for file in X_data:#range(len(X_data)):
        count = count + 1

        hamming = 0
        lowAscii = 0
        highAscii = 0
        meduimAscii = 0
        block_sum = 0
        my_block_data = file#X_data[file]
        counters = {byte: 0 for byte in range(2 ** 8)}
        Shannon2 = 0
        block_geo_mean = 0
        block_harmonic_mean = 0

        for j in file:#range(len(X_data[file])):
            my_byte = j #X_data[file, j]
            hamming = hamming + bin(my_byte).count('1')
            counters[my_byte] = counters[my_byte] + 1
            # print(hamming, my_byte, bin(my_byte), bin(my_byte).count('1'))
            if (my_byte < 32):
                lowAscii = lowAscii + 1
            elif (my_byte > 127):
                highAscii = highAscii + 1
            else:
                meduimAscii = meduimAscii + 1
        byte_values = []

        for char in my_block_data:
            # Append the byte value to the byte_values list
            byte_values.append(char)

        # Calculate the sum of byte values
        block_sum = sum(byte_values)
        #block_sum = my_block_data.sum()
        lowAsciiAv = lowAscii / block_size
        meduimAsciiAv = meduimAscii / block_size
        highAsciiAv = highAscii / block_size

        hamming_weight = hamming / (block_size * 8)
        # print(counters)
        if (block_size != 0):
            probabilities = [counter / block_size for counter in
                             counters.values()]  # calculate probabilities for each byte
            Shannon2 = -sum(
                probability * math.log2(probability) for probability in probabilities if probability > 0)  # final sum

        else:
            print("block_size iz zero, Shannon entorpy cannot be calculated", file, block_size, y_train[file])

        if (block_sum > 0):
            # remove zeros from array
            x_new = [i for i in my_block_data if i != 0]
            block_geo_mean = gmean(x_new)
            block_harmonic_mean = statistics.harmonic_mean(x_new)
        else:
            print("all file is 0!", file, labels[file])

        # print(count, y_train[file], block_sum)
        longestStreakRepBytes = max(re.split(br'((.)\2*)', np.array(my_block_data)), key=len)
        block_ar_mean = block_sum / block_size
        block_std = np.std(block_sum)#X_data[file])

        f_model.write(f"{hamming_weight},{Shannon2},{lowAsciiAv},{meduimAsciiAv},{highAsciiAv},"
                      f"{block_ar_mean},{block_std},{block_geo_mean},{block_harmonic_mean}, {len(longestStreakRepBytes)}, {block_sum}\n")

        if (count % 100000) == 0:
            print("Fragment", count, "is processed.\n************************")
            # model = gensim.models.Word2Vec(vocab, min_count=1)
    f_model.close()
    return count


def model_generation_statistical_features(X_train, y_train, what_data_str):
    # path = fragment_creation.absolute_path + '/512_4/dump'

    save_to_path = fragment_creation.absolute_path + '/' + fragment_creation.data_set_type + "/stat_features/"

    os.makedirs(save_to_path, exist_ok=True)
    start = time.time()
    # sizes = [20,50,100,150,200]
    sizes = fragment_creation.sizes  # sizes = list(np.arange(5, 105, 5)) # vector length
    # sizes = [100]
    no_vocab_cnt = 0  # finds if any model type is missed or not
    f = open('model_gen_time_stat.txt', 'a')
    f.write("vector_size" + "\t" + "model_gen_time" + "\n")
    count = 0
    for i in range(len(sizes)):
        s = time.time()
        size = sizes[i]

        count = count + create_statistical_features(X_train, save_to_path, what_data_str, y_train)
        e = time.time()
        el = e - s
        f.write(str(size) + "\t" + str(el) + "\n")

        print("Done for vector length: ", str(size), count)
        end = time.time()
        print("Voila! finished building statistical features vectors for fragments!\n")
        if (no_vocab_cnt == 0):
            print("No missed vocabulary\n")
        else:
            print(no_vocab_cnt, " missed vocabualry, please check!\n")
        print("Time elapsed:", format(round((end - start) / 3600, 4)), "hours \nTotal files processed:", count)


def count_zero_files(X_data, what_data_str, labels):
    vocab = []
    count = 0
    save_to_path = fragment_creation.absolute_path + '/' + fragment_creation.data_set_type + "/stat_features/"

    os.makedirs(save_to_path, exist_ok=True)

    f_model = open(save_to_path + 'count_zeros_' + str(100) + what_data_str, 'a')
    block_size = len(X_data[0])
    print(len(X_data))
    counters = {byte: 0 for byte in range(75)}
    count_labels = {byte: 0 for byte in range(75)}
    for file in range(len(X_data)):
        count = count + 1
        my_block_data = X_data[file]
        block_sum = my_block_data.sum()
        count_labels[labels[file]] = count_labels[labels[file]] + 1

        # print(counters)
        if (block_sum == 0):
            counters[labels[file]] = counters[labels[file]] + 1

        if (count % 100000) == 0:
            print("Fragment", count, "is processed.\n************************")
            # model = gensim.models.Word2Vec(vocab, min_count=1)

    f_model.writelines(["label, zeros, total\n"])
    for i in counters:
        f_model.writelines(f"{i}, {counters[i]}, {count_labels[i]}\n")

    f_model.close()
    return count


def train_random_forest_model(features_train, train_labels, features_test, test_labels):
    # Fit a random forest to the training data, using 100 trees
    forest = RandomForestClassifier(n_estimators=100)

    print("Fitting a random forest to labeled training data...")
    forest = forest.fit(features_train, train_labels)

    joblib.dump(forest, train_base_path + "RF_compressed.joblib", compress=3)  # compression is ON!
    print(
        f"Compressed Random Forest: {np.round(os.path.getsize(train_base_path + 'RF_compressed.joblib') / 1024 / 1024, 2)} MB")
    print("Predicting labels for test data..")
    result = forest.predict(features_test)
    # Print the ROC curve, classification report and confusion matrix
    print('\nClassification Report:')
    print(classification_report(test_labels, result))
    print('\nConfusion Matrix:')
    print(confusion_matrix(test_labels, result))

    probs = forest.predict_proba(features_test)[:, 1]

    fpr, tpr, _ = roc_curve(test_labels, probs)
    auc = roc_auc_score(test_labels, probs)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='AUC {:.3f}'.format(auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()


def train_xgb_model(train, train_labels, test, test_labels):
    # Generate sample data
    # X, y = make_multilabel_classification(n_samples=1000, n_features=20, n_classes=75, random_state=1)

    # Define the model
    xgb_model = xgb.XGBClassifier(random_state=2)

    # Define the hyperparameters to tune
    parameters = {
        'max_depth': [5],  # , 5, 7],
        'n_estimators': [100],# 0, 100, 150],
        'learning_rate': [0.1], # , 0.5, 1.0],
        'tree_method': ['gpu_hist']
        # 'booster': ['gbtree']#, 'gblinear', 'dart']
    }
    # 0.39
    # Define the scoring metric
    scorer = make_scorer(f1_score, average='micro')

    # Grid search for best hyperparameters
    grid_search = GridSearchCV(xgb_model, parameters, scoring=scorer, cv=2, n_jobs=-1, verbose=2, error_score='raise')
    grid_search.fit(train, train_labels)

    # Get the best hyperparameters
    best_parameters = grid_search.best_params_

    # Train the model with best hyperparameters
    best_model = xgb.XGBClassifier(**best_parameters)
    best_model.fit(train, train_labels)

    y_pred_xgb = best_model.predict(test)

    print("labels: ", np.unique(test_labels), np.unique(y_pred_xgb), set(test_labels) - set(y_pred_xgb))
    print("2")
    # scores
    print("Accuracy XGB:", metrics.accuracy_score(test_labels, y_pred_xgb))
    print("Precision XGB:", metrics.precision_score(test_labels, y_pred_xgb, average='micro'))
    print("Recall XGB:", metrics.recall_score(test_labels, y_pred_xgb, average='micro'))
    print("F1 Score XGB:", metrics.f1_score(test_labels, y_pred_xgb, average='micro'))

    print('\nXGBoost Classification Report:')
    print(classification_report(test_labels, y_pred_xgb))

############################################################
# Load statistical features from file, previously generated using
# function model_generation_statistical_features
#############################################################
def load_features_from_file(what_data_str, dir_path = "/stat_features/"):
    file_name = str(train_base_path) + dir_path + 'model_' + str(100) + what_data_str
    df_train = pd.read_csv(file_name)
#drop block_sum n 11 column from the dataset for DNN nework statistical file predictions
    df_train1 = df_train.drop(df_train.iloc[:, 10:11],axis = 1)
    print(df_train1.head())

    return df_train1


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(train_base_path)
    # model_generation_statistical_features(X_train, y_train, "_train")
    # model_generation_statistical_features(X_val, y_val, "_val",)
    # model_generation_statistical_features(X_test, y_test, "_test")

    train_labels = indicize_labels(X_train, y_train)
    val_labels = indicize_labels(X_val, y_val)
    test_labels = indicize_labels(X_test, y_test)

    features_train = load_features_from_file("_train")
    # count_zero_files(X_train, "_train", y_train)
    features_test = load_features_from_file("_test")
    # train_random_forest_model(features_train.values, train_labels, features_test.values, test_labels)
    train_xgb_model(features_train.values, train_labels, features_test.values, test_labels)

"""
Created on 17 March 2023
@purpose:
1. Statistical features DNN wo classifier Keras, tensorflow
2. Fifty network wo classifier
3. OOXML classifier
4. Transfer learning for 1, 2, 3
//3. KNN classification (?)
@author: Simona Lisker
@inistitute: HIT
"""
import csv
import inspect
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, hamming_loss

from sklearn.model_selection import cross_val_score, cross_validate
# Pytorch to Keras works but bad precision 0.17 vs 0.46
import numpy as np
import pandas as pd
import seaborn
from keras.layers import Dense, BatchNormalization, Dropout, Reshape, Embedding, Conv1D, GlobalMaxPooling1D, LSTM
#from cuml.metrics import confusion_matrix
from matplotlib import pyplot as plt, pyplot
from numpy import dtype
from sklearn.metrics import classification_report
from tqdm import tqdm

from src.LoadData import load_dataset, train_base_path
from src.StatisticalFeatures import load_features_from_file, indicize_labels, create_categories
from src.TorchDataLoaderWithBert import n_categories
#from src.classification_gpu import chunked_dataset
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
import torch
from keras.utils import get_custom_objects, to_categorical
from keras.optimizers import SGD


# optimizer from hugging face transformers
from torch.optim import Adam
import os
from torch import nn
import torch.nn.functional as F
#import cupy as cp
from torch.optim.lr_scheduler import StepLR
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

#from src.Transformers_Fifty import StorePredictionsInFile
model_directory = train_base_path + 'history' # directory to save model history after every epoch
ooxml_test = 2 #0 - 10 static features, 1 - 11 features with category features, 2 - 11+x feartues with the 1 x gram, 3 - 11+x+y features with sequence
# sequence of length 10 will require 26 features
ooxml_per_file = 1
#add new input - sequence before training more combided model of stat and fifty
# this is a 3rd input. We have stat, fifty, sequence
add_sequence_input_third = 0#1
m_max_length_third_input = 3#512
print('Loading model to GPU...')
device = torch.device('cuda')
print('GPU:', torch.cuda.get_device_name(0))
print('DONE.')

n_num_inputs_stat_model = 21#17#26#10

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

EPOCHS = 5
BATCH_SIZE = 2048*100
LEARNING_RATE = 0.0007#0.0007 is best
NUM_FEATURES = n_num_inputs_stat_model
NUM_CLASSES = 75
file_name_tf_weights = 'tf_stat_weights.h5'
file_name_tf_model = 'tf_stat_model.h5'
file_name_tf_stat_fifty_model = 'tf_stat_fifty_model.h5'
file_name_tf_stat_fifty_weights = 'tf_stat_fifty_weights.h5'
file_name_tf_byte_embed_weights = 'byte_embed_tf.h5'
saveModelPath = train_base_path + "dnn_stat_model.pt"

import tensorflow as tf
import keras
from keras import layers, models, Input, Sequential

from keras.models import Sequential
from keras.layers import Dense

#create data with ms_zip only
ms_zip_6 = [27, 28, 45, 48, 50, 52]
new_labels = [0, 1, 2, 3, 4, 5]

archive_12 = range(31, 33)#41) (29, 41)
first_archive_label = 31#29
def replace_labels_archive(y_train):
    data_len= len(y_train)
    replaced_labels = list(range(data_len))
    for i in range(data_len):
        replaced_labels[i] = y_train[i] - first_archive_label
    return replaced_labels
def replace_labels(y_train, cat_list):

    if (cat_list == archive_12):
        return replace_labels_archive(y_train)

    data_len = len(y_train)
    replaced_labels = list(range(data_len))

    for i in range(data_len):
        if (y_train[i] == 27):
            replaced_labels[i] = 0
        if (y_train[i] == 28):
            replaced_labels[i] = 1
        if (y_train[i] == 45):
            replaced_labels[i] = 2
        if (y_train[i] == 48):
            replaced_labels[i] = 3
        if (y_train[i] == 50):
            replaced_labels[i] = 4
        if (y_train[i] == 52):
            replaced_labels[i] = 5

    return replaced_labels
def getDataByCategory_zip(X_train, y_train, category_labels):
    my_array = np.array(category_labels)

    mask = np.isin(y_train, my_array)
    new_x_train = X_train[mask]
    #new_y_train = [ms_zip_6[index] for index in mask]

    indices = np.where(mask)[0]  # Get the indices of True values in the mask
    indices = indices.astype(int)  # Convert indices to integer type

    new_y_train = np.take(y_train, indices)

    new_y_train = replace_labels(new_y_train, category_labels)
    return new_x_train, new_y_train

# def createDataMSZIP():
#     X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(train_base_path)
#     train_labels = indicize_labels(X_train, y_train)
#     val_labels = indicize_labels(X_val, y_val)
#     categories = sorted(list(set(train_labels)))  # set will return the unique different entries
#     n_categories = len(categories)
#     print("Categories_Byte_embed_DNN_model_train_predict_tensorflow classes = ", n_categories)
#     X_train, train_labels = getDataByCategory_zip(X_train, train_labels)
#     X_val, val_labels = getDataByCategory_zip(X_val, val_labels)
#     return X_train, y_train, X_val, y_val
from keras.utils.vis_utils import plot_model
########################################################
# build sequences using latin-1 encoding and keras tokenizer
# then use this embedding layer as 1st layer (no spaces)
###########################################
def LSTMKerasModel(num_chars, num_classes):
    # Define the model architecture
    model = Sequential()
    model.add(LSTM(units=64,input_shape=(num_chars, m_max_length_third_input), return_sequences=True))
    #model.add(LSTM(512, return_sequences=True))
   # model.add(LSTM(256))

    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))

    model.add(Dense(units=num_classes, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
def EmbedingsKerasModel(num_chars, num_classes):
    embedding_dim = 10
    # Define the model architecture
    model = Sequential()
    model.add(Embedding(input_dim=num_chars, output_dim=embedding_dim, input_length=m_max_length_third_input))
   # model.add(LSTM(10, dropout=0.2, return_sequences=True))
    model.add(LSTM(6))
   # model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(units=6, activation='relu'))
    model.add(Dense(units=3, activation='relu'))
    #model.add(Dense(units=64, activation='relu'))

    model.add(Dense(units=num_classes, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
def EmbedingsKerasModel_1(num_chars, num_classes):
    embedding_dim = 1000
    # Define the model architecture
    model = Sequential()
    model.add(Embedding(input_dim=num_chars, output_dim=embedding_dim, input_length=m_max_length_third_input))
    model.add(LSTM(512, dropout=0.2, return_sequences=True))
    model.add(LSTM(256))
    # model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))

    model.add(Dense(units=num_classes, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
    # Train the
def MulticlassClassificationKeras_3(num_feature, num_class, train = True):

    model = tf.keras.Sequential([
        #        normalizer,
        tf.keras.layers.Dense(512, input_dim=num_feature, activation='relu', name='stat_d1'),
        tf.keras.layers.BatchNormalization(name='batchnorm1'),
        tf.keras.layers.Dense(128, activation='relu', name='stat_d2'),
        tf.keras.layers.BatchNormalization(name='batchnorm2'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu', name='stat_d3'),
        tf.keras.layers.BatchNormalization(name='batchnorm3'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_class, activation='softmax', name='stat_output')
    ])

   # opt = SGD(lr=0.0001, momentum=0.9)
    model.compile(optimizer='adam',  # 'adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.summary()
    return model
class MulticlassClassificationKeras(models.Model):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassificationKeras, self).__init__()

        self.layer_1 = layers.Dense(512, name='layer_1', input_shape=(None, num_feature))
        self.layer_2 = layers.Dense(128, name='layer_2')
        self.layer_3 = layers.Dense(64, name='layer_3')
        self.layer_out = layers.Dense(num_class, activation='softmax', name='layer_out')

        self.relu = layers.ReLU()
        self.dropout = layers.Dropout(0.2)
        self.batchnorm1 = layers.BatchNormalization(axis=-1, name='batchnorm1')
        self.batchnorm2 = layers.BatchNormalization(axis=-1, name='batchnorm2')
        self.batchnorm3 = layers.BatchNormalization(axis=-1, name='batchnorm3')
        self.num_feature = num_feature
        self.num_class = num_class
    def call(self, x):
        print("Call Keras")
        print(x)
        x = self.layer_1(x)
        print(x)
        x = self.batchnorm1(x)
        print(x)
        x = self.relu(x)

        x = self.layer_2(x)
        print(x)
        x = self.batchnorm2(x)
        print(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        print(x)
        x = self.batchnorm3(x)
        print(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)
        print(x)
        return x

    def get_config(self):
        config = super(MulticlassClassificationKeras, self).get_config()
        config.update({'num_feature': self.num_feature, 'num_class': self.num_class})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(num_feature=config['num_feature'], num_class=config['num_class'])

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.num_class),
                                      initializer='uniform',
                                      trainable=True)

class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 512)
        #self.layer_11 = nn.Linear(1024, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        #self.batchnorm1 = nn.BatchNorm1d(1024)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)

    def forward(self, x):
        print("Pytorch")
        print(x)
        x = self.layer_1(x)
        print(x)
        x = self.batchnorm1(x)
        print(x)
        x = self.relu(x)

        # x = self.layer_11(x)
        # x = self.batchnorm11(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        print(x)
        x = self.layer_2(x)
        print(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        print(x)
        x = self.layer_3(x)
        print(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        print(x)
        x = self.layer_out(x)
        print(x)
        return x

def train_stat_DNN_pytorch(model, optimizer, criterion, train_dataset, val_dataset):
    i = 0
    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE
                              )
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)


    print("Begin training.")
    for e in tqdm(range(1, EPOCHS + 1)):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)

            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        # VALIDATION
        with torch.no_grad():

            val_epoch_loss = 0
            val_epoch_acc = 0

            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                y_val_pred = model(X_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()


        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        loss_stats['val'].append(val_epoch_loss / len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

        print(
            f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Val Loss: {val_epoch_loss / len(val_loader):.5f} | Train Acc: {train_epoch_acc / len(train_loader):.3f}| Val Acc: {val_epoch_acc / len(val_loader):.3f}')

    torch.save(model.state_dict(), saveModelPath)
    # Create dataframes
    train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(
        columns={"index": "epochs"})
    train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(
        columns={"index": "epochs"})
    # Plot the dataframes
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
    sns.lineplot(data=train_val_acc_df, x="epochs", y="value", hue="variable", ax=axes[0]).set_title(
        'Train-Val Accuracy/Epoch')
    sns.lineplot(data=train_val_loss_df, x="epochs", y="value", hue="variable", ax=axes[1]).set_title(
        'Train-Val Loss/Epoch')
    plt.show()
    return model
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

def train_stat_DNN_tensorflow(model, train_dataset, val_dataset, train_labels, val_labels):
    print("Fit model on training data")
#this case is for sequence definition. I used total 26 features, last
#column is a feature with a sequence, which should be pre-processed separaterly.
    if (ooxml_test==3):
        tokenizer = Tokenizer()

        string_sequence = train_dataset[:,-1]
        tokenizer.fit_on_texts(string_sequence)
        # Convert the string sequences to sequences of integers
        string_sequence_2 = tokenizer.texts_to_sequences(string_sequence)

        # Pad the integer sequences to a fixed length
        max_length = max(len(seq) for seq in string_sequence_2)
        print("max_length = ", max_length)
        string_sequence_1 = pad_sequences(string_sequence_2, padding='post', truncating='post')
        train_dataset[:, -1] = string_sequence_1

#    print("dataset train size", len(train_dataset.values))
    dummy_train_y = to_categorical(train_labels)
    dummy_val_y = to_categorical(val_labels)
    tf.convert_to_tensor(train_dataset)
    tf.convert_to_tensor(val_dataset)

    EPOCHS = 20  # 30

    #model = MulticlassClassificationKeras_3(num_feature=NUM_FEATURES, num_class=NUM_CLASSES)
    history = model.fit(
        train_dataset,
        dummy_train_y,
        batch_size=64,
        epochs=EPOCHS,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(val_dataset, dummy_val_y),
    )

    model.save_weights(train_base_path + file_name_tf_weights)
    model.save(train_base_path + file_name_tf_model, save_format='tf')
    model.summary()
    tf_model_stat_output = model.layers[-1].output
    print(tf_model_stat_output)
    draw_train_results_tf(history, EPOCHS)
    return model, history, EPOCHS

def draw_train_results_tf(history, EPOCH):
    # Plot training & validation loss values
    plt.style.use("ggplot")
    plt.plot(range(1, EPOCH + 1),
             history.history['loss'])
    plt.plot(range(1, EPOCH + 1),
             history.history['val_loss'],
             linestyle='--')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Batch')
    plt.legend(['Train', 'Val'], loc='upper left')
    fig1 = plt.gcf()
    plt.show()
    plt.draw()


class ClassifierDataset():

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)



def predict_stat_DNN_pytorch(model, test_dataset, y_test):
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    y_pred_list = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            _, y_pred_tags = torch.max(y_test_pred, dim=1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_pred_list = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            _, y_pred_tags = torch.max(y_test_pred, dim=1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

    print(classification_report(y_test, y_pred_list, digits=3))  #not sure if y_test or test_labels shoud be used here
def StorePredictionsInFile(pred_hist, file_name = "/pred_hist.csv"):

    #pred_history_file_name= "/pred_hist.csv"

    with open(model_directory + file_name,'a') as f:
        y=csv.writer(f)
        y.writerow(pred_hist)
        return
def predict_stat_DNN_tensorflow(model, test_dataset, y_test):
    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    if (ooxml_test == 0 ):#ooxml_test !=1
        tf.convert_to_tensor(test_dataset)
    #dummy_test_y = to_categorical(y_test)

    # results = model.evaluate(test_dataset.values, dummy_test_y, batch_size=128)
    # #print("test loss, test acc:", results)
    # print("\n%s: %.2f%%" % (model.metrics_names[1], results[1] * 100))
    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
  #  print("Generate predictions for 3 samples")
   # test_dataset = tf.reshape(test_dataset, (-1, 1, 1))
    logits = model.predict(test_dataset,  batch_size=64)
    prob = tf.nn.softmax(logits, axis=1).numpy()
    predictions = np.argmax(prob, axis=1)
    pred_hist = []
    pred_hist = np.hstack((pred_hist, predictions))
    StorePredictionsInFile(predictions, "/pred_stat_tf.csv")

    print("predictions shape:", pred_hist.shape)
    print("y_test shape:", y_test.shape)

    confusion_matrix = tf.math.confusion_matrix(y_test, pred_hist, num_classes=n_categories)
    #plt.figure(figsize=(15, 13))
    seaborn.heatmap(confusion_matrix, annot=True, fmt='d', cmap='seismic', square=True)
    #seaborn.heatmap(confusion_matrix)
    fig1 = plt.gcf()
    plt.show()
    plt.draw()

    fig1.savefig(train_base_path + '/predict.png')

    target_names = [np.unique(y_test)]
    print(target_names, [np.unique(pred_hist)])
    print(classification_report(y_test, pred_hist, digits=3))  # , target_names=target_names))

    print("end of execution")

   # print(classification_report(y_test, y_pred_list))  #not sure if y_test or test_labels shoud be used here

# Before we start our training, let’s define a function to calculate accuracy per epoch.
# This function takes y_pred and y_test as input arguments.
# We then apply log_softmax to y_pred and extract the class which has a higher probability.
# After that, we compare the predicted classes and the actual classes to calculate the accuracy.
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc
###############################################################################
#Pytorch statistical features DNN model creation - 10 features, precision 0.46
###############################################################################
# def create_statistical_DNN_model_tensorflow(categories):
#     #inputs = tf.keras.Input(shape=(10))
#
#     # x = MulticlassClassificationKeras(num_feature=NUM_FEATURES, num_class=NUM_CLASSES)(inputs)
#     # model = tf.keras.Model(inputs=inputs, outputs=x)
#     #
#     # model.compile(
#     #     optimizer='adam',  # RMSprop(),  # Optimizer
#     #     # Loss function to minimize
#     #     loss='categorical_crossentropy',
#     #     # List of metrics to monitor
#     #     metrics=['accuracy'],
#     # )
#     model = MulticlassClassificationKeras_2(num_feature=NUM_FEATURES, num_class=NUM_CLASSES)
#
#     return model
def create_statistical_DNN_model_pytorch(categories):
    model = MulticlassClassification(num_feature=NUM_FEATURES, num_class=NUM_CLASSES)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(model)

    model = model.to(device)
    print(str(torch.cuda.memory_allocated(device) / 1000000) + 'M')

    torch.cuda.empty_cache()

    return model, optimizer, criterion
#keras + keras
#################################################
# Predicr combined statistical and fifty models
#################################################
def predict_stat_and_fifty_DNN_tensorflow(new_model):
    # features_train_stat = load_features_from_file("_train")
    features_test_stat = load_features_from_file("_test")
    # features_val_stat = load_features_from_file("_val")

    X_test, y_test = load_dataset(train_base_path, True)
    # train_labels = indicize_labels(X_train, y_train)
    # val_labels = indicize_labels(X_val, y_val)
    if (ooxml_test>=1):
        features_test_stat = add_ooxml_cat(features_test_stat, y_test, X_test)
    if (add_sequence_input_third == 1):
        string_sequence_test = BuildThirdInput(X_test, y_test)

        string_sequence_test = tf.convert_to_tensor(string_sequence_test)

    tf.convert_to_tensor(features_test_stat)
    tf.convert_to_tensor(X_test)
    # new_model_inputs = [np.array(X_test.tolist()), features_test_stat.to_numpy()]
    if (add_sequence_input_third == 1):
        new_model_inputs = [X_test, features_test_stat, string_sequence_test]
    else:
        new_model_inputs = [X_test, features_test_stat]
    # x_test_with_channels = x_test[:, :, :, np.newaxis].astype(K.floatx())
    #test_labels = indicize_labels(X_test, y_test)

    logits = new_model.predict(new_model_inputs)
    print(logits)
   # new_model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
    prob = tf.nn.softmax(logits, axis=-1).numpy()
    predictions = np.argmax(prob, axis=-1)
    pred_hist = []
    pred_hist = np.hstack((pred_hist, predictions))
    StorePredictionsInFile(predictions, "/pred_stat_tf_fs.csv")

    print("predictions shape:", pred_hist.shape)
    print("y_test shape:", y_test.shape)
    confusion_matrix = tf.math.confusion_matrix(y_test, pred_hist, num_classes=n_categories)
    #plt.figure(figsize=(15, 13))
    seaborn.heatmap(confusion_matrix, annot=True, fmt='d', cmap='seismic', square=True)
    #seaborn.heatmap(confusion_matrix)
    fig1 = plt.gcf()
    plt.show()
    plt.draw()

    fig1.savefig(train_base_path + '/predict-fs.png')

    target_names = [np.unique(y_test)]
    print(target_names, [np.unique(pred_hist)])
    print(classification_report(y_test, pred_hist, digits=3))  # , target_names=target_names))

def BuildThirdInputWithSpaces(X_train, train_labels, bytes_num = 100):

    grams_lst = [' '] * len(train_labels)
    string_sequence = [' '] * len(train_labels)
    grams_num = bytes_num
    counter = 0
    for i in range(len(train_labels)):
       # grams_lst[i] = X_train[i][:grams_num]
        temp_lst = X_train[i][:grams_num]
        data_temp = ''

        for j in range(grams_num):
            # print("temp_lst[j] = ", temp_lst[j], ", j = ", j)
            # tt = ""

            data_temp = data_temp + " " + str(temp_lst[j])

        string_sequence[i] = data_temp.strip()
        # data_temp = word_tokenize(data_temp)
        # vocab.append(data_temp)

        # string_sequence[i] = tokenizer.texts_to_sequences(np.char.decode(text_block[i][:grams_num],'latin-1'))  # Convert bytes to string
        # if isinstance(grams_lst[i], bytes):
        #     string_sequence[i] = data_temp.decode('latin-1')  # Convert bytes to string
        # elif isinstance(grams_lst[i], np.ndarray):
        #     string_sequence[i] = data_temp.tobytes().decode('latin-1')  # Convert numpy array to bytes and then to string
        # else:
        #     raise ValueError('Invalid type for byte_sequence_feature')

    tokenizer = Tokenizer()

    # string_sequence = train_dataset[:, -1]
    tokenizer.fit_on_texts(string_sequence)


    # Vocabulary size
    vocab_size = len(tokenizer.word_index)
    print("Vocabulary Size:", vocab_size)

    # Word Counts
    # word_counts = tokenizer.word_counts
    # print("Word Counts:", word_counts)

    # Convert text to sequences
    # sequences = tokenizer.texts_to_sequences(string_sequence)
    # print("Text to Sequences:", sequences)

    # Unknown Words
    oov_count = tokenizer.word_counts.get(tokenizer.oov_token, 0)
    print("Unknown Words Count:", oov_count)
    # Convert the string sequences to sequences of integers
    string_sequence_2 = tokenizer.texts_to_sequences(string_sequence)

    # Pad the integer sequences to a fixed length
    m_max_length_third_input = max(len(seq) for seq in string_sequence_2)
    print("max_length = ", m_max_length_third_input)
    string_sequence_1 = pad_sequences(string_sequence_2, padding='post', truncating='post')
    # train_dataset[:, -1] = string_sequence_1
    return string_sequence_1
def BuildThirdInput(X_train, train_labels):

    grams_lst = [-1] * len(train_labels)
    string_sequence = [-1] * len(train_labels)
    grams_num = 100
    for i in range(len(train_labels)):
        grams_lst[i] = X_train[i][:grams_num]

        # string_sequence[i] = tokenizer.texts_to_sequences(np.char.decode(text_block[i][:grams_num],'latin-1'))  # Convert bytes to string
        if isinstance(grams_lst[i], bytes):
            string_sequence[i] = grams_lst[i].decode('latin-1')  # Convert bytes to string
        elif isinstance(grams_lst[i], np.ndarray):
            string_sequence[i] = grams_lst[i].tobytes().decode(
                'latin-1')  # Convert numpy array to bytes and then to string
        else:
            raise ValueError('Invalid type for byte_sequence_feature')

    tokenizer = Tokenizer()

    # string_sequence = train_dataset[:, -1]
    tokenizer.fit_on_texts(string_sequence)
    # Convert the string sequences to sequences of integers
    string_sequence_2 = tokenizer.texts_to_sequences(string_sequence)

    # Pad the integer sequences to a fixed length
    max_length = max(len(seq) for seq in string_sequence_2)
    print("max_length = ", max_length)
    string_sequence_1 = pad_sequences(string_sequence_2, padding='post', truncating='post')
    # train_dataset[:, -1] = string_sequence_1
    return string_sequence_1
def BuildThirdInput_justbytes(X_train, train_labels, seq_len=100):

    grams_lst = [-1] * len(train_labels)
    grams_num = seq_len
    for i in range(len(train_labels)):
        grams_lst[i] = X_train[i][:grams_num]

    return grams_lst
#################################################
# Train cobmined statistical and fifty models
#################################################
def train_stat_and_fifty_DNN_tensorflow(model, epocs = 10, ix='-1'):

    features_train_stat = load_features_from_file("_train")
    features_val_stat = load_features_from_file("_val")

    X_train, y_train,X_val, y_val, _, _ = load_dataset(train_base_path)
    train_labels = indicize_labels(X_train, y_train)
    val_labels = indicize_labels(X_val, y_val)

    if (ooxml_test>=1):
        features_train_stat = add_ooxml_cat(features_train_stat, train_labels, X_train)
        features_val_stat = add_ooxml_cat(features_val_stat, val_labels, X_val)

    if (add_sequence_input_third == 1):
        string_sequence_train = BuildThirdInput(X_train, train_labels)
        string_sequence_val = BuildThirdInput(X_val, val_labels)
        string_sequence_train = tf.convert_to_tensor(string_sequence_train)
        string_sequence_val = tf.convert_to_tensor(string_sequence_val)

    tf.convert_to_tensor(features_train_stat)
    tf.convert_to_tensor(X_train)
    tf.convert_to_tensor(features_val_stat)
    tf.convert_to_tensor(X_val)
    batch_size = 64
    if (add_sequence_input_third == 1):
        # num_features = m_max_length_third_input
        # data_reshaped_train = np.reshape(string_sequence_train, (len(train_labels) // batch_size, batch_size, num_features))
        # data_reshaped_val = np.reshape(string_sequence_val, (len(val_labels) // batch_size, batch_size, num_features))

        new_model_train_inputs = [X_train, features_train_stat, string_sequence_train]
        new_model_val_inputs = [X_val, features_val_stat, string_sequence_val]
    else:
        # new_model_inputs = [np.array(X_test.tolist()), features_test_stat.to_numpy()]
        new_model_train_inputs = [X_train, features_train_stat]
        new_model_val_inputs = [X_val, features_val_stat]
        # x_test_with_channels = x_test[:, :, :, np.newaxis].astype(K.floatx())
    train_labels = indicize_labels(X_train, y_train)
    val_labels = indicize_labels(X_val, y_val)
    print("Train fifty and stat model")

    dummy_train_y = to_categorical(train_labels)
    dummy_val_y = to_categorical(val_labels)

    #model = MulticlassClassificationKeras_3(num_feature=NUM_FEATURES, num_class=NUM_CLASSES)

    history = model.fit(
        new_model_train_inputs,
        dummy_train_y,
        batch_size=batch_size,
        epochs=epocs,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(new_model_val_inputs, dummy_val_y),
    )

    model.save_weights(train_base_path + file_name_tf_stat_fifty_weights + ix)
    model.save(train_base_path + file_name_tf_stat_fifty_model + ix, save_format='tf')
    model.summary()
    tf_model_stat_output = model.layers[-1].output
    print(tf_model_stat_output)
    draw_train_results_tf(history, epocs)
    return model#, history, EPOCHS

#################################################
# Combine pretrained statistical model inputs with
# fifty state of the art model
#################################################
def create_model_stat_and_fifty_keras():
    batch_size = 32
    # statistical features
    # Load the TensorFlow model fifty
    tf_model_fifty = tf.keras.models.load_model(train_base_path + '512_1.h5')
    tf_model_fifty.trainable = False
    #tf_model_fifty.layers[-3].trainable = True
    tf_model_fifty_output = tf_model_fifty.layers[-2].output
    print("Now build fifty original\n")

    tf_model_fifty.summary()
    #input_1 = tf.keras.Input(shape=(None, 512), name='net_1_input')
    model_1_headless = tf.keras.Model(inputs=tf_model_fifty.input,
                                      outputs=tf_model_fifty.layers[-2].output,
                                      name='net_1_model')
    model_1_headless.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(),
                             metrics=['accuracy'])
    model_1_headless.summary()

    # Load the TensorFlow model stat
    print("Now build keras statistical\n")
    tf_model_stat = tf.keras.models.load_model(train_base_path + 'tf_stat_model.h5')
    tf_model_stat.trainable = False
    tf_model_stat.training = False
    for layer in tf_model_stat.layers:
        print(layer.name)
        layer._name = layer.name + '_stat'
    for layer in tf_model_stat.layers:
        print(layer.name)

    tf_model_stat.summary()

    model_2_headless = tf.keras.Model(inputs=tf_model_stat.input, outputs=tf_model_stat.layers[-2].output,
                                      name='net_2_model')
    model_2_headless.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(),
                             metrics=['accuracy'])
    model_2_headless.summary()
    tf_model_stat_output = tf_model_stat.layers[-2].output
    #tf_model_stat.layers[-3].trainable = True

    #if (add_sequence_input_third ==1):
        # X_train, y_train, X_val, y_val, _, _ = load_dataset(train_base_path)
        # train_labels = indicize_labels(X_train, y_train)

        # if (add_sequence_input_third == 1):
        #     string_sequence_train = BuildThirdInput(X_train, train_labels)
        #     string_sequence_train = tf.convert_to_tensor(string_sequence_train)
        # dummy_input = tf.ones((1, 3))

      #  combined_features = tf.keras.layers.concatenate([tf_model_fifty_output, tf_model_stat_output, dummy_input], name="concatenated_layer")
  #  else:
    combined_features = tf.keras.layers.concatenate([tf_model_fifty_output, tf_model_stat_output], name="concatenated_layer")
    # inputs = tf.keras.Input(shape=(combined_features.shape[1],))
    x = tf.keras.layers.Dense(128, activation='relu', name="new_dense_128")(combined_features)
    #x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.Dropout(0.2, name="new_dropout_1")(x)
    # x = tf.keras.layers.Dense(64, activation='relu', name="new_dense_64")(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dropout(0.2, name="new_dropout_2")(x)
    if (add_sequence_input_third == 1):
        input_shape = (m_max_length_third_input,)  # Shape of the input tensor, with None for variable batch size
        input_tensor = tf.keras.Input(shape=input_shape)
        inputs = [tf_model_fifty.input, tf_model_stat.input, input_tensor]
    else:
        inputs = [tf_model_fifty.input, tf_model_stat.input]
    outputs = tf.keras.layers.Dense(n_categories, activation='softmax')(x)
    new_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return new_model
#     new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     new_model.summary()
#
#    # plot_model(new_model, to_file=train_base_path + 'new_model_plot.png', show_shapes=True, show_layer_names=True)
#     # combine input of 2 models and train them both together, then predict
#
#     new_model = train_stat_and_fifty_DNN_tensorflow(new_model, 30, "-1")
#     new_model.trainable = True
#     new_model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
#     new_model.summary()
# #    new_model.load_weights(train_base_path + file_name_tf_stat_fifty_model + '-2')
#
#     new_model = train_stat_and_fifty_DNN_tensorflow(new_model, 10, "-2")
#     #new_model = tf.keras.models.load_model(train_base_path + file_name_tf_stat_fifty_model)
#     predict_stat_and_fifty_DNN_tensorflow(new_model)
#################################################
# Create combined pretrained statistical model inputs with
# fifty state of the art model, train and predict
#################################################
def keras_stat_fifty_exec():
    new_model = create_model_stat_and_fifty_keras()
    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    new_model.summary()

    # plot_model(new_model, to_file=train_base_path + 'new_model_plot.png', show_shapes=True, show_layer_names=True)
    # combine input of 2 models and train them both together, then predict

    new_model = train_stat_and_fifty_DNN_tensorflow(new_model, 7, "-1")
    new_model.trainable = True
    new_model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    new_model.summary()
   # new_model.load_weights(train_base_path + file_name_tf_stat_fifty_model + '-2')

    new_model = train_stat_and_fifty_DNN_tensorflow(new_model, 10, "-2")
   # new_model = tf.keras.models.load_model(train_base_path + file_name_tf_stat_fifty_model + '-2')
    predict_stat_and_fifty_DNN_tensorflow(new_model)
#########################################
# KNN classifier for Fifty and statistical
#########################################
def KNN_Classifier():
    k = 3
    trained_model = create_model_stat_and_fifty_keras()
    trained_model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    trained_model.summary()

    trained_model.load_weights(train_base_path + file_name_tf_stat_fifty_model + '-2')

    print("Stat + Fifty + modelKNN")
    model = KNeighborsClassifier(n_neighbors=k).fit(X, y)
    y_hat = model.predict(test_X)
    # 10 fold cross validation
    accuracy = cross_val_score(model, X, y, cv=cross, scoring='accuracy').mean()
    scoring = ['precision_macro', 'recall_macro']
    clf = KNeighborsClassifier(n_neighbors=k)
    scores = cross_validate(clf, X, y, scoring=scoring, cv=cross, return_train_score=True)
    precision = scores['test_precision_macro'].mean()
    recall = scores['test_recall_macro'].mean()
    print("*********Results**********")
    print("Avg. Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    f.write(str(sizes[i]) + '\t' + str(accuracy) + '\t' + str(precision) + '\t' + str(recall) + '\t' + str(
        hamming_loss(test_Y, y_hat)) + '\n')
    print("Done for vector length:" + str(sizes[i]) + " and sample size:" + str(sample_size))
    print("*************************************")
    # Compute confusion matrix
    y_test = test_Y
    y_pred = y_hat
    class_names = np.unique(y_pred)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=3)
    # save confusion matrix to csv file
    df_confusion = pd.crosstab(y_test, y_pred)
    # Its important to use binary mode
    knnPickle = open('knnpickle_file', 'wb')
    # source, destination
    pickle.dump(model, knnPickle)
    # close the file
    knnPickle.close()
def preprocess(x):
    byte_sequence = x
    string_sequence = byte_sequence.decode('utf-8')  # Convert bytes to string
    x = string_sequence
    return x

def add_ooxml_cat(features_train, train_labels, text_block):
    grams_num = 10

    new_cat_oxxml =[-1]*len(train_labels)
    grams_lst = [-1]*len(train_labels)
    string_sequence = [-1]*len(train_labels)
    ooxml_labels = [27, 28, 45, 46, 48, 50, 52]
    for i in range(len(train_labels)):
        if (ooxml_per_file == 1):
            if (train_labels[i] == 27):
                new_cat_oxxml[i] = 0
            elif (train_labels[i] == 28):
                new_cat_oxxml[i] = 1
            elif (train_labels[i] == 45):
                new_cat_oxxml[i] = 2
            elif (train_labels[i] == 46):
                new_cat_oxxml[i] = 3
            elif (train_labels[i] == 48):
                new_cat_oxxml[i] = 4
            elif (train_labels[i] == 50):
                new_cat_oxxml[i] = 5
            elif (train_labels[i] == 52):
                new_cat_oxxml[i] = 6
            else:
                new_cat_oxxml[i] = 7

        else:
            if train_labels[i] in ooxml_labels:
                new_cat_oxxml[i] = 1
            else:
                new_cat_oxxml[i] = 0

        if (ooxml_test >= 2):
            grams_lst[i] = text_block[i][:grams_num]

        if (ooxml_test == 3):
            # string_sequence[i] = tokenizer.texts_to_sequences(np.char.decode(text_block[i][:grams_num],'latin-1'))  # Convert bytes to string
            if isinstance(grams_lst[i], bytes):
                string_sequence[i] = grams_lst[i].decode('latin-1')  # Convert bytes to string
            elif isinstance(grams_lst[i], np.ndarray):
                string_sequence[i] = grams_lst[i].tobytes().decode('latin-1')  # Convert numpy array to bytes and then to string
            else:
                raise ValueError('Invalid type for byte_sequence_feature')

    # #features_train['ooxml'] = new_cat_oxxml
    # #features_train['sequence_6'] = grams_lst

    # Assume your dataframe is called 'df' and has two new columns 'int_column' and 'ndarray_column'
    int_values = np.array(new_cat_oxxml)
    if (ooxml_test >= 2):
        ndarray_values = np.stack(np.asarray(grams_lst))
        print("cat - ", int_values[:grams_num], "10 bytes - ", ndarray_values[:grams_num])
        print("string_sequence - ", string_sequence[:grams_num])
        # Combine the integer and ndarray values into a single array

    if (ooxml_test == 1):
        #this is only stat features with cat zip category
        combined_values = np.column_stack((features_train, int_values))
    elif (ooxml_test == 2):  # this is without the sequence inside, but with 1-gram features
        combined_values = np.column_stack((features_train, int_values, ndarray_values))
    elif (ooxml_test == 3):  # 26 features, grams_num = 10
        combined_values = np.column_stack((features_train, int_values, ndarray_values, string_sequence))

    print("18 features - ", combined_values[:grams_num])

    return combined_values

def Statistical_Feat_DNN_model_train_predict_tensorflow_ooxml():
    features_train = load_features_from_file("_train")
    features_test = load_features_from_file("_test")
    features_val = load_features_from_file("_val")

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(train_base_path)
    train_labels = indicize_labels(X_train, y_train)
    val_labels = indicize_labels(X_val, y_val)
    # test_labels = indicize_labels(X_test, y_test)

    n_categories, categories = create_categories(features_train.values, train_labels)

    features_train = add_ooxml_cat(features_train, train_labels, X_train)
    features_val = add_ooxml_cat(features_val, val_labels, X_val)

    model = MulticlassClassificationKeras_3(NUM_FEATURES, n_categories)
    model.load_weights(train_base_path + file_name_tf_weights)
    #
    model, history_dataframe, EPOCH = train_stat_DNN_tensorflow(model, features_train, features_val, train_labels,
                                                                val_labels)

    new_model = model  # MulticlassClassificationKeras_3(NUM_FEATURES, n_categories, False)

    print("Now build keras\n")
    keras_input = (32, n_num_inputs_stat_model)
    new_model.build(input_shape=keras_input)
    # # dummy_input = tf.ones((None, n_num_inputs_stat_model))
    # # new_model(dummy_input)
    # new_model.load_weights(train_base_path + file_name_tf_weights)
    # # new_model=model
    features_test = add_ooxml_cat(features_test, y_test, X_test)
    predict_stat_DNN_tensorflow(new_model, features_test, y_test)
############################################################
# Byte embedding tensorflow model, result file name file_name_tf_byte_embed_weights = "byte_embed_tf.h5"
#
###################################
def train_byte_embed_DNN_tensorflow(model, string_sequence_train, string_sequence_val, train_labels, val_labels):
    #    print("dataset train size", len(train_dataset.values))
    dummy_train_y = to_categorical(train_labels)
    dummy_val_y = to_categorical(val_labels)

    # # Reshape the training data
    # string_sequence_train = np.array(string_sequence_train).reshape(len(string_sequence_train), m_max_length_third_input, 1)
    #
    # # Reshape the validation data
    # string_sequence_val = np.array(string_sequence_val).reshape(len(string_sequence_val), m_max_length_third_input, 1)

    string_sequence_train = tf.convert_to_tensor(string_sequence_train)
    string_sequence_val = tf.convert_to_tensor(string_sequence_val)
    EPOCHS = 2  # 30

    # model = MulticlassClassificationKeras_3(num_feature=NUM_FEATURES, num_class=NUM_CLASSES)
    history = model.fit(
        string_sequence_train,
        dummy_train_y,
        batch_size=64,
        epochs=EPOCHS,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(string_sequence_val, dummy_val_y),
    )

    model.save_weights(train_base_path + file_name_tf_byte_embed_weights)
    model.save(train_base_path + file_name_tf_byte_embed_weights , save_format='tf')
    model.summary()
    tf_model_stat_output = model.layers[-1].output
    print(tf_model_stat_output)
    draw_train_results_tf(history, EPOCHS)
    return model, history, EPOCHS
def predict_byte_embed_DNN_tensorflow(model, test_dataset, y_test, n_categories):
    test_dataset = tf.convert_to_tensor(test_dataset)
    # dummy_test_y = to_categorical(y_test)

    # results = model.evaluate(test_dataset.values, dummy_test_y, batch_size=128)
    # #print("test loss, test acc:", results)
    # print("\n%s: %.2f%%" % (model.metrics_names[1], results[1] * 100))
    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    #  print("Generate predictions for 3 samples")
    # test_dataset = tf.reshape(test_dataset, (-1, 1, 1))
    logits = model.predict(test_dataset, batch_size=64)
    prob = tf.nn.softmax(logits, axis=1).numpy()
    predictions = np.argmax(prob, axis=1)
    pred_hist = []
    pred_hist = np.hstack((pred_hist, predictions))
    StorePredictionsInFile(predictions, "/pred_byte_embed_tf.csv")

    print("predictions shape:", pred_hist.shape)
#    print("y_test shape:", y_test.shape)

    confusion_matrix = tf.math.confusion_matrix(y_test, pred_hist, num_classes=n_categories)
    # plt.figure(figsize=(15, 13))
    seaborn.heatmap(confusion_matrix, annot=True, fmt='d', cmap='seismic', square=True)
    # seaborn.heatmap(confusion_matrix)
    fig1 = plt.gcf()
    plt.show()
    plt.draw()

    fig1.savefig(train_base_path + '/predict.png')

    target_names = [np.unique(y_test)]
    print(target_names, [np.unique(pred_hist)])
    print(classification_report(y_test, pred_hist, digits=3))  # , target_names=target_names))

    print("end of execution")
def Byte_embed_DNN_model_train_predict_tensorflow():
    zip_only_model = 1
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(train_base_path)
    train_labels = indicize_labels(X_train, y_train)
    val_labels = indicize_labels(X_val, y_val)
    if (zip_only_model == 1):
        cat_list = archive_12#ms_zip_6
        X_train, train_labels = getDataByCategory_zip(X_train, train_labels, cat_list)
        X_val, val_labels = getDataByCategory_zip(X_val, val_labels, cat_list)

    categories = sorted(list(set(train_labels)))  # set will return the unique different entries
    n_categories = len(categories)
    print("Byte_embed_DNN_model_train_predict_tensorflow classes = ", n_categories)

    features_num = 256
    model = EmbedingsKerasModel(features_num, n_categories)#MulticlassClassificationKeras_3(NUM_FEATURES_EMBED_VECTOR, n_categories)

    #read exisitng model start
    #keras_input = (32, n_num_inputs_stat_model)
    #model.build(input_shape=keras_input)

    #model.load_weights(train_base_path + file_name_tf_weights)
    # read exisitng model end
    string_sequence_train = BuildThirdInput_justbytes(X_train, train_labels, m_max_length_third_input)
    print(len(string_sequence_train))
    string_sequence_val = BuildThirdInput_justbytes(X_val, val_labels, m_max_length_third_input)
    print(len(string_sequence_val))

    model, history_dataframe, EPOCH = train_byte_embed_DNN_tensorflow(model, string_sequence_train, string_sequence_val, train_labels, val_labels)

    new_model = model #MulticlassClassificationKeras_3(NUM_FEATURES, n_categories, False)

    print("Now build keras\n")
    if (zip_only_model == 1):
        keras_input = (32, m_max_length_third_input)
    else:
        keras_input = (32, n_num_inputs_stat_model)
    new_model.build(input_shape=keras_input)
    # dummy_input = tf.ones((None, n_num_inputs_stat_model))
    # new_model(dummy_input)
    new_model.load_weights(train_base_path + file_name_tf_byte_embed_weights)
    new_model=model

    if (zip_only_model == 1):
        X_test, y_test = getDataByCategory_zip(X_test, y_test, cat_list)

    string_sequence_test = BuildThirdInput_justbytes(X_test, y_test, m_max_length_third_input)
    predict_byte_embed_DNN_tensorflow(new_model, string_sequence_test, y_test, n_categories)
############################################################
# Statistical 10 features tensorflow model, result file name tf_stat.h5
# precision 0.42
###################################
def Statistical_Feat_DNN_model_train_predict_tensorflow():
    features_train = load_features_from_file("_train")
    features_test = load_features_from_file("_test")
    features_val = load_features_from_file("_val")

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(train_base_path)
    train_labels = indicize_labels(X_train, y_train)
    val_labels = indicize_labels(X_val, y_val)
    #test_labels = indicize_labels(X_test, y_test)
    n_categories, categories = create_categories(features_train.values, train_labels)
    model = MulticlassClassificationKeras_3(NUM_FEATURES, n_categories)
    #
    model, history_dataframe, EPOCH = train_stat_DNN_tensorflow(model, features_train, features_val, train_labels, val_labels)

    new_model = model #MulticlassClassificationKeras_3(NUM_FEATURES, n_categories, False)

    print("Now build keras\n")
    keras_input = (32, n_num_inputs_stat_model)
    new_model.build(input_shape=keras_input)
    # dummy_input = tf.ones((None, n_num_inputs_stat_model))
    # new_model(dummy_input)
    #new_model.load_weights(train_base_path + file_name_tf_weights)
   # new_model=model

    predict_stat_DNN_tensorflow(new_model, features_test, y_test)
#############################################################
#Train and prefict DNN Statistical model = pytorch
#Read statistical features from file 0.46
#############################################################
def Statistical_Feat_DNN_model_train_predict_pytorch():
    features_train = load_features_from_file("_train")
    features_test = load_features_from_file("_test")
    features_val = load_features_from_file("_val")

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(train_base_path)
    train_labels = indicize_labels(X_train, y_train)
    val_labels = indicize_labels(X_val, y_val)
    test_labels = indicize_labels(X_test, y_test)
    # n_categories, categories = create_categories(features_train.values, train_labels)
    model, optmizer, criterion = create_statistical_DNN_model_pytorch(n_categories)
    model.to(device)
    # model = train_stat_CNN(features_train.values, train_labels)
    train_dataset = ClassifierDataset(torch.tensor(features_train.values, dtype=torch.float32),
                                      torch.tensor(train_labels, dtype=torch.int64))
    val_dataset = ClassifierDataset(torch.tensor(features_val.values, dtype=torch.float32),
                                    torch.tensor(val_labels, dtype=torch.int64))
    test_dataset = ClassifierDataset(torch.tensor(features_test.values, dtype=torch.float32),
                                     torch.tensor(test_labels, dtype=torch.int64))
    model = train_stat_DNN_pytorch(model, optmizer, criterion, train_dataset, val_dataset)
    saveModelPath = train_base_path + "dnn_stat_model.pt"
    state_dict = torch.load(saveModelPath)
    ## Load the state dictionary into your model
    model.load_state_dict(state_dict)
    predict_stat_DNN_pytorch(model, test_dataset, test_labels)

# def convertStatPytorchToKerasModel():
#     # Load the PyTorch model weights
#     pytorch_model = MulticlassClassification(n_num_inputs_stat_model, NUM_CLASSES)
#     saveModelPath = train_base_path + "dnn_stat_model.pt"
#     pytorch_model.load_state_dict(torch.load(saveModelPath))
#     pytorch_model.to(device)
#     summary(pytorch_model, input_size=(10, ))
#
#     # Convert the PyTorch model to Keras
#     keras_model = MulticlassClassificationKeras(n_num_inputs_stat_model, NUM_CLASSES)
#     # Compile the Keras model
#     keras_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     keras_model.build(input_shape=(None, n_num_inputs_stat_model))
#     keras_model.summary()
#
#     # pytoch weights to keras
#     trained_weights = pytorch_model.state_dict()
#     pytorch_layers = []  # get pytroch layers names and suffixes
#     for x in (trained_weights):
#         pytorch_layers.append(x[:x.find('.')])
#
#     unique_layers = np.unique(pytorch_layers)
#
#     for layer in unique_layers:
#         weights = trained_weights['{}.weight'.format(layer)].cpu().numpy()  # torch weights (nf, ch, x, y)
#         biases = trained_weights['{}.bias'.format(layer)].cpu().numpy()
#         if 'batchnorm' in layer:
#             running_mean = trained_weights['{}.running_mean'.format(layer)].cpu().numpy()
#             running_var = trained_weights['{}.running_var'.format(layer)].cpu().numpy()
#             W = [weights, biases, running_mean, running_var]
#         elif 'layer' in layer:
#             biases = trained_weights['{}.bias'.format(layer)].cpu().numpy()
#             W = [weights.T, biases]
#         else:
#             W = [np.moveaxis(weights, [0, 1], [3, 2]), biases]  # transpose to (x, y, ch, nf) keras version
#         keras_model.get_layer(layer).set_weights(W)
#
#     # Train the Keras model
#     # x_train = np.random.rand(100, n_num_inputs_stat_model)
#     # y_train = np.random.rand(100, NUM_CLASSES)
#     # keras_model.fit(x_train, y_train, epochs=10, batch_size=32)
#
#     features_test = load_features_from_file("_test")
#
#     _,_,_,_, X_test, y_test = load_dataset(train_base_path)
#
#     test_labels = indicize_labels(X_test, y_test)
#     #score = keras_model.evaluate(np.array(features_test.values), np.array(test_labels), verbose=0)
#     #print('Test loss:', score[0])
#     #print('Test accuracy:', score[1])
#     # Predict using the Keras model
#     y_pred_prob = keras_model.predict(features_test, batch_size=BATCH_SIZE)
#     y_pred = y_pred_prob.argmax(axis=1)
#
#     # Print classification report
#     print(classification_report(test_labels, y_pred))


# define and fit model on a training dataset
def create_model(trainX, trainy):
	# define model
    model = create_model_stat_and_fifty_keras()

    #model.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    #     new_model.summary()
# repeated evaluation of a standalone model
def eval_standalone_model(trainX, trainy, testX, testy, n_repeats):
    scores = list()
    model = create_model(trainX, trainy)
    for _ in range(n_repeats):
        #define and fit a new model on the train dataset

        print('eval_standalone_model, start train')
        model.fit(trainX, trainy, epochs=1, verbose=1)
        # evaluate model on test dataset
        _, test_acc = model.evaluate(testX, testy, verbose=1)
        scores.append(test_acc)
    return scores, model


# repeated evaluation of a model with transfer learning
def eval_transfer_model(model,trainX, trainy, testX, testy, n_fixed, n_repeats):
    scores = list()

    for _ in range(n_repeats):
        # load model
       # model.trainable = True
       # model = create_model()#load_model('model.h5')
        # mark layer weights as fixed or not trainable
        for i in range(n_fixed):
            model.layers[-i].trainable = True
        # re-compile model
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        print("some freezed layers = ", n_fixed)
        model.summary()
        # fit model on train dataset
        model.fit(trainX, trainy,  batch_size=1, epochs=30, verbose=0)
        # evaluate model on test dataset
        _, test_acc = model.evaluate(testX, testy, verbose=0)
        scores.append(test_acc)
    return scores, model
from numpy import mean
from numpy import std
def transfer_learning():
    # prepare data for problem 2
    #trainX, trainy, testX, testy = samples_for_seed(2)
    features_train_stat = load_features_from_file("_train")
    features_test_stat = load_features_from_file("_test")

    X_train, y_train,X_val, y_val, X_test, y_test = load_dataset(train_base_path)
    # train_labels = indicize_labels(X_train, y_train)
    # val_labels = indicize_labels(X_val, y_val)
    tf.convert_to_tensor(features_train_stat)
    tf.convert_to_tensor(X_train)
    tf.convert_to_tensor(features_test_stat)
    tf.convert_to_tensor(X_test)

    # new_model_inputs = [np.array(X_test.tolist()), features_test_stat.to_numpy()]
    new_model_train_inputs = [X_train, features_train_stat]
    new_model_test_inputs = [X_test, features_test_stat]
    # x_test_with_channels = x_test[:, :, :, np.newaxis].astype(K.floatx())
    train_labels = indicize_labels(X_train, y_train)
    test_labels = indicize_labels(X_test, y_test)
    print("Train fifty and stat model")

    dummy_train_y = to_categorical(train_labels)
    dummy_test_y = to_categorical(test_labels)

    n_repeats = 1
    dists, dist_labels = list(), list()

    # repeated evaluation of standalone model
    standalone_scores, model = eval_standalone_model(new_model_train_inputs,
        dummy_train_y, new_model_test_inputs, dummy_test_y, n_repeats)
    print('Standalone %.3f (%.3f)' % (mean(standalone_scores), std(standalone_scores)))
    dists.append(standalone_scores)
    dist_labels.append('standalone')

    # repeated evaluation of transfer learning model, vary fixed layers
    n_fixed = 3
    for i in range(n_fixed):
        scores = eval_transfer_model(model, new_model_train_inputs,
        dummy_train_y, new_model_test_inputs, dummy_test_y, i, n_repeats)
        print('Transfer (fixed=%d) %.3f (%.3f)' % (i, mean(scores), std(scores)))
        dists.append(scores)
        dist_labels.append('transfer f=' + str(i))

    # box and whisker plot of score distributions
    pyplot.boxplot(dists, labels=dist_labels)
    pyplot.show()

#############################################################
# 1. First need to train statistical model separately
# Statistical_Feat_DNN_model_train_predict_tensorflow
# 2. Combine pretrained statistical model inputs with
# fifty state of the art model, train and preidct
# create_model_stat_and_fifty_keras
#############################################################
if __name__ == "__main__":
    ###Statistical_Feat_DNN_model_train_predict_pytorch() - 0.46 deprecated
    keras_stat_fifty_exec()
    #Byte_embed_DNN_model_train_predict_tensorflow()
    # Statistical_Feat_DNN_model_train_predict_tensorflow() #0.44 30 epocs
  #  Statistical_Feat_DNN_model_train_predict_tensorflow_ooxml() #0.517 30 epocs cat and 1-gram, 0.47 30 epocs, add sequence is not working
    ##end#####################################
    #transfer_learning()

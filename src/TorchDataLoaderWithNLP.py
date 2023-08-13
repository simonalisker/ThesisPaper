"""
Created on Nov 2022
@purpose:
1. Load FiFTy dataset from files .npz provided by FiFTy
2. Train and predict using Roberta model
@author: Simona Lisker
@inistitute: HIT
"""
import numpy as np
from src.LoadData import load_dataset, train_base_path
from sklearn.metrics import classification_report
from transformers import RobertaTokenizer, RobertaForSequenceClassification

import torch
# optimizer from hugging face transformers
from torch.optim import Adam

from torch import nn

from matplotlib import pyplot as plt
print('Loading model to GPU...')
device = torch.device('cuda')
print('GPU:', torch.cuda.get_device_name(0))
print('DONE.')

model_directory = train_base_path + 'history' # directory to save model history after every epoch
model_history_file_name = '/model_history.csv'
file_name = './saved_weights_512_1_full_chunked_roberta.pt'

n_categories = 75
read_from_path = train_base_path + "/transformers_evaluation_data_small/"
size = 100
try_size = False
try_len = 300000
def load_data_from_file(what_data_str, size):
  inputs = ""
  count = 0
  texts = []
  f = open(str(read_from_path) + 'model_' + str(size) + what_data_str , "r")
  while True :
    count += 1
    line = f.readline()
    if not line:# or count > 1000000:
         break
    texts.append(line)
  f.close()
  print(count)
  return texts
#https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html


class RobertoDataLoader:
    def __init__(self, tokenizer1, training_file, labels, batch_size, max_length):
        self.tokenizer = tokenizer1
        self.training_file = training_file
        self.batch_size = batch_size
        self.max_length = max_length
        self.texts = load_data_from_file(training_file, size)
        self.labels = labels
        if (try_size):
            self.len = try_len#len(labels)
        else:
            self.len = len(labels)
    def data_len(self):
        return self.len

    def __iter__(self):
        # Open the training file
       # with open(self.training_file, 'r') as f:
            # Read the lines in the file
          #  lines = f.readlines()
          #print("111")
            if (try_size):
                a = try_len
            else:
                a = len(self.texts)

            # Calculate the number of batches
            num_batches = a // self.batch_size
            #print("num_batches", num_batches)
            # Iterate over the number of batches
            for i in range(num_batches):
                # Get the batch start and end indices

                # if i % 500 == 0 and not i == 0:
                #     print('  Batch {:>5,}  of  {:>5,}.'.format(i, num_batches))
                batch_start = i * self.batch_size
                batch_end = (i+1) * self.batch_size
                # Get the batch lines
                batch_lines = self.texts[batch_start:batch_end]
                batch_labels = self.labels[batch_start:batch_end]
                # Tokenize the batch lines
                tokenized_batch = self.tokenizer.batch_encode_plus(
                    list(batch_lines), max_length=self.max_length, pad_to_max_length=True,
                    truncation=True)
                # Convert the tokenized batch to tensors
                input_ids = torch.tensor(tokenized_batch['input_ids'])
                attention_mask = torch.tensor(tokenized_batch['attention_mask'])
                # Yield the input_ids, attention_mask tensors as a tuple
                torch_labels = torch.tensor(batch_labels)#.tolist())
                targets = torch_labels.type(torch.LongTensor)
                yield input_ids, attention_mask, targets


class BertBinaryClassifier(nn.Module):
    def __init__(self, bert):
        super(BertBinaryClassifier, self).__init__()

        self.bert = bert
        # print(bert)
      #  self.dropout = nn.Dropout(0.1)
        #self.linear = nn.Linear(768, 5)
        self.sigmoid = nn.Sigmoid()
       # self.softmax = nn.Softmax()

    def forward(self, tokens, masks=None):
        #_, pooled_output \
        pooled_output = self.bert(tokens, attention_mask=masks, return_dict=False)
       # dropout_output = self.dropout(pooled_output)
       # linear_output = self.linear(dropout_output)
       # proba = self.softmax(linear_output)
        proba = self.sigmoid(pooled_output)
        return proba

def loss_fn(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets)

def create_model():#tokens_train):


    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=n_categories)
    model = model.to(device)
    print(str(torch.cuda.memory_allocated(device)/1000000 ) + 'M')
    optimizer = Adam(model.parameters(), lr=3e-5)

    torch.cuda.empty_cache()
    epocs = 2
    return model, optimizer, epocs


# function to train the model
def train(model, train_dataloader):
    model.train()

    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions
    total_preds = []
    total_losses = []
    # iterate over batches
    for step, batch in enumerate(train_dataloader):

        # progress update after every 50 batches.
        if step % 5000 == 0 and not step == 0:
            avg_loss = total_loss / step
            print('  Batch {:>5,}  of  {:>5,}, current loss {:>5,}, avg_loss {:>5,}.'.format(step, (train_dataloader.data_len()/batch_size),
                                                                                loss, avg_loss))
        # push the batch to gpu

        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch

        #labels = labels.view(-1, 1)
        #print('labels ', labels, '\nsize =', labels.size())
        # clear previously calculated gradients

        model.zero_grad()

        # get model predictions for the current batch -> here
        preds = model(sent_id, mask)
       # loss, preds, last_hidden_states, pooler_output = model(sent_id, mask)
        #print('preds ', preds, '\nshape =', preds.size())
        # # compute the loss between actual and predicted values
       # #labels = labels.to(torch.float32)

        loss = loss_fn(preds.logits, labels)
        # predicted_token_class_ids = preds.logits.argmax(-1)
        # pred_labels = predicted_token_class_ids
        # loss = model(sent_id, labels=pred_labels).loss

        # add on to the total loss
        total_loss = total_loss + round(loss.item(), 2)
        # backward pass to calculate the gradients
        total_losses.append(loss.item())
        model.zero_grad()
        loss.backward()

        # clip the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds = preds.logits.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / (train_dataloader.data_len()/batch_size)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    # returns the loss and predictions
    return avg_loss, total_preds, total_losses


# function for evaluating the model
def evaluate(model, val_dataloader):
    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []
    total_losses = []
    # iterate over batches
    for step, batch in enumerate(val_dataloader):

        # Progress update every 10 batches.
        if step % 1000 == 0 and not step == 0:
            #Calculate elapsed time in minutes.
            #elapsed = format_time(time.time() - t0)

           # Report progress.
            avg_loss = total_loss / step
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, (val_dataloader.data_len()/batch_size), loss, avg_loss))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]  # to(device)

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds = model(sent_id, mask)
           # labels = labels.view(-1, 1)
            # compute the validation loss between actual and predicted values
            #labels = labels.to(torch.float32)
            loss = loss_fn(preds.logits, labels)

            # predicted_token_class_ids = preds.logits.argmax(-1)
            # pred_labels = predicted_token_class_ids
            # loss = model(sent_id, labels=pred_labels).loss
            #
            # # add on to the total loss
            total_loss = total_loss + round(loss.item(), 2)

            preds = preds.logits.detach().cpu().numpy()

            total_losses.append(loss.item())
            total_preds.append(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / (val_dataloader.data_len()/batch_size)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds, total_losses


def train_epocs(epocs, model, train_dataloader, val_dataloader, y_val):
    # set initial loss to infinite
    best_valid_loss = float('inf')
    best_train_loss = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_losses = []
    valid_losses = []

    # for each epoch
    for epoch in range(epocs):

        print('\n Epoch {:} / {:}'.format(epoch + 1, epocs))

        print("Start training")
        # train model
        train_loss, _, train_losses_list = train(model, train_dataloader)#train()

        train_losses.append(train_loss)
        train_loss_mean = np.mean(train_losses)

        print("train_loss_mean= ", train_loss_mean, "; train_loss = ", train_loss)
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), train_base_path + file_name )
        torch.save(model.state_dict(), train_base_path + file_name + str(epoch))

        print("End training, start evaluation")
        # evaluate model
        valid_loss, val_preds, val_losses_list = evaluate(model, val_dataloader)#evaluate()
        val_loss_mean = np.mean(val_losses_list)
        print("val_loss_mean= ", val_loss_mean, "; val_loss = ", valid_loss)

        print("End evaluation")
        # save the best model
        # if valid_loss < best_valid_loss:
        #     best_valid_loss = valid_loss
        #     torch.save(model.state_dict(), train_base_path + file_name)

        # append training and validation loss

        valid_losses.append(valid_loss)

        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')


        # if (try_size):
        #     print(classification_report(val_y[:try_len], valid_losses))
        # else:
        #     print(classification_report(val_y, valid_losses))
        print(classification_report(y_val[:len(val_preds)], val_preds.argmax(-1)))
       # plot_train_logs(train_losses, valid_losses)

        np.mean(valid_losses)
        model.eval()
    return train_losses, valid_losses

def predict(model, test_dataloader, y_test):
    print("Predict")
    bert_predicted = []
    all_logits = []
    # test_labels = []
    with torch.no_grad():
        for step_num, batch_data in enumerate(test_dataloader):
            token_ids, masks, labels = tuple(t.to(device) for t in batch_data)

            if step_num % 1000 == 0 and not step_num == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step_num, (test_dataloader.data_len()/batch_size)))
            logits = model(token_ids, masks)
            numpy_logits = logits.logits.cpu().detach().numpy()

            # bert_predicted += list(numpy_logits[:, 0] > 0.5)
            bert_predicted += list(numpy_logits.argmax(-1))
            all_logits += list(numpy_logits)
            # test_labels.append(labels)

    print("bert_predicted len = ", len(bert_predicted), "; all_logits = ", len(all_logits))

    # np.mean(bert_predicted)
    # np.mean(all_logits)
    print("mean bert_predicted len = ", len(bert_predicted), "; mean all_logits = ", len(all_logits))
    # if (try_size):
    #     print(classification_report(test_labels[:len(all_logits)], all_logits))
    # else:
    print(classification_report(y_test[:len(bert_predicted)], bert_predicted))

    # confusion_matrix = sklearn.metrics.confusion_matrix(y_test[:len(bert_predicted)], bert_predicted, labels=n_categories)
    # seaborn.heatmap(confusion_matrix, annot=True, fmt='d', cmap='seismic', square=True)
    # fig1 = plt.gcf()
    # plt.show()
    # plt.draw()
    #
    # fig1.savefig(model_directory + '/roberta_predict.png')
def plot_train_logs(t_losses, v_losses):
    # plot training progress
    plt.plot(t_losses)
    plt.plot(v_losses)
    plt.title('loss=loss_fn')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    # plt.ylim(bottom=0, top=3)
    plt.show()
    plt.draw()

    fig1 = plt.gcf()

    fig1.savefig(model_directory + '/training_log.png')
    #plt.close()

def indicize_labels(labels):
    categories = sorted(list(set(y_train)))
    n_categories = len(categories)
    """Transforms string labels into indices"""
    indices=[]
    for j in range(len(labels)):
        for i in range(n_categories):
            if labels[j]==categories[i]:
                indices.append(i)
    return indices, n_categories

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(train_base_path)

    batch_size = 8
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    train_labels, n_categories = indicize_labels(y_train)
    val_labels, _ = indicize_labels(y_val)
    test_labels , _= indicize_labels(y_test)
    train_dataloader = RobertoDataLoader(tokenizer, "_train", train_labels, batch_size, 512)
    val_dataloader = RobertoDataLoader(tokenizer, "_val", val_labels, batch_size, 512)
    test_dataloader = RobertoDataLoader(tokenizer, "_test", test_labels, batch_size, 512)

    print("1")
    model, optimizer, epocs = create_model()
    print("2")
    model.load_state_dict(torch.load(train_base_path + file_name))
    model.eval()
    train_losses, valid_losses = train_epocs(epocs, model, train_dataloader, val_dataloader, val_labels)
    plot_train_logs(train_losses, valid_losses)
   #  model.load_state_dict(torch.load(train_base_path + file_name))
   #  model.eval()
    predict(model, test_dataloader, test_labels)
    #train_epocs(epocs, model, val_dataloader, y_val)



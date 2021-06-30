#### Robotic activity (surgeme) classification using LSTM with PyTorch
# Adapted from PyTorch Sample here: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
import numpy as np
from sklearn.preprocessing import scale
from mln import pracmln_inference, getformatedweights, getformatedweights_gym
import time

torch.manual_seed(1)

data_dir = "../logic_constraints"

############ config ##############
ROBOT = "Taurus" # Gym/Taurus
SEGMENTED = True
COMBINED_TYPE = "conflation"
iscombined = True
early_frac = 0.5
total_epoch = 50

MLNLAYER = True
train_count = 1405
resultfile_train = data_dir+"/results/mlayer_"+str(MLNLAYER)+"_"+ROBOT+"_results_train_"+str(train_count)+".txt"
resultfile_test = data_dir+"/results/mlayer_"+str(MLNLAYER)+"_"+ROBOT+"_results_test_"+str(train_count)+".txt"

##########################
# Prepare data:

def prepare_sequence(seq):
    return torch.tensor(seq, dtype=torch.float)
def prepare_class_sequence(seq):
    return torch.tensor(seq, dtype=torch.long)


def data_preparation_xy(xyfile, isScale=True):
    data_surgeme = []
    features = np.loadtxt(xyfile, delimiter=' ')
    X, y = features[:, :-1], features[:, -1]
    if isScale:
        X = scale(X, axis=0, with_mean=True, with_std=True, copy=True )  # normalize
    X = X.tolist()
    y = y.tolist()
    seg = []
    labels = []
    prev_label = y[0] # the first label considered as previous label at first
    for i in range(len(y)):
        current_label = y[i]
        if current_label != prev_label:
            data_surgeme.append((seg, labels))
            seg = []
            labels = []
        seg.append(X[i])
        labels.append(current_label)
        prev_label = current_label
    data_surgeme.append((seg, labels)) # for the last
    return data_surgeme

def sample_frame(listfeature, sn=2):
    sampledlistfeature = []
    for i in range(len(listfeature)):
        if i % sn == 0:
            sampledlistfeature.append(listfeature[i])
    sampledlistfeature = np.asarray(sampledlistfeature)
    return sampledlistfeature
def gym_data_preparation_xy(xyfile, sn=1):
    data_surgeme = []
    features = np.loadtxt(xyfile, delimiter=' ')
    features = sample_frame(features, sn) # sample frame

    X, y = features[:, :-1], features[:, -1]
    X = scale(X, axis=0, with_mean=True, with_std=True, copy=True )  # normalize
    X = X.tolist()
    y = y.tolist()
    seg = []
    labels = []
    prev_label = 0
    for i in range(len(y)):
        current_label = y[i] # - 1 for gym # to match with index 0
        if current_label != prev_label:
            data_surgeme.append((seg, labels))
            seg = []
            labels = []
        seg.append(X[i])
        labels.append(current_label)
        prev_label = current_label
    data_surgeme.append((seg, labels)) # for the last
    return data_surgeme

def data_preparation_noscale_gym(dfile, sn=1):
    data_surgeme = []
    features = np.loadtxt(data_dir+"/"+dfile, delimiter=' ')
    # features = features.copy()
    
    features = sample_frame(features, sn) # sample frame

    X, y = features[:, :-1], features[:, -1]
    # X = scale(X, axis=0, with_mean=True, with_std=True, copy=True )  # normalize
    # X = np.copy(X)
    # print(X[0])
    X = X.tolist()
    # print(X[0])
    y = y.tolist()
    seg = []
    labels = []
    prev_label = 0
    for i in range(len(y)):
        current_label = y[i]  # gym already indexing.
        if current_label != prev_label:
            data_surgeme.append((seg, labels))
            seg = []
            labels = []
        seg.append(X[i])
        labels.append(current_label)
        prev_label = current_label
    data_surgeme.append((seg, labels)) # for the last
    return data_surgeme

def toMLNformat(lstmdata):
    mlndata = []
    for i in range(len(lstmdata)):
        segment, labels = lstmdata[i]
        for j in range(len(segment)):
            row = (segment[j]).copy()
            row.append(labels[j])
            mlndata.append(row)
    return mlndata
def datastat(sdata):
    print(len(sdata))
    countclass = [0] * 7
    countinstance = [0] * 7
    for i in range(len(sdata)):
        row = sdata[i][1]
        # print(len(row))
        # print(row)
        # print(len(row), int(row[0]))
        # print(countclass)
        countclass[int(row[0])] += len(row)
        countinstance[int(row[0])] += 1
    print(countclass)
    print(countinstance)
    for i in range(len(countclass)):
        print(i, countclass[i]/countinstance[i])


def countframe(tdata):
    totalframe = 0
    for i in range(len(tdata)):
        elem = tdata[i][0]
        totalframe += len(elem)
        # print(elem)
        # print(totalframe, len(elem))
    return totalframe

# data processing
if ROBOT == "Taurus":
    train_file = data_dir+"/taurus_kinematics_train.txt"
    test_file = data_dir+"/taurus_kinematics_test.txt"
    train_constraint_file = data_dir+"/taurus_kinematics_train_motion_features.txt"
    test_constraint_file = data_dir+"/taurus_kinematics_test_motion_features.txt"
elif ROBOT == "Yumi":
    train_file = data_dir+"/yumi_kinematics_train.txt"
    test_file = data_dir+"/yumi_kinematics_test.txt"
    train_constraint_file = data_dir+"/yumi_kinematics_train_motion_features.txt"
    test_constraint_file = data_dir+"/yumi_kinematics_test_motion_features.txt"
elif ROBOT == "Gym":
    train_file = data_dir+"/gym_kinematics_train.txt"
    test_file = data_dir+"/gym_kinematics_test.txt"
    train_constraint_file = data_dir+"/gym_kinematics_train_motion_features.txt"
    test_constraint_file = data_dir+"/gym_kinematics_test_motion_features.txt"

train_data = data_preparation_xy(train_file, isScale=True)
train_data_constraint = data_preparation_xy(train_constraint_file, isScale=False)

# if ROBOT == 'Gym':
#     count = train_count
#     train_data = train_data[:count]
#     train_data_constraint = train_data_constraint[:count]
# else: # for test of taurus
#     count = train_count
#     train_data = train_data[:count]
#     train_data_constraint = train_data_constraint[:count]

test_data = data_preparation_xy(test_file, isScale=True)
test_data_constraint = data_preparation_xy(test_constraint_file, isScale=False)

print("Data stats", "Robot", ROBOT)
print("train", len(train_data), "train-frames", countframe(train_data), "test", len(test_data), 'test-frames', countframe(test_data))
print("train-motion", len(train_data_constraint), "test-motion", len(test_data_constraint))
# sys.exit()
if ROBOT == "Gym":
    surgeme_class = [0, 1, 2, 3] # 3 surgeme class for gym
else:
    surgeme_class = [0, 1, 2, 3, 4, 5, 6] # 7 surgeme class

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
seg, labels = train_data[0]
FEATURE_DIM = len(seg[0]) # example: 14
HIDDEN_DIM = 32

######################################################################
# Create the model:
class LSTMClassifier_mlnlayer(nn.Module):
    def __init__(self, feature_dim, hidden_dim, class_size):
        super(LSTMClassifier_mlnlayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(feature_dim, hidden_dim) # The LSTM takes kinematics features as inputs, and outputs hidden states with dimensionality hidden_dim.
        self.hidden2tag = nn.Linear(hidden_dim, class_size) # The linear layer that maps from hidden state space to tag space
        # self.class2class = nn.Linear(class_size, class_size) # give learned class constraints as input here
        self.class2class = nn.Linear(class_size*2, class_size) # w1*h2c + w2* c2c
    def forward(self, surgeme_segment, constraints_score):
        features = surgeme_segment # change if any formating on the data required
        lstm_out, _ = self.lstm(features.view(len(features), 1, -1))
        class_space = self.hidden2tag(lstm_out.view(len(features), -1))
        # for constraitns bias
        # class_space_constraints = self.class2class(class_space+constraints_score) # this also a way learn weights of the concatenation. Check and update paper. weights can be learned here, instead of + (concatenation) using a linear layer: nn.Linear(class_space, constraints_score)
        # print(class_space.size(), constraints_score.size())
        combined_score = torch.cat((class_space, constraints_score), 1)
        # print('combined_score', combined_score.size())
        class_space_constraints = self.class2class(combined_score) # this also a way learn weights of the concatenation. Check and update paper. weights can be learned here, instead of + (concatenation) using a linear layer: nn.Linear(class_space, constraints_score)
        class_scores = F.log_softmax(class_space_constraints, dim=1)
        return class_scores
    
    # def forward(self, surgeme_segment, constraints_score):
    #     features = surgeme_segment # change if any formating on the data required
    #     lstm_out, _ = self.lstm(features.view(len(features), 1, -1))
    #     class_space = self.hidden2tag(lstm_out.view(len(features), -1))
    #     # for constraitns bias
    #     class_space_constraints = self.class2class(class_space+constraints_score) # this also a way learn weights of the concatenation. Check and update paper. weights can be learned here, instead of + (concatenation) using a linear layer: nn.Linear(class_space, constraints_score)
    #     class_scores = F.log_softmax(class_space_constraints, dim=1)
    #     return class_scores
class LSTMClassifier(nn.Module):
    def __init__(self, feature_dim, hidden_dim, class_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(feature_dim, hidden_dim) # The LSTM takes kinematics features as inputs, and outputs hidden states with dimensionality hidden_dim.
        self.hidden2tag = nn.Linear(hidden_dim, class_size) # The linear layer that maps from hidden state space to tag space
    def forward(self, surgeme_segment):
        features = surgeme_segment # change if any formating on the data required
        lstm_out, _ = self.lstm(features.view(len(features), 1, -1))
        class_space = self.hidden2tag(lstm_out.view(len(features), -1)) # can add another linear layer to be in same layer size as the mlnlayer network, ablation study
        class_scores = F.log_softmax(class_space, dim=1)
        return class_scores

def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if int(truth[i]) == int(pred[i]):
            right += 1.0
    return right/len(truth)
def get_class_accuracy(truth, pred, labelcount=7):
    correctcount = [0] * labelcount
    totalcount = [0] * labelcount
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if int(truth[i]) == int(pred[i]):
            right += 1.0
            correctcount[int(truth[i])] += 1
        totalcount[int(truth[i])] += 1
    classacc = [0.0] * labelcount
    for i in range(len(totalcount)):
        if totalcount[i] >0:
            classacc[i] = correctcount[i]/totalcount[i]
        else:
            classacc[i] = 0.0
    return classacc, totalcount
def evaluate(model, tdata):
    total_acc = 0.0
    for segment, labels in tdata:
        segment_in = prepare_sequence(segment)
        # targets = prepare_class_sequence(labels)
        targets = labels
        class_scores = model(segment_in)
        # print("\n==============\n")
        # print(class_scores) 
        pred_label = class_scores.data.max(1)[1].numpy()
        # print(pred_label, targets)
        acc = get_accuracy(targets, pred_label)
        # print(acc)
        total_acc += acc
    avg_acc = total_acc / len(tdata) # check for consistency
    # print(avg_acc)
    return avg_acc
def evaluate_sequence(model, tdata, position_frac):
    
    total_acc = 0.0
    pred_seq = []
    gt_seq = []
    for segment, labels in tdata:
        segment_in = prepare_sequence(segment)
        # targets = prepare_class_sequence(labels)
        targets = labels
        class_scores = model(segment_in)
        # print("\n==============\n")
        # print(class_scores) 
        pred_label = class_scores.data.max(1)[1].numpy()
        # print(pred_label, targets)
        backindex = int(len(pred_label) * position_frac) - 1 # -1 for 0 indexing
        pred_seq.append(pred_label[backindex]) # take last prediction
        gt_seq.append(targets[backindex]) # take last gt
        # print(acc)
        # total_acc += acc
    print(len(gt_seq), len(pred_seq))
    avg_acc = get_accuracy(gt_seq, pred_seq)
    print(avg_acc)
    return avg_acc
def combined_classprob(lstmclassprobs, mlnclassprobs, lambdaparam=0.5):
    # both prob in log scale so any mathematical operation is possible directly
    # both numpy array
    # print(lstmclassprobs.shape, mlnclassprobs.shape)
    part1 = np.asarray((1-lambdaparam) * lstmclassprobs)
    part2 = np.asarray(lambdaparam * mlnclassprobs)
    combined = np.asarray(part1 + part2) # weighted average
    return combined
# https://stats.stackexchange.com/questions/194878/combining-two-probability-scores
def combined_classprob_confidence(lstmclassprobs, mlnclassprobs, confidence_score_lstm, confidence_score_mln, targets, lambdaparam=0.5):
    assert(len(lstmclassprobs)==len(confidence_score_lstm))
    assert(len(mlnclassprobs)==len(confidence_score_mln))
    prev_lstm_prediction = lstmclassprobs[0].argmax()
    combined = []
    for i in range(len(lstmclassprobs)):
        # print("GT: ", targets[i])
        # print("LSTMPron: ", lstmclassprobs[i])
        # print("LSTM Prediction ", lstmclassprobs[i].argmax())
        # print("LSTMConf: ", confidence_score_lstm[i])
        # print("MLNPron: ", mlnclassprobs[i])
        # print("MLN Prediction ", mlnclassprobs[i].argmax())
        # print("MLNConf: ", confidence_score_mln[i], confidence_score_mln[i].item())
        history_weights = [0] * 7
        for j in range(len(lstmclassprobs[0])): # length sequence, assuming 0-1-2-3-4-5-6 sequence
            # distance = abs(prev_lstm_prediction-j) + 1
            
            history_weights[(j+prev_lstm_prediction)%7] = 1/(j+1) # use gaussian style distribution
        # print(history_weights)
        history_weights = np.asarray(history_weights)
        sumval = np.sum(history_weights)
        history_prob = history_weights / sumval
        # history_prob = np.log(history_prob)
        combinedi = []
        normal_prob = np.exp(lstmclassprobs[i])
        # print(normal_prob, lstmclassprobs[i], np.sum(normal_prob))
        
        # for k in range(len(history_prob)):
        #     hp = history_prob[k]
        #     lp = normal_prob[k]
        #     conflation_p = (hp * lp)/(hp*lp + (1-hp)*(1-lp))
        #     combinedi.append(conflation_p)
        normal_prob_mln = np.exp(mlnclassprobs[i])

        output_constraint_prob = [0.125] * 7
        if prev_lstm_prediction == 0:
            index1 = 6
            index2 = 0
            index3 = 1
        elif prev_lstm_prediction == 6:
            index1 = 5
            index2 = 6
            index3 = 0
        else:
            index1 = prev_lstm_prediction - 1
            index2 = prev_lstm_prediction
            index3 = prev_lstm_prediction + 1
        output_constraint_prob[index1] = 0.167
        output_constraint_prob[index2] = 0.167
        output_constraint_prob[index3] = 0.167

        # print(normal_prob_mln, np.sum(normal_prob_mln))
        # print(output_constraint_prob, sum(output_constraint_prob))
        normal_prob_mln = output_constraint_prob # manually changing it to check
        combinedi_lstm_mln = []
        for k in range(len(mlnclassprobs[i])):
            hp = normal_prob_mln[k]
            lp = normal_prob[k]
            conflation_p = (hp * lp)/(hp*lp + (1-hp)*(1-lp))
            combinedi_lstm_mln.append(conflation_p)
        # for k in range(len(combinedi_lstm_mln)):
        #     hp = history_prob[k]
        #     lp = combinedi_lstm_mln[k]
        #     conflation_p = (hp * lp)/(hp*lp + (1-hp)*(1-lp))
        #     combinedi.append(conflation_p)
                
        # sys.exit()
        hlambda = 0.5
        # print(history_weights)
        # make it normal probabilty and then use "conflation" https://stats.stackexchange.com/questions/194878/combining-two-probability-scores
        
        # combinedi_history = (1-hlambda) * lstmclassprobs[i] + hlambda * history_prob
        
        # print(targets[i], lstmclassprobs[i].argmax(), confidence_score_lstm[i].item(), mlnclassprobs[i].argmax(), confidence_score_mln[i].item(), combinedi_history.argmax())
        
        # if confidence_score_mln[i].item() < 0.10: # low confidence of MLN
        #     lambdaparam = 0.0 # zero weight on MLN prob
        # if confidence_score_lstm[i].item() > 0.5: # use only LSTM for high confidence
        #     hlambda = 0.0
        #     lambdaparam = 0.0
        # combinedi = (1-lambdaparam) * lstmclassprobs[i] + lambdaparam * mlnclassprobs[i] + hlambda * history_prob
        
        combined.append(combinedi_lstm_mln)
        # combined.append(combinedi.tolist())
        # combined.append(combinedi_history.tolist())
        prev_lstm_prediction = np.asarray(combinedi_lstm_mln).argmax()
        # print(type(combinedi))
        # sys.exit()    
        # print("Combined Prediction ", combinedi.argmax())
    # part1 = np.asarray((1-lambdaparam) * lstmclassprobs)
    # part2 = np.asarray(lambdaparam * mlnclassprobs)
    combined = np.asarray(combined)
    return combined
def combined_classprob_conflation(lstmclassprobs, mlnclassprobs):
    assert(len(lstmclassprobs)==len(mlnclassprobs))
    combined = []
    for i in range(len(lstmclassprobs)):
        normal_prob = np.exp(lstmclassprobs[i]) # make it to non-log probability
        normal_prob_mln = np.exp(mlnclassprobs[i]) # make it to non-log probability
        # print(normal_prob_mln, np.sum(normal_prob_mln))
        # calculate conflation
        combinedi_lstm_mln = []
        for k in range(len(mlnclassprobs[i])):
            hp = normal_prob_mln[k]
            lp = normal_prob[k]
            conflation_p = (hp * lp)/(hp*lp + (1-hp)*(1-lp))
            combinedi_lstm_mln.append(conflation_p)
        combined.append(combinedi_lstm_mln)
    combined = np.asarray(combined)
    return combined
def evaluate_sequence_mln(model, tdata, tdata_mln, mln_model_classprob, position_frac, iscombined=False):
    # print(len(tdata), len(tdata_mln))
    # print(type(tdata), type(tdata_mln))
    total_acc = 0.0
    pred_seq = []
    gt_seq = []
    pred_seq_mln = []
    total_framewise_avg_acc_lstm = 0.0
    total_framewise_avg_acc_mln = 0.0
    total_framewise_avg_acc_lstm_mln = 0.0

    gt_seq = []
    pred_seq_lstm = []
    pred_seq_mln = []
    pred_seq_lstm_mln = []

    early_gt_seq = []
    early_pred_seq_lstm = []
    early_pred_seq_mln = []
    early_pred_seq_lstm_mln = []
    # for segment, labels in tdata:

    for i in range(len(tdata)):
        segment, labels = tdata[i]
        segment_noscale, l2 = tdata_mln[i]
        # print("Eval", i, len(segment), len(segment_noscale))
        if len(segment) == 0:
            continue
        segment_in = prepare_sequence(segment)
        segment_in_noscale = prepare_sequence(segment_noscale)
        # targets = prepare_class_sequence(labels)
        targets = labels
        # print("GT: ", targets)
        
        # calculate constraints class weight
        mln_preds = []
        mln_classprobs = []
        mln_weights = []
        if iscombined:
            mln_preds = []
            mln_classprobs = []
            count = 0
            for single_segment_in in segment_in_noscale:
                single_segment_in = single_segment_in.data.numpy()
                gt = targets[count]
                count += 1

                mlnpred_label, classweight_combined, classprobability = pracmln_inference(formula_weight, single_segment_in, robot=ROBOT)

                mln_preds.append(mlnpred_label)
                mln_classprobs.append(classprobability)
                mln_weights.append(classweight_combined)

        mln_classprobs = torch.FloatTensor(mln_classprobs)
        mln_weights = torch.FloatTensor(mln_weights)
        if MLNLAYER:
            class_scores = model(segment_in, mln_weights)
        else:
            class_scores = model(segment_in)
        # print("\n==============\n")
        # print(class_scores)
        if iscombined:
            mln_preds = []
            mln_classprobs = []
            count = 0
            for single_segment_in in segment_in_noscale:
                single_segment_in = single_segment_in.data.numpy()
                gt = targets[count]
                # print ("Ground Truth: "+str(gt))
                # print("Single data: ")
                count += 1
                mlnpred_label, classweight_combined, classprobability = pracmln_inference(formula_weight, single_segment_in, robot=ROBOT)
                mln_preds.append(mlnpred_label)
                mln_classprobs.append(classprobability)
            # calculate confidence score, required if any combined appraoch depends on confidence score
            topk = torch.topk(class_scores, 2)
            mln_classprobs_T = torch.FloatTensor(mln_classprobs)
            topk_mln = torch.topk(mln_classprobs_T, 2)
            val, ind = topk
            confidence_score_lstm = []
            confidence_score_mln = []
            
            for i in range(len(topk)):
                if i%2 != 0: # skipping one entry, that entry is not part of the probability that is required for condfidence score calculation
                    continue
                tval = topk[i]
                tval_mln = topk_mln[i]
                for j in range(len(tval)):
                    # print(tval[j])
                    val1 = tval[j][0]
                    val2 = tval[j][1]
                    conf = val1 - val2
                    confidence_score_lstm.append(conf)
                    # print(tval_mln[j])
                    val1_mln = tval_mln[j][0]
                    val2_mln = tval_mln[j][1]
                    conf_mln = val1_mln - val2_mln
                    confidence_score_mln.append(conf_mln)
            mln_classprobs = np.asarray(mln_classprobs)
            
            # combinedclassscore = combined_classprob_confidence(class_scores.detach().numpy(), mln_classprobs, confidence_score_lstm, confidence_score_mln, targets, lambdaparam=0.5)
            if COMBINED_TYPE == "conflation":
                combinedclassscore = combined_classprob_conflation(class_scores.detach().numpy(), mln_classprobs)
            else: # weighted average
                combinedclassscore = combined_classprob(class_scores.detach().numpy(), mln_classprobs, lambdaparam=0.5)
            combinedclassscore = torch.from_numpy(combinedclassscore) # convert numpy to torch type
            mln_classscore = torch.from_numpy(mln_classprobs) # convert numpy to torch type
        else: # dummy LSTM score
            combinedclassscore = class_scores
            mln_classscore = class_scores
        
        
        pred_label_lstm = class_scores.data.max(1)[1].numpy()
        pred_label_mln = mln_classscore.data.max(1)[1].numpy()
        pred_label_lstm_mln = combinedclassscore.data.max(1)[1].numpy() # combined
        # print(pred_label, targets)
        # print(class_scores, mln_classprobs)

        # print("Checking")
        # print(targets)
        # print(pred_label_lstm)
        # print(pred_label_mln)
        # print(pred_label_lstm_mln)

        # combined results for a batch
        gt_seq.extend(targets)
        pred_seq_mln.extend(pred_label_mln) # mln_preds variable might not required
        pred_seq_lstm.extend(pred_label_lstm)
        pred_seq_lstm_mln.extend(pred_label_lstm_mln) # combined

        # early sequence, take i th prediction
        backindex = int(len(targets) * position_frac) - 1 #early index,  -1 for 0 indexing
        early_gt_seq.append(targets[backindex])
        early_pred_seq_lstm.append(pred_label_lstm[backindex])
        early_pred_seq_mln.append(pred_label_mln[backindex])
        early_pred_seq_lstm_mln.append(pred_label_lstm_mln[backindex])

        framewise_avg_acc_mln = get_accuracy(targets, pred_label_mln)
        framewise_avg_acc_lstm = get_accuracy(targets, pred_label_lstm)
        framewise_avg_acc_lstm_mln = get_accuracy(targets, pred_label_lstm_mln)

        total_framewise_avg_acc_mln += framewise_avg_acc_mln
        total_framewise_avg_acc_lstm += framewise_avg_acc_lstm
        total_framewise_avg_acc_lstm_mln += framewise_avg_acc_lstm_mln
    # print(len(gt_seq), len(pred_seq))
    # print(len(gt_seq), len(pred_seq_mln))

    class_framewise_avg_acc_mln, classcount = get_class_accuracy(gt_seq, pred_seq_mln, labelcount=len(surgeme_class))
    class_framewise_avg_acc_lstm, _ = get_class_accuracy(gt_seq, pred_seq_lstm, labelcount=len(surgeme_class))
    class_framewise_avg_acc_lstm_mln, _ = get_class_accuracy(gt_seq, pred_seq_lstm_mln, labelcount=len(surgeme_class))

    final_framewise_avg_acc_mln = total_framewise_avg_acc_mln / len(tdata)
    final_framewise_avg_acc_lstm = total_framewise_avg_acc_lstm / len(tdata)
    final_framewise_avg_acc_lstm_mln = total_framewise_avg_acc_lstm_mln / len(tdata)

    early_avg_acc_lstm = get_accuracy(early_gt_seq, early_pred_seq_lstm)
    early_avg_acc_mln = get_accuracy(early_gt_seq, early_pred_seq_mln)
    early_avg_acc_lstm_mln = get_accuracy(early_gt_seq, early_pred_seq_lstm_mln)

    early_class_framewise_avg_acc_mln, early_classcount = get_class_accuracy(early_gt_seq, early_pred_seq_mln, labelcount=len(surgeme_class))
    early_class_framewise_avg_acc_lstm, _ = get_class_accuracy(early_gt_seq, early_pred_seq_lstm, labelcount=len(surgeme_class))
    early_class_framewise_avg_acc_lstm_mln, _ = get_class_accuracy(early_gt_seq, early_pred_seq_lstm_mln, labelcount=len(surgeme_class))


    return early_avg_acc_lstm, early_avg_acc_mln, early_avg_acc_lstm_mln, final_framewise_avg_acc_lstm, final_framewise_avg_acc_mln, final_framewise_avg_acc_lstm_mln, class_framewise_avg_acc_lstm, class_framewise_avg_acc_mln, class_framewise_avg_acc_lstm_mln, classcount, early_class_framewise_avg_acc_lstm, early_class_framewise_avg_acc_mln, early_class_framewise_avg_acc_lstm_mln, early_classcount
######################################################################
# Train the model:
if ROBOT == "Gym":
    wfile = data_dir+"/gym.w"
    formula_weight = getformatedweights_gym(wfile)
elif ROBOT == "Taurus":
    wfile = data_dir+"/taurus.w"
    formula_weight = getformatedweights(wfile)
elif ROBOT == "Yumi":
    wfile = data_dir+"/yumi.w"
    formula_weight = getformatedweights(wfile)
else:
    wfile = data_dir+"/taurus.w"
    formula_weight = getformatedweights(wfile)
if MLNLAYER:
    model = LSTMClassifier_mlnlayer(FEATURE_DIM, HIDDEN_DIM, len(surgeme_class))
else:
    model = LSTMClassifier(FEATURE_DIM, HIDDEN_DIM, len(surgeme_class)) # no 

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# logging results

fw = open(resultfile_test, "a")
fw.write(str("MLNLAYER")+"\t"+str("COMBINED_TYPE")+"\t"+str("epoch")+"\t"+str("frac")+"\t"+str("early_avg_acc_lstm")+"\t"+str("early_avg_acc_mln")+"\t"+str("early_avg_acc_lstm_mln")+"\t"+str("final_framewise_avg_acc_lstm")+"\t"+str("final_framewise_avg_acc_mln")+"\t"+str("final_framewise_avg_acc_lstm_mln")+"\t"+str("class_framewise_avg_acc_lstm")+"\t"+str("class_framewise_avg_acc_mln")+"\t"+str("class_framewise_avg_acc_lstm_mln")+"\t"+str("classcount")+"\t"+str("early_class_framewise_avg_acc_lstm")+"\t"+str("early_class_framewise_avg_acc_mln")+"\t"+str("early_class_framewise_avg_acc_lstm_mln")+"\t"+str("early_classcount")+"\n")
fw.close()

fw = open(resultfile_train, "a")
fw.write(str("MLNLAYER")+"\t"+str("COMBINED_TYPE")+"\t"+str("epoch")+"\t"+str("frac")+"\t"+str("early_avg_acc_lstm")+"\t"+str("early_avg_acc_mln")+"\t"+str("early_avg_acc_lstm_mln")+"\t"+str("final_framewise_avg_acc_lstm")+"\t"+str("final_framewise_avg_acc_mln")+"\t"+str("final_framewise_avg_acc_lstm_mln")+"\t"+str("class_framewise_avg_acc_lstm")+"\t"+str("class_framewise_avg_acc_mln")+"\t"+str("class_framewise_avg_acc_lstm_mln")+"\t"+str("classcount")+"\t"+str("early_class_framewise_avg_acc_lstm")+"\t"+str("early_class_framewise_avg_acc_mln")+"\t"+str("early_class_framewise_avg_acc_lstm_mln")+"\t"+str("early_classcount")+"\n")
fw.close()

total_time_diff = 0
for epoch in range(total_epoch):  # again, normally you would NOT do 300 epochs, it is toy data
    loss = 0
    icount = 0
    
    start_time = time.time()
    for segment, labels in train_data:
        
        segment_noscale, l2 = train_data_constraint[icount] # motion/constraints data - to generate formula
        icount += 1
        # print(icount, "Training", len(segment), len(segment_noscale))
        segment_in = prepare_sequence(segment)
        segment_in_noscale = prepare_sequence(segment_noscale)
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        segment_in = prepare_sequence(segment)
        targets = prepare_class_sequence(labels)

        mln_preds = []
        mln_classprobs = []
        mln_weights = []
        count = 0
        # mln inference
        for single_segment_in in segment_in_noscale:
            single_segment_in = single_segment_in.data.numpy()
            gt = targets[count]
            count += 1

            mlnpred_label, classweight_combined, classprobability = pracmln_inference(formula_weight, single_segment_in, robot=ROBOT)
            mln_preds.append(mlnpred_label)
            mln_classprobs.append(classprobability)
            mln_weights.append(classweight_combined)

        mln_classprobs = torch.FloatTensor(mln_classprobs)
        mln_weights = torch.FloatTensor(mln_weights)  
        if MLNLAYER:
            class_scores = model(segment_in, mln_weights)
        else:
            class_scores = model(segment_in)
        loss = loss_function(class_scores, targets)
        loss.backward()
        optimizer.step()
    end_time = time.time()
    elapsed = end_time - start_time
    print(start_time, end_time)
    print ("Epoch: ", epoch, " Time required for training only (in second): ", elapsed)
    total_time_diff += elapsed
    print("Average time per epoch: ", total_time_diff/(epoch+1))
    for i in range(1, 10):
        frac = i/10.0
        if frac != early_frac:
            continue
        # '''
        early_avg_acc_lstm, early_avg_acc_mln, early_avg_acc_lstm_mln, final_framewise_avg_acc_lstm, final_framewise_avg_acc_mln, final_framewise_avg_acc_lstm_mln, class_framewise_avg_acc_lstm, class_framewise_avg_acc_mln, class_framewise_avg_acc_lstm_mln, classcount, early_class_framewise_avg_acc_lstm, early_class_framewise_avg_acc_mln, early_class_framewise_avg_acc_lstm_mln, early_classcount = evaluate_sequence_mln(model, test_data, test_data_constraint, formula_weight, position_frac=frac, iscombined=iscombined)
        early_avg_acc_lstm_testing = early_avg_acc_lstm
        final_framewise_avg_acc_lstm_testing = final_framewise_avg_acc_lstm

        fw = open(resultfile_test, "a")
        # fw.write(str(epoch)+"\t"+str(frac)+"\t"+str(early_avg_acc_lstm)+"\t"+str(early_avg_acc_mln)+"\t"+str(early_avg_acc_lstm_mln)+"\t"+str(final_framewise_avg_acc_lstm)+"\t"+str(final_framewise_avg_acc_mln)+"\t"+str(final_framewise_avg_acc_lstm_mln)+"\n")
        fw.write(str(MLNLAYER)+"\t"+str(COMBINED_TYPE)+"\t"+str(epoch)+"\t"+str(frac)+"\t"+str(early_avg_acc_lstm)+"\t"+str(early_avg_acc_mln)+"\t"+str(early_avg_acc_lstm_mln)+"\t"+str(final_framewise_avg_acc_lstm)+"\t"+str(final_framewise_avg_acc_mln)+"\t"+str(final_framewise_avg_acc_lstm_mln)+"\t"+str(class_framewise_avg_acc_lstm)+"\t"+str(class_framewise_avg_acc_mln)+"\t"+str(class_framewise_avg_acc_lstm_mln)+"\t"+str(classcount)+"\t"+str(early_class_framewise_avg_acc_lstm)+"\t"+str(early_class_framewise_avg_acc_mln)+"\t"+str(early_class_framewise_avg_acc_lstm_mln)+"\t"+str(early_classcount)+"\n")
        fw.close()

        
        # '''
        #training

        early_avg_acc_lstm, early_avg_acc_mln, early_avg_acc_lstm_mln, final_framewise_avg_acc_lstm, final_framewise_avg_acc_mln, final_framewise_avg_acc_lstm_mln, class_framewise_avg_acc_lstm, class_framewise_avg_acc_mln, class_framewise_avg_acc_lstm_mln, classcount, early_class_framewise_avg_acc_lstm, early_class_framewise_avg_acc_mln, early_class_framewise_avg_acc_lstm_mln, early_classcount = evaluate_sequence_mln(model, train_data, train_data_constraint, formula_weight, position_frac=frac, iscombined=iscombined)
        # print(epoch, frac, loss.item(), early_avg_acc_lstm, early_avg_acc_lstm_testing, final_framewise_avg_acc_lstm, final_framewise_avg_acc_lstm_testing)
        
        # fw = open(data_dir+"/train_taurus_bias_mln_weigh_"+str(int(train_test_frac*100))+"_epoch50.txt", "a")
        fw = open(resultfile_train, "a")
        # fw.write(str(epoch)+"\t"+str(frac)+"\t"+str(loss.item())+"\t"+str(early_avg_acc_lstm)+"\t"+str(early_avg_acc_lstm_testing)+"\t"+str(final_framewise_avg_acc_lstm)+"\t"+str(final_framewise_avg_acc_lstm_testing)+"\n")
        fw.write(str(MLNLAYER)+"\t"+str(COMBINED_TYPE)+"\t"+str(epoch)+"\t"+str(frac)+"\t"+str(early_avg_acc_lstm)+"\t"+str(early_avg_acc_mln)+"\t"+str(early_avg_acc_lstm_mln)+"\t"+str(final_framewise_avg_acc_lstm)+"\t"+str(final_framewise_avg_acc_mln)+"\t"+str(final_framewise_avg_acc_lstm_mln)+"\t"+str(class_framewise_avg_acc_lstm)+"\t"+str(class_framewise_avg_acc_mln)+"\t"+str(class_framewise_avg_acc_lstm_mln)+"\t"+str(classcount)+"\t"+str(early_class_framewise_avg_acc_lstm)+"\t"+str(early_class_framewise_avg_acc_mln)+"\t"+str(early_class_framewise_avg_acc_lstm_mln)+"\t"+str(early_classcount)+"\n")
        fw.close()




    

import sys
import numpy as np
from sklearn.preprocessing import scale

data_dir = "../constraints_dir"

def data_preparation(dfile):
    data_surgeme = []
    features = np.loadtxt(data_dir+"/"+dfile, delimiter=' ')
    X, y = features[:, :-1], features[:, -1]
    X = scale(X, axis=0, with_mean=True, with_std=True, copy=True )  # normalize
    X = X.tolist()
    y = y.tolist()
    seg = []
    labels = []
    prev_label = 0
    for i in range(len(y)):
        current_label = y[i] - 1 # to match with index 0
        if current_label != prev_label:
            data_surgeme.append((seg, labels))
            seg = []
            labels = []
        seg.append(X[i])
        labels.append(current_label)
        prev_label = current_label
    data_surgeme.append((seg, labels)) # for the last
    return data_surgeme
def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if int(truth[i]) == int(pred[i]):
            right += 1.0
    return right/len(truth), right

'''
mln_analysis: general version. Will be removed soon. Use mln_train() and mln_test() instead.
'''
def mln_analysis(train_features, test_features):
    train_features = np.asarray(train_features)
    test_features = np.asarray(test_features)
    print(len(train_features))
    print(train_features[0])
    print(len(train_features[0]))
    # current_arms_distance arm_dist lgs left_gs rgs right_gs labels
    F = ["f1"] # forumulas
    formula2class = {}
    # training learn class probabilty under each true form of the formula

    # testing/inference: take average (weighted?) of all the true forumlas

    # why mln work: ML cannot predict where it has less examples, thus when data is less MLN/constraints should give the accuracy a boost, increase transfer learning accuracy as MLN is invarient of any particular domain.
    arms_dist = train_features[:, 1]
    left_gs = train_features[:, 3]
    right_gs = train_features[:, 5]
    labels = train_features[:, 6] - 1
    print(len(arms_dist), len(left_gs), len(right_gs), len(labels))
    print(arms_dist[100:120])
    print(left_gs[100:120])
    print(right_gs[100:120])
    # use grippe status here
    print(labels[100:120])
    labelcount = [0] * 7
    for m in labels:
        # print(int(m))
        labelcount[int(m)] += 1
    print (labelcount)
    print(labelcount[0]+labelcount[1]+labelcount[2]+labelcount[3]+labelcount[4]+labelcount[5]+labelcount[6])
    for i in range(5):
        for j in range(5):
            fol_gs = (i, j)
            truecount = [0] * 7
            falsecount = labelcount.copy() # init with all false
            formula2class[fol_gs] = (truecount, falsecount)
            for k in range(3): # arms distance status
                fol_gs = (i, j, k)
                truecount = [0] * 7
                falsecount = labelcount.copy() # init with all false
                formula2class[fol_gs] = (truecount, falsecount)
    print(formula2class)
    print(len(formula2class))

    # training MLN
    for i in range(len(labels)):
        label = int(labels[i])
        fol = (left_gs[i], right_gs[i])
        # if (i+1) % int(len(labels)/2) == 0:
        #     print(formula2class)
        #     print(len(formula2class))
        if fol in formula2class:
            truecount, falsecount = formula2class[fol]
            truecount[label] += 1 # incease true
            falsecount[label] -= 1 # decrease false
            formula2class[fol] = (truecount, falsecount)
        else:
            print("Formulla not found, should not print")

        fol = (left_gs[i], right_gs[i], arms_dist[i])
        if fol in formula2class:
            truecount, falsecount = formula2class[fol]
            truecount[label] += 1 # incease true
            falsecount[label] -= 1 # decrease false
            formula2class[fol] = (truecount, falsecount)
        else:
            print("Formulla not found, should not print")
    print(len(formula2class))
    print(formula2class)

    # get the probability
    formula2class_prob = {}
    for fol in formula2class:
        truecount, falsecount = formula2class[fol]
        classprob = [0.0] * 7
        for i in range(len(truecount)):
            tcount = truecount[i]
            fcount = falsecount[i]
            if fcount == 0:
                classprob[i] = tcount # always exist
            else:
                classprob[i] = tcount / fcount
        formula2class_prob[fol] = classprob.copy()

    print(len(formula2class_prob))
    print(formula2class_prob)
    
    # for fol in formula2class_prob:
    #   print(fol, formula2class_prob[fol])

    # inference / testing
    arms_dist = test_features[:, 1]
    left_gs = test_features[:, 3]
    right_gs = test_features[:, 5]
    labels = test_features[:, 6] - 1
    pred_labels = []
    
    classwise_pred_labels = []
    classwise_pred_labels.append([])
    classwise_pred_labels.append([])
    classwise_pred_labels.append([])
    classwise_pred_labels.append([])
    classwise_pred_labels.append([])
    classwise_pred_labels.append([])
    classwise_pred_labels.append([])

    for i in range(len(arms_dist)):
        label = int(labels[i])
        fol1 = (left_gs[i], right_gs[i])
        fol2 = (left_gs[i], right_gs[i], arms_dist[i])
        classweight1 = [0.0] * 7
        classweight2 = [0.0] * 7
        
        if fol1 in formula2class_prob:
            classweight1 = formula2class_prob[fol1]
        if fol2 in formula2class_prob:
            classweight2 = formula2class_prob[fol2]

        classweight_combined = [0.0] * 7
        for j in range(len(classweight_combined)):
            classweight_combined[j] = (classweight1[j] + classweight2[j])/2 # average
            # print(classweight1, classweight2, classweight_combined)
        classweight_combined = np.asarray(classweight_combined)
        pred_label = np.argmax(classweight_combined)
        pred_labels.append(pred_label)
        classwise_pred_labels[int(labels[i])].append(pred_label)
        # print(pred_label, labels[i])

    testacc, correct = get_accuracy(labels, pred_labels)
    fw = open(data_dir+"/mln_yumi_results_labels_truth_pred.txt", "w")
    fw.write("Test accuracy: "+str(testacc))
    for i in range(len(labels)):
        fw.write(str(labels[i])+"\t"+str(pred_labels[i])+"\n")
    fw.close()
    print("testacc: ", testacc, "Correct: ", correct, "total instance: ", len(labels))
    classacc = []
    # classwise
    print(len(classwise_pred_labels[0]))
    for i in range(len(classwise_pred_labels)):
        pred = (classwise_pred_labels[i]).copy()
        gt = [i] * len(pred)
        
        acc, correct = get_accuracy(gt, pred)
        print(len(pred), correct)
        classacc.append(acc)
    print(classacc)

'''
mln_train
training formula weight

Input: training data to get forumla weight

Output: formula weight per class. Return all formula weight. If any formula never occurs then return 0 as its weight.
Todo: get variable and data range automatically from the training data. For a feature possible data value can be calculated which can be used for formula generation.

'''
def mln_train(train_features):
    train_features = np.asarray(train_features)  # convert to numpy array for better manipulation
    formula2class = {}

    arms_dist = train_features[:, 1]
    left_gs = train_features[:, 3]
    right_gs = train_features[:, 5]
    labels = train_features[:, 6] - 1
    labelcount = [0] * 7
    for m in labels:
        labelcount[int(m)] += 1
    for i in range(5):
        for j in range(5):
            fol_gs = (i, j)
            truecount = [0] * 7
            falsecount = labelcount.copy() # init with all false
            formula2class[fol_gs] = (truecount, falsecount)
            for k in range(3): # arms distance status
                fol_gs = (i, j, k)
                truecount = [0] * 7
                falsecount = labelcount.copy() # init with all false
                formula2class[fol_gs] = (truecount, falsecount)

    # training MLN
    for i in range(len(labels)):
        label = int(labels[i])
        fol = (left_gs[i], right_gs[i])
        if fol in formula2class:
            truecount, falsecount = formula2class[fol]
            truecount[label] += 1 # incease true
            falsecount[label] -= 1 # decrease false
            formula2class[fol] = (truecount, falsecount)
        else:
            print("Formulla not found, should not print")
        fol = (left_gs[i], right_gs[i], arms_dist[i])
        if fol in formula2class:
            truecount, falsecount = formula2class[fol]
            truecount[label] += 1 # incease true
            falsecount[label] -= 1 # decrease false
            formula2class[fol] = (truecount, falsecount)
        else:
            print("Formulla not found, should not print")
    # get the probability
    formula2class_prob = {}
    for fol in formula2class:
        truecount, falsecount = formula2class[fol]
        classprob = [0.0] * 7
        for i in range(len(truecount)):
            tcount = truecount[i]
            fcount = falsecount[i]
            if fcount == 0:
                classprob[i] = tcount # always exist
            else:
                classprob[i] = tcount / fcount
        formula2class_prob[fol] = classprob.copy()
    return formula2class_prob

'''
mln_inference
inference / testing
Input: mln_train model and test data
Output: accuracy, classwise_accuracy, class_probability per data point
'''
def mln_inference(formula2class_prob, test_features, debug=True):

    test_features = np.asarray(test_features)
    arms_dist = test_features[:, 1]
    left_gs = test_features[:, 3]
    right_gs = test_features[:, 5]
    labels = test_features[:, 6] - 1
    pred_labels = []
    
    classwise_pred_labels = [] # initialize
    classwise_pred_labels.append([])
    classwise_pred_labels.append([])
    classwise_pred_labels.append([])
    classwise_pred_labels.append([])
    classwise_pred_labels.append([])
    classwise_pred_labels.append([])
    classwise_pred_labels.append([])

    classweights = []
    classprobabilities = []
    for i in range(len(arms_dist)):
        label = int(labels[i])
        fol1 = (left_gs[i], right_gs[i])
        fol2 = (left_gs[i], right_gs[i], arms_dist[i])
        classweight1 = [0.0] * 7
        classweight2 = [0.0] * 7
        
        if fol1 in formula2class_prob:
            classweight1 = formula2class_prob[fol1]
        if fol2 in formula2class_prob:
            classweight2 = formula2class_prob[fol2]
        classweight_combined = [0.0] * 7
        for j in range(len(classweight_combined)):
            classweight_combined[j] = (classweight1[j] + classweight2[j])/2 # average
        classweight_combined = np.asarray(classweight_combined)
        # make it probability
        maxval = np.sum(classweight_combined)
        if maxval > 0:
            classprobability = classweight_combined / maxval
        else:
            classprobability = classweight_combined.copy()

        # print(classweight_combined, classprobabilities)
        # sys.exit()

        classweights.append(classweight_combined) # for return
        classprobabilities.append(classprobability)

        pred_label = np.argmax(classweight_combined)
        pred_labels.append(pred_label)
        classwise_pred_labels[int(labels[i])].append(pred_label)
    testacc, correct = get_accuracy(labels, pred_labels)
    if debug == True:
        fw = open(data_dir+"/mln_results_labels_truth_pred.txt", "w")
        fw.write("Test accuracy: "+str(testacc))
        for i in range(len(labels)):
            fw.write(str(labels[i])+"\t"+str(pred_labels[i])+"\n")
        fw.close()
    classacc = []
    for i in range(len(classwise_pred_labels)):
        pred = (classwise_pred_labels[i]).copy()
        gt = [i] * len(pred)
        acc, correct = get_accuracy(gt, pred)
        classacc.append(acc)
    return testacc, classacc, classweights, classprobabilities

def mln_inference_single(formula2class_prob, single_row_feature, debug=True):
    # print(single_row_feature)
    arms_dist = int(single_row_feature[1])
    left_gs = int(single_row_feature[3])
    right_gs = int(single_row_feature[5])
    # labels = single_row_feature[:, 6] - 1

    classwise_pred_labels = [] # initialize
    classwise_pred_labels.append([])
    classwise_pred_labels.append([])
    classwise_pred_labels.append([])
    classwise_pred_labels.append([])
    classwise_pred_labels.append([])
    classwise_pred_labels.append([])
    classwise_pred_labels.append([])

    fol1 = (left_gs, right_gs)
    fol2 = (left_gs, right_gs, arms_dist)
    # print(fol1, fol2)
    # sys.exit()
    classweight1 = [0.0] * 7
    classweight2 = [0.0] * 7
    
    if fol1 in formula2class_prob: # make formula searching robust so that we can handle float and other type of similarity between formulas
        classweight1 = formula2class_prob[fol1]
    if fol2 in formula2class_prob:
        classweight2 = formula2class_prob[fol2]
    classweight_combined = [0.0] * 7
    for j in range(len(classweight_combined)):
        classweight_combined[j] = (classweight1[j] + classweight2[j])/2 # average
    classweight_combined = np.asarray(classweight_combined)
    # make it probability
    classweight_combined_forlogprob = classweight_combined + 1 # 1 to remove zero prob, smooting

    sumval = np.sum(classweight_combined_forlogprob)
    # if maxval > 0:

    classprobability = classweight_combined_forlogprob / sumval
    # classprobability = np.asarray(classprobability)
    # print(type(classprobability))
    # classprobability = classprobability.astype(float)
    classprobability = np.log(classprobability)
    pred_label = np.argmax(classweight_combined)
    
    return pred_label, classweight_combined, classprobability
def pracmln_inference_single(formula_weight, single_row_feature, debug=True):
    # print(single_row_feature)
    arms_dist = int(single_row_feature[1])
    left_gs = int(single_row_feature[3])
    right_gs = int(single_row_feature[5])
    # labels = single_row_feature[:, 6] - 1
    # formula
    formula1 = "arms_dist_"+str(arms_dist)
    formula2 = "left_gs_"+str(left_gs)
    formula3 = "right_gs_"+str(right_gs)
    formula4 = "left_right_gs_"+str(left_gs)+"_"+str(right_gs)
    formulas = [formula1, formula2, formula3, formula4]
    fweights_list = []
    for f in formulas:
        if f in formula_weight:
            fweights_list.append(formula_weight[f])
    classweight = [0.0] * 7
    for wclasses in fweights_list:
        for i in range(len(classweight)):
            classweight[i] = classweight[i] + wclasses[i] # taking total sum
    # print (classweight)
    # make all positive, ignore negative value and make it to zero
    for i in range(len(classweight)):
        if classweight[i] < 0:
            classweight[i] = 0.0
    # print (classweight)
    # print (formulas)
    classweight_combined = np.asarray(classweight) # make it numpy

    
    # make it probability
    classweight_combined_forlogprob = classweight_combined + 1 # 1 to remove zero prob, smooting
    sumval = np.sum(classweight_combined_forlogprob)
    # if maxval > 0:
    classprobability = classweight_combined_forlogprob / sumval
    # classprobability = np.asarray(classprobability)
    # print(type(classprobability))
    # classprobability = classprobability.astype(float)
    classprobability = np.log(classprobability)
    pred_label = np.argmax(classweight_combined)
    
    return pred_label, classweight_combined, classprobability
def pracmln_inference_single2(formula_weight, single_row_feature, debug=True):
    # print("=============")
    # print(single_row_feature)
    arms_dist = int(single_row_feature[1])
    left_gs = int(single_row_feature[3])
    right_gs = int(single_row_feature[5])
    # labels = single_row_feature[:, 6] - 1
    # formula

    formula1 = "arms_dist_"+str(arms_dist)
    formula2 = "left_gs_"+str(left_gs)
    formula3 = "right_gs_"+str(right_gs)
    formula4 = "left_right_gs_"+str(left_gs)+"_"+str(right_gs)
    formulas = [formula1, formula2, formula3, formula4]
    fweights_list = []
    for f in formulas:
        if f in formula_weight:
            fweights_list.append(formula_weight[f])
    classweight = [0.0] * 7
    for wclasses in fweights_list:
        for i in range(len(classweight)):
            classweight[i] = classweight[i] + wclasses[i] # taking total sum
    # print ("Before normalize", classweight)
    # make all positive, ignore negative value and make it to zero
    minweight = min(classweight)
    for i in range(len(classweight)):
        # if classweight[i] < 0:
        #     classweight[i] = 0.0
        classweight[i] = classweight[i] + (-1 * minweight) # make it minimum to 0.
    # print ("After normalize",classweight)
    # print(fweights_list)
    # print (formulas)
    classweight_combined = np.asarray(classweight) # make it numpy

    
    # make it probability
    classweight_combined_forlogprob = classweight_combined + 1 # 1 to remove zero prob, smooting
    sumval = np.sum(classweight_combined_forlogprob)
    # if maxval > 0:
    classprobability = classweight_combined_forlogprob / sumval
    # classprobability = np.asarray(classprobability)
    # print(type(classprobability))
    # classprobability = classprobability.astype(float)
    classprobability = np.log(classprobability)
    pred_label = np.argmax(classweight_combined)
    
    return pred_label, classweight_combined, classprobability
def pracmln_inference_single_gym(formula_weight, single_row_feature, debug=True):
    # print("=============")
    # print(single_row_feature)
    # print(formula_weight)
    gs = int(single_row_feature[0])
    # labels = single_row_feature[:, 6] - 1
    # formula

    formula1 = "gs_"+str(gs)
    
    formulas = [formula1]
    fweights_list = []
    for f in formulas:
        if f in formula_weight:
            fweights_list.append(formula_weight[f])
    classweight = [0.0] * 4
    for wclasses in fweights_list:
        for i in range(len(classweight)):
            classweight[i] = classweight[i] + wclasses[i] # taking total sum
    # print ("Before normalize", classweight)
    # make all positive, ignore negative value and make it to zero
    minweight = min(classweight)
    for i in range(len(classweight)):
        # if classweight[i] < 0:
        #     classweight[i] = 0.0
        classweight[i] = classweight[i] + (-1 * minweight) # make it minimum to 0.
    # print ("After normalize",classweight)
    # print(fweights_list)
    # print (formulas)
    classweight_combined = np.asarray(classweight) # make it numpy

    
    # make it probability
    classweight_combined_forlogprob = classweight_combined + 1 # 1 to remove zero prob, smooting
    sumval = np.sum(classweight_combined_forlogprob)
    # if maxval > 0:
    classprobability = classweight_combined_forlogprob / sumval
    # classprobability = np.asarray(classprobability)
    # print(type(classprobability))
    # classprobability = classprobability.astype(float)
    classprobability = np.log(classprobability)
    pred_label = np.argmax(classweight_combined)
    # print(pred_label, classweight_combined, classprobability)
    return pred_label, classweight_combined, classprobability
def pracmln_inference(formula_weight, single_row_feature, robot="Gym"):
    # formula
    if robot == "Gym":
        gs = int(single_row_feature[0])
        formula1 = "gs_"+str(gs)
        formulas = [formula1]
        class_dim = 4
    else: #
        arms_dist = int(single_row_feature[1])
        left_gs = int(single_row_feature[3])
        right_gs = int(single_row_feature[5])

        formula1 = "arms_dist_"+str(arms_dist)
        formula2 = "left_gs_"+str(left_gs)
        formula3 = "right_gs_"+str(right_gs)
        formula4 = "left_right_gs_"+str(left_gs)+"_"+str(right_gs)
        formulas = [formula1, formula2, formula3, formula4]
        class_dim = 7
    
    fweights_list = []
    for f in formulas:
        if f in formula_weight:
            fweights_list.append(formula_weight[f])
    classweight = [0.0] * class_dim
    for wclasses in fweights_list:
        for i in range(len(classweight)):
            classweight[i] = classweight[i] + wclasses[i] # taking total sum
    # Normalization - make all positive, ignore negative value and make it to zero
    minweight = min(classweight)
    for i in range(len(classweight)):
        classweight[i] = classweight[i] + (-1 * minweight) # make it minimum to 0.
    classweight_combined = np.asarray(classweight) # make it numpy

    # make it probability
    classweight_combined_forlogprob = classweight_combined + 1 # 1 to remove zero prob, smooting
    sumval = np.sum(classweight_combined_forlogprob)
    classprobability = classweight_combined_forlogprob / sumval
    classprobability = np.log(classprobability)
    pred_label = np.argmax(classweight_combined)
    
    return pred_label, classweight_combined, classprobability


import re
def getformatedweights(wfile):
    # fw = open(data_dir+'/learned_weights_formated.w', 'w')
    formula_dict = {}
    with open(wfile) as f:
        for line in f:
            line = line.replace("\r", "")
            line = line.replace("\n", "")
            line = re.sub(' +', ' ', line)
            row = line.split(" ")
            # print(row)
            weight = row[0]
            constraint = row[1]
            label = row[3]
            
            constraint = constraint.replace('Has("', '')
            constraint = constraint.replace('",p)', '')

            label = label.replace('Topic("','')
            label = label.replace('",p)', '')


            # print(weight, constraint, label)
            # fw.write(str(weight)+"\t"+str(constraint)+"\t"+str(label)+"\n")

            label = int(label.replace('C', ''))
    
            if constraint not in formula_dict:
                classweight = [0.0] * 7
                classweight[label] = float(weight)
                formula_dict[constraint] = classweight
            else:
                classweight[label] = float(weight)
                formula_dict[constraint] = classweight
    # print(len(formula_dict))
    return formula_dict
def getformatedweights_gym(wfile):
    # fw = open(data_dir+'/learned_weights_formated.w', 'w')
    formula_dict = {}
    with open(wfile) as f:
        for line in f:
            line = line.replace("\r", "")
            line = line.replace("\n", "")
            line = re.sub(' +', ' ', line)
            row = line.split(" ")
            # print(row)
            weight = row[0]
            constraint = row[1]
            label = row[3]
            
            constraint = constraint.replace('Has("', '')
            constraint = constraint.replace('",p)', '')

            label = label.replace('Topic("','')
            label = label.replace('",p)', '')


            # print(weight, constraint, label)
            # fw.write(str(weight)+"\t"+str(constraint)+"\t"+str(label)+"\n")

            label = int(label.replace('C', ''))
    
            if constraint not in formula_dict:
                classweight = [0.0] * 4
                classweight[label] = float(weight)
                formula_dict[constraint] = classweight
            else:
                classweight[label] = float(weight)
                formula_dict[constraint] = classweight
    # print(len(formula_dict))
    return formula_dict

def dbformat():
    dfile = "Taurus_motion_features.txt"
    data = np.loadtxt(data_dir+"/"+dfile, delimiter=' ')
    fw_db = open(data_dir+"/mln_data/"+"taurus_motion_small.db", "w")
    for i in range(len(data)):
        if i > 500:
            break
        single_row_feature = data[i]
        arms_dist = int(single_row_feature[1])
        left_gs = int(single_row_feature[3])
        right_gs = int(single_row_feature[5])
        label = "C"+str(int(single_row_feature[-1]) - 1)
        # print(arms_dist, left_gs, right_gs, label)
        fw_db.write('Has("arms_dist_'+str(arms_dist)+'","'+str(i)+'")\n')
        # fw_db.write('Has("left_gs_'+str(left_gs)+'","'+str(i)+'")\n')
        # fw_db.write('Has("right_gs_'+str(right_gs)+'","'+str(i)+'")\n')
        fw_db.write('Topic("'+str(label)+'","'+str(i)+'")\n')
    fw_db.close()

if __name__ == '__main__':
    wfile = "../constraints_dir/mln_data/learned_weights.w"
    fdict = getformatedweights(wfile)
    print(fdict)
    print(len(fdict))
    for key in fdict:
        print(key, fdict[key])

    # dbformat()
    # dfile = "Taurus_motion_features.txt"
    # # dfile = "Taurus_sim_motion_features.txt"
    # # dfile = "Yumi_motion_features.txt"
    # data1 = np.loadtxt(data_dir+"/"+dfile, delimiter=' ')
    # data1 = data_preparation(dfile)

    # dfile = "Yumi_motion_features.txt"
    # data2 = data_preparation(dfile)


    # print(len(data1), len(data2))



    # count = int(len(data1)* 0.8)
    # training_data = data1[:count]
    # testing_data = data1[count:]
    
    # print(len(training_data), len(testing_data))

    # surgeme_class = [0, 1, 2, 3, 4, 5, 6] # 7 surgeme class
    # # mln_analysis(training_data, testing_data)
    # model_classprob = mln_train(training_data)
    # testacc, classacc, classweights = mln_inference(model_classprob, testing_data)
    # print(testacc, classacc, len(classweights))






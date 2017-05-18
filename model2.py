from __future__ import division
import json
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
import operator

from sklearn.metrics import roc_curve, precision_recall_curve, precision_recall_fscore_support


##################################################### data prep phase #####################################################

# data directory
data_directory = r'/Users/ZhangHaotian/desktop/LO/'

# uuid & integer transfering
uuidAndIntMapping = json.loads(open(data_directory + r'uuidAndIntMapping.json').read())
intAndUuidMapping = json.loads(open(data_directory + r'intAndUuidMapping.json').read())

# load global reference of pairs
global_pos_ref = json.loads(open(data_directory + r'global_pos_ref.json').read())
global_neg_ref = json.loads(open(data_directory + r'global_neg_ref.json').read())

# load basic global data
global_pos_10_40262 = json.loads(open(data_directory + r'global_pos_10_40262.json').read()) # 10-feat
global_neg_10_402620 = json.loads(open(data_directory + r'global_neg_10_402620.json').read())
global_pos_200_embed_40262 = json.loads(open(data_directory + r'global_pos_200_embed_40262.json').read()) # doc vec
global_neg_200_embed_402620 = json.loads(open(data_directory + r'global_neg_200_embed_402620.json').read())
global_neg_200_walk_402620 = json.loads(open(data_directory + r'global_neg_walk_200_402620.json').read()) # random walk (name diff)
global_pos_200_walk_40262 = json.loads(open(data_directory + r'global_pos_walk_200_40262.json').read())

# load referene of pairs of each partition
part1PosTuples = json.loads(open(data_directory + r'part1PosTuples.json').read())
part2PosTuples = json.loads(open(data_directory + r'part2PosTuples.json').read())
part3PosTuples = json.loads(open(data_directory + r'part3PosTuples.json').read())
part4PosTuples = json.loads(open(data_directory + r'part4PosTuples.json').read())
part5PosTuples = json.loads(open(data_directory + r'part5PosTuples.json').read())

# load basic partition data
part1_pos_10 = json.loads(open(data_directory + r'part1_pos_10.json').read()) # 1552
part1_pos_200 = json.loads(open(data_directory + r'part1_pos_200.json').read())
part1_pos_walk_200 = json.loads(open(data_directory + r'part1_pos_walk_200.json').read())
part2_pos_10 = json.loads(open(data_directory + r'part2_pos_10.json').read()) # 24251
part2_pos_200 = json.loads(open(data_directory + r'part2_pos_200.json').read())
part2_pos_walk_200 = json.loads(open(data_directory + r'part2_pos_walk_200.json').read())
part3_pos_10 = json.loads(open(data_directory + r'part3_pos_10.json').read()) # 1353
part3_pos_200 = json.loads(open(data_directory + r'part3_pos_200.json').read())
part3_pos_walk_200 = json.loads(open(data_directory + r'part3_pos_walk_200.json').read())
part4_pos_10 = json.loads(open(data_directory + r'part4_pos_10.json').read()) # 3399
part4_pos_200 = json.loads(open(data_directory + r'part4_pos_200.json').read())
part4_pos_walk_200 = json.loads(open(data_directory + r'part4_pos_walk_200.json').read())
part5_pos_10 = json.loads(open(data_directory + r'part5_pos_10.json').read()) # 11692
part5_pos_200 = json.loads(open(data_directory + r'part5_pos_200.json').read())
part5_pos_walk_200 = json.loads(open(data_directory + r'part5_pos_walk_200.json').read())

# combine data from different categories
def combineData(source1_pos=None,
                source1_neg=None,
                source2_pos=None,
                source2_neg=None,
                source3_pos=None,
                source3_neg=None):

    # assert (len(source1_pos) == len(source2_pos) == len(source3_pos)), "pos should be equal length"
    # assert (len(source1_neg) == len(source2_neg) == len(source3_neg)), "neg should be equal length"

    comb_pos = []
    comb_neg = []

    if source3_pos == None: # only combine two datasets
        for i in range(len(source1_pos)):
            comb_pos.append(source1_pos[i] + source2_pos[i])

        if source1_neg != None:
            for i in range(len(source1_neg)):
                comb_neg.append(source1_neg[i] + source2_neg[i])
    else:
        for i in range(len(source1_pos)):
            comb_pos.append(source1_pos[i] + source2_pos[i] + source3_pos[i])

        if source1_neg != None:
            for i in range(len(source1_neg)):
                comb_neg.append(source1_neg[i] + source2_neg[i] + source3_neg[i])

    if len(comb_neg) == 0:
        return comb_pos
    else:
        return (comb_pos, comb_neg)

# combinations of each partition
part1_pos_10_dv = combineData(source1_pos=part1_pos_10, source2_pos=part1_pos_200)
part1_pos_10_walk = combineData(source1_pos=part1_pos_10, source2_pos=part1_pos_walk_200)
part1_pos_walk_dv = combineData(source1_pos=part1_pos_walk_200, source2_pos=part1_pos_200)
part1_pos_10_walk_dv = combineData(source1_pos=part1_pos_10, source3_pos=part1_pos_200, source2_pos=part1_pos_walk_200)

part2_pos_10_dv = combineData(source1_pos=part2_pos_10, source2_pos=part2_pos_200)
part2_pos_10_walk = combineData(source1_pos=part2_pos_10, source2_pos=part2_pos_walk_200)
part2_pos_walk_dv = combineData(source1_pos=part2_pos_walk_200, source2_pos=part2_pos_200)
part2_pos_10_walk_dv = combineData(source1_pos=part2_pos_10, source3_pos=part2_pos_200, source2_pos=part2_pos_walk_200)

part3_pos_10_dv = combineData(source1_pos=part3_pos_10, source2_pos=part3_pos_200)
part3_pos_10_walk = combineData(source1_pos=part3_pos_10, source2_pos=part3_pos_walk_200)
part3_pos_walk_dv = combineData(source1_pos=part3_pos_walk_200, source2_pos=part3_pos_200)
part3_pos_10_walk_dv = combineData(source1_pos=part3_pos_10, source3_pos=part3_pos_200, source2_pos=part3_pos_walk_200)

part4_pos_10_dv = combineData(source1_pos=part4_pos_10, source2_pos=part4_pos_200)
part4_pos_10_walk = combineData(source1_pos=part4_pos_10, source2_pos=part4_pos_walk_200)
part4_pos_walk_dv = combineData(source1_pos=part4_pos_walk_200, source2_pos=part4_pos_200)
part4_pos_10_walk_dv = combineData(source1_pos=part4_pos_10, source3_pos=part4_pos_200, source2_pos=part4_pos_walk_200)

part5_pos_10_dv = combineData(source1_pos=part5_pos_10, source2_pos=part5_pos_200)
part5_pos_10_walk = combineData(source1_pos=part5_pos_10, source2_pos=part5_pos_walk_200)
part5_pos_walk_dv = combineData(source1_pos=part5_pos_walk_200, source2_pos=part5_pos_200)
part5_pos_10_walk_dv = combineData(source1_pos=part5_pos_10, source3_pos=part5_pos_200, source2_pos=part5_pos_walk_200)

# combination of global data
(combPos_10_walk, combNeg_10_walk) = combineData(source1_pos=global_pos_10_40262,
                                 source1_neg=global_neg_10_402620,
                                 source2_pos=global_pos_200_walk_40262,
                                 source2_neg=global_neg_200_walk_402620,
                                 source3_pos=None,
                                 source3_neg=None)

(combPos_walk_dv, combNeg_walk_dv) = combineData(source1_pos=global_pos_200_walk_40262,
                                 source1_neg=global_neg_200_walk_402620,
                                 source2_pos=global_pos_200_embed_40262,
                                 source2_neg=global_neg_200_embed_402620,
                                 source3_pos=None,
                                 source3_neg=None)

(combPos_10_dv, combNeg_10_dv) = combineData(source1_pos=global_pos_10_40262,
                                 source1_neg=global_neg_10_402620,
                                 source2_pos=global_pos_200_embed_40262,
                                 source2_neg=global_neg_200_embed_402620,
                                 source3_pos=None,
                                 source3_neg=None)

(combPos_10_walk_dv, combNeg_10_walk_dv) = combineData(source1_pos=global_pos_10_40262,
                                 source1_neg=global_neg_10_402620,
                                 source2_pos=global_pos_200_walk_40262,
                                 source2_neg=global_neg_200_walk_402620,
                                 source3_pos=global_pos_200_embed_40262,
                                 source3_neg=global_neg_200_embed_402620)

##################################################### experiment phase #####################################################

# functions
# general function for taking samples from a list
def takingSamples(alist, num=0, portion=0):
    assert ((num > 0 and portion == 0) or (num == 0 and portion > 0)), "should offer only one method, num or portion"
    seed = int(round(time.time() * 1000)) % 100000000
    random.seed(seed)
    length_of_list = len(alist)
    listPicked = []
    listNotPicked = []

    if num > 0:
        chosen_ids = set()
        while len(chosen_ids) < num:
            tmpRandInt = random.randint(0, length_of_list - 1) # cover both head and tail
            chosen_ids.add(tmpRandInt)

        t_f_list = [False for i in range(length_of_list)]
        for i in chosen_ids:
            t_f_list[i] = True

        for i,j in enumerate(t_f_list):
            if j:
                listPicked.append(alist[i])
            else:
                listNotPicked.append(alist[i])

    if portion > 0:
        num = int(length_of_list * portion)
        chosen_ids = set()
        while len(chosen_ids) < num:
            tmpRandInt = random.randint(0, length_of_list - 1)  # cover both head and tail
            chosen_ids.add(tmpRandInt)

        t_f_list = [False for i in range(length_of_list)]
        for i in chosen_ids:
            t_f_list[i] = True

        for i, j in enumerate(t_f_list):
            if j:
                listPicked.append(alist[i])
            else:
                listNotPicked.append(alist[i])

    return (listPicked, listNotPicked)

    # usage e.g.
    # (listPicked, listNotPicked) = takingSamples([1,2,3,4,5,6], num=4)
    # (listPicked, listNotPicked) = takingSamples([[1,2],[2,5],[3,7],[4,6],[5,5],[6,1]], num=4)
    # print listPicked
    # print listNotPicked

# averaging the results from trials
def avgProcess(trialsAns):
    trialsAns_np = np.array(trialsAns)
    num_trial = len(trialsAns_np) # 10

    # place holder for average threshold, precision, recall, f1
    avg_thres = np.array([0.0 for i in range(len(trialsAns_np[0]))])
    avg_prec = np.array([0.0 for i in range(len(trialsAns_np[0]))])
    avg_rec = np.array([0.0 for i in range(len(trialsAns_np[0]))])
    avg_f1 = np.array([0.0 for i in range(len(trialsAns_np[0]))])

    for i in range(num_trial):
        tmp = np.array(trialsAns_np[i])
        avg_thres += tmp[:, 0] # the 0th column
        avg_prec += tmp[:, 1]
        avg_rec += tmp[:, 2]
        avg_f1 += tmp[:, 3]

    avg_thres = avg_thres / float(num_trial)
    avg_prec = avg_prec / float(num_trial)
    avg_rec = avg_rec / float(num_trial)
    avg_f1 = avg_f1 / float(num_trial)

    avg_thres = list(avg_thres)
    avg_prec = list(avg_prec)
    avg_rec = list(avg_rec)
    avg_f1 = list(avg_f1)

    return (avg_thres, avg_prec, avg_rec, avg_f1)

# input should be lists of 10 or 210 dimensions
def oneTrialWithCertainTrainSize(num_pos_sample=50,
                                 neg_pos_ratio=1,
                                 pos_dataset=None,
                                 neg_dataset=None,
                                 train_test_split=0, # obselete feature, keep default parameter to bypass, feature achieved by "num_pos_sample" param
                                 test_stratify=True, # obselete feature, keep default parameter to bypass, feature achieved by "num_pos_sample" param
                                 scoring="f1",
                                 plt_or_not=True):

    assert(type(pos_dataset) == list and type(neg_dataset) == list), "input datasets should be lists"

    num_neg_sample = int(num_pos_sample * neg_pos_ratio)

    # take sample of num_pos_sample number of positive examples
    (posPicked, posNotPicked) = takingSamples(pos_dataset, num=num_pos_sample)
    (negPicked, negNotPicked) = takingSamples(neg_dataset, num=num_neg_sample)

    # create train_X, train_y
    train_X = pd.DataFrame(posPicked + negPicked)
    train_y = np.array([1 for i in range(len(posPicked))] + [0 for i in range(len(negPicked))])

    # create test_X and test_y
    if train_test_split != 0:
        testSize = int((num_pos_sample + num_neg_sample) / train_test_split * (1 - train_test_split)) # size of test set
        if test_stratify:
            testPosSize = int(float(testSize) / (neg_pos_ratio + 1))
            testNegSize = testSize - testPosSize
            test_X = pd.DataFrame(takingSamples(posNotPicked, num=testPosSize)[0] + takingSamples(negNotPicked, num=testNegSize)[0]) #
            test_y = np.array([1 for i in range(testPosSize)] + [0 for i in range(testNegSize)])
        else:
            for idx in range(len(posNotPicked)):
                posNotPicked[idx].append(1)
            for idx in range(len(negNotPicked)):
                negNotPicked[idx].append(0)
            test_X = pd.DataFrame(takingSamples(posNotPicked + negNotPicked, num=testSize)[0])

            test_y = np.array()
            for i in test_X:
                if i[-1] == 1:
                    test_y.append(1)
                else:
                    test_y.append(0)

            for idx in range(len(test_X)):
                del test_X[idx][-1]

    else:
        test_X = pd.DataFrame(posNotPicked + negNotPicked)
        test_y = np.array([1 for i in range(len(posNotPicked))] + [0 for i in range(len(negNotPicked))])


    # train and test the model
    reg = LogisticRegressionCV(scoring=scoring)
    LogModel = reg.fit(train_X, train_y)
    y_predlog = LogModel.predict_proba(test_X)
    y_predlog_1 = y_predlog[:, 1]

    prec, rec, thresholds = precision_recall_curve(test_y, y_predlog_1)
    if plt_or_not:
        plt.plot(rec, prec)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title("Rec-Prec Curve of Logistic Regression Trials")

    # pred_combine sorted
    pred_combine = []
    for i in range(len(test_y)):
        pred_combine.append((y_predlog_1[i], test_y[i]))

    pred_combine = sorted(pred_combine, key=operator.itemgetter(0))

    # create an array of 0.1:0.01:0.99
    thres_new = []
    initial = 0.1
    while initial <= 0.99:
        thres_new.append(initial)
        initial += 0.01
        initial = round(initial, 2)

    # generate "threshold, prec, rec, f1" list
    # test_y is truth, y_predlog_1 is prob of being 1
    result = []
    item_index = 0

    FN_accu = 0
    TN_accu = 0
    TP_accu = list(test_y).count(1)
    FP_accu = list(test_y).count(0)

    for i in thres_new:  # i is [0.1:0.01:0.99]
        if (item_index < len(pred_combine)):
            while pred_combine[item_index][0] < i:
                if pred_combine[item_index][1] == 1:  # this item actually 1, predict as 0
                    FN_accu += 1
                    TP_accu -= 1
                else:  # this item is actually 0, predict as 0, pred_combine[item_index][1] == 0
                    TN_accu += 1
                    FP_accu -= 1
                item_index += 1
                if (item_index == len(pred_combine)): break

        # print "th: " + str(i) + ", TP: " + str(TP_accu) + ", FP: " + str(FP_accu) + ", FN: " + str(FN_accu) + ", TN: " + str(TN_accu)

        if (TP_accu == 0):
            preci = 0
        else:
            preci = float(TP_accu) / (TP_accu + FP_accu)

        if (TP_accu == 0):
            recal = 0
        else:
            recal = float(TP_accu) / (FN_accu + TP_accu)

        if (2 * preci * recal == 0):
            fone = 0
        else:
            fone = 2 * preci * recal / (preci + recal)

        result.append([i, preci, recal, fone])

    return result # 90

    # outArr = oneTrialWithCertainTrainSize(num_pos_sample=60, pos_neg_ratio=1, pos_dataset=global_pos_10_40262, neg_dataset=global_neg_10_402620)
    # print "finish"

# trialsWithVariedTrainSize
def trialsWithVariedTrainSize(num_pos_sample=50,
                              num_pos_sample_cap=1500,
                              neg_pos_ratio=1,
                              pos_dataset=None,
                              neg_dataset=None,
                              train_test_split=0, # obselete feature, keep default parameter to bypass, feature achieved by "num_pos_sample" param
                              test_stratify=True, # obselete feature, keep default parameter to bypass, feature achieved by "num_pos_sample" param
                              scoring="f1",
                              plt_or_not=True,
                              num_trial=10,
                              save=False,
                              saveName="0"):

    generalResults = []
    generalResultsPosNumRef = []
    generalStdDev = []

    while num_pos_sample <= num_pos_sample_cap:
        trialsAns = []

        # for each num_pos_sample, perform 10 trials
        for trialsCount in range(num_trial):
            # one single trial
            outArr = oneTrialWithCertainTrainSize(num_pos_sample=num_pos_sample, neg_pos_ratio=neg_pos_ratio, pos_dataset=pos_dataset, neg_dataset=neg_dataset, train_test_split=train_test_split, test_stratify=test_stratify, scoring=scoring, plt_or_not=plt_or_not)
            # put outArr together
            trialsAns.append(outArr) # outArr = [threshold, prec, rec, f1tmp]

            print "trial #" + str(trialsCount + 1)  + " finished!"

        # with open('trialsAns.json', 'w') as f:
        #     json.dump(trialsAns, f)

        print str(num_pos_sample) + " all trials finished!"

        # calc std dev of max f1 based on trialsAns
        # stdArray = []
        # for e in range(len(trialsAns[0])):
        #     tmpArr = []
        #     for k in trialsAns:
        #         tmpArr.append(k[e][3])
        #     stdArray.append(np.std(np.array(tmpArr)))
        #
        # stddev = np.average(stdArray)
        # generalStdDev.append(stddev)
        #
        if save == True:
            fileName = "rawResults_" + saveName + ".json"
            with open(fileName, 'w') as f: json.dump(trialsAns, f)

        (avg_thres, avg_prec, avg_rec, avg_f1) = avgProcess(trialsAns)

        #
        generalResults.append([avg_thres, avg_prec, avg_rec, avg_f1])
        generalResultsPosNumRef.append(num_pos_sample)
        #
        print str(num_pos_sample) + " positive finished!"

        num_pos_sample += 50

        # if num_pos_sample < 200: num_pos_sample += 10
        # elif num_pos_sample < 500: num_pos_sample += 50
        # else: num_pos_sample += 100

    # return (generalResults, generalStdDev, generalResultsPosNumRef)
    return (generalResults, generalResultsPosNumRef)
    # return None

    # usage e.g. (generalResults, generalStdDev, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=50, num_pos_sample_cap=1500, pos_neg_ratio=1, pos_dataset=global_pos_10_40262, neg_dataset=global_neg_10_402620)


##################################################### single experiment global #####################################################

# usefuldata
# global_pos_10_40262
# global_neg_10_402620
# global_pos_200_embed_40262
# global_neg_200_embed_402620
# global_pos_200_walk_40262
# global_neg_200_walk_402620

# combPos_10_dv, combNeg_10_dv, combPos_10_walk, combNeg_10_walk, combPos_10_walk_dv, combNeg_10_walk_dv, combPos_walk_dv, combNeg_walk_dv

# 10
(global_10, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=12078,
                                                                     num_pos_sample_cap=12078,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=global_pos_10_40262,
                                                                     neg_dataset=global_neg_10_402620,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=True,
                                                                     save=True,
                                                                     saveName="10")

targ = global_10
max_f1 = max(targ[0][3]) # 0.5178
index_max_f1 = targ[0][3].index(max(targ[0][3])) # 65
prec_at_max_f1 = targ[0][1][index_max_f1] # 0.4846
rec_at_max_f1 = targ[0][2][index_max_f1] # 0.5560

print "index: " + str(index_max_f1) + ", f1: " + str(round(max_f1,4)) + ", prec: " + str(round(prec_at_max_f1,4)) + ", rec: " + str(round(rec_at_max_f1,4))

with open('global_10.json', 'w') as f: json.dump(global_10, f)

rawResults_10 = json.loads(open(data_directory + r'rawResults_10.json').read()) # thres, preci, recal, fone


# dv
(global_dv, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=12078,
                                                                     num_pos_sample_cap=12078,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=global_pos_200_embed_40262,
                                                                     neg_dataset=global_neg_200_embed_402620,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=True,
                                                                     save=True,
                                                                     saveName="dv")

targ = global_dv
max_f1 = max(targ[0][3]) # 0.319
index_max_f1 = targ[0][3].index(max(targ[0][3])) # 56
prec_at_max_f1 = targ[0][1][index_max_f1] # 0.2673
rec_at_max_f1 = targ[0][2][index_max_f1] # 0.3999

print "index: " + str(index_max_f1) + ", f1: " + str(round(max_f1,4)) + ", prec: " + str(round(prec_at_max_f1,4)) + ", rec: " + str(round(rec_at_max_f1,4))

with open('global_dv.json', 'w') as f: json.dump(global_dv, f)

rawResults_dv = json.loads(open(data_directory + r'rawResults_dv.json').read()) # thres, preci, recal, fone


# walk
(global_walk, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=12078,
                                                                     num_pos_sample_cap=12078,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=global_pos_200_walk_40262,
                                                                     neg_dataset=global_neg_200_walk_402620,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=True,
                                                                     save=True,
                                                                     saveName="walk")

targ = global_walk
max_f1 = max(targ[0][3]) # 0.3652
index_max_f1 = targ[0][3].index(max(targ[0][3])) # 59
prec_at_max_f1 = targ[0][1][index_max_f1] # 0.269
rec_at_max_f1 = targ[0][2][index_max_f1] # 0.5709

print "index: " + str(index_max_f1) + ", f1: " + str(round(max_f1,4)) + ", prec: " + str(round(prec_at_max_f1,4)) + ", rec: " + str(round(rec_at_max_f1,4))

with open('global_walk.json', 'w') as f: json.dump(global_walk, f)

rawResults_walk = json.loads(open(data_directory + r'rawResults_walk.json').read()) # thres, preci, recal, fone


# walk_dv
(global_walk_dv, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=12078,
                                                                     num_pos_sample_cap=12078,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=combPos_walk_dv,
                                                                     neg_dataset=combNeg_walk_dv,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=True,
                                                                     save=True,
                                                                     saveName="walk_dv")

targ = global_walk_dv
max_f1 = max(targ[0][3]) # 0.4016
index_max_f1 = targ[0][3].index(max(targ[0][3])) # 60
prec_at_max_f1 = targ[0][1][index_max_f1] # 0.3121
rec_at_max_f1 = targ[0][2][index_max_f1] # 0.5669

print "index: " + str(index_max_f1) + ", f1: " + str(round(max_f1,4)) + ", prec: " + str(round(prec_at_max_f1,4)) + ", rec: " + str(round(rec_at_max_f1,4))

with open('global_walk_dv.json', 'w') as f: json.dump(global_walk_dv, f)

rawResults_walk_dv = json.loads(open(data_directory + r'rawResults_walk_dv.json').read()) # thres, preci, recal, fone


# 10_dv
(global_10_dv, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=12078,
                                                                     num_pos_sample_cap=12078,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=combPos_10_dv,
                                                                     neg_dataset=combNeg_10_dv,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=True,
                                                                     save=True,
                                                                     saveName="10_dv")

targ = global_10_dv
max_f1 = max(targ[0][3]) # 0.5676
index_max_f1 = targ[0][3].index(max(targ[0][3])) # 72
prec_at_max_f1 = targ[0][1][index_max_f1] # 0.5492
rec_at_max_f1 = targ[0][2][index_max_f1] # 0.5874

print "index: " + str(index_max_f1) + ", f1: " + str(round(max_f1,4)) + ", prec: " + str(round(prec_at_max_f1,4)) + ", rec: " + str(round(rec_at_max_f1,4))

with open('global_10_dv.json', 'w') as f: json.dump(global_10_dv, f)

rawResults_10_dv = json.loads(open(data_directory + r'rawResults_10_dv.json').read()) # thres, preci, recal, fone

# 10_walk
(generalResults_10_walk, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=12078,
                                                                     num_pos_sample_cap=12078,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=combPos_10_walk,
                                                                     neg_dataset=combNeg_10_walk,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=True,
                                                                     save=True,
                                                                     saveName="10_walk")

targ = generalResults_10_walk
max_f1 = max(targ[0][3]) # 0.5885
index_max_f1 = targ[0][3].index(max(targ[0][3])) # 73
prec_at_max_f1 = targ[0][1][index_max_f1] # 0.5536
rec_at_max_f1 = targ[0][2][index_max_f1] # 0.6204

print "index: " + str(index_max_f1) + ", f1: " + str(round(max_f1,4)) + ", prec: " + str(round(prec_at_max_f1,4)) + ", rec: " + str(round(rec_at_max_f1,4))

with open('global_10_walk.json', 'w') as f: json.dump(generalResults_10_walk, f)

rawResults_10_walk = json.loads(open(data_directory + r'rawResults_10_walk.json').read()) # thres, preci, recal, fone

# 10_walk_dv
(global_10_walk_dv, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=12078,
                                                                     num_pos_sample_cap=12078,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=combPos_10_walk_dv,
                                                                     neg_dataset=combNeg_10_walk_dv,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=True,
                                                                     save=True,
                                                                     saveName="10_walk_dv")

targ = global_10_walk_dv
max_f1 = max(targ[0][3]) # 0.5939
index_max_f1 = targ[0][3].index(max(targ[0][3])) # 70
prec_at_max_f1 = targ[0][1][index_max_f1] # 0.5663
rec_at_max_f1 = targ[0][2][index_max_f1] # 0.6248

print "index: " + str(index_max_f1) + ", f1: " + str(round(max_f1,4)) + ", prec: " + str(round(prec_at_max_f1,4)) + ", rec: " + str(round(rec_at_max_f1,4))

with open('global_10_walk_dv.json', 'w') as f: json.dump(global_10_walk_dv, f)

rawResults_10_walk_dv = json.loads(open(data_directory + r'rawResults_10_walk_dv.json').read()) # thres, preci, recal, fone

##################################################### varying training sizes #####################################################
# varying training size, 3
(generalResults_3_comp, generalResultsPosNumRef_3_comp) = trialsWithVariedTrainSize(num_pos_sample=50,
                                                                                     num_pos_sample_cap=5000,
                                                                                     neg_pos_ratio=1,
                                                                                     pos_dataset=combPos_10_walk_dv,
                                                                                     neg_dataset=combNeg_10_walk_dv,
                                                                                     train_test_split=0,
                                                                                     test_stratify=True,
                                                                                     scoring="f1",
                                                                                     plt_or_not=False,
                                                                                     num_trial=10)
generalF1_3_comp = []
for i in generalResults_3_comp:
    generalF1_3_comp.append(max(i[3]))

print generalF1_3_comp

with open('generalResults_3_comp.json' , 'w') as f: json.dump(generalResults_3_comp, f)
with open('generalResultsPosNumRef_3_comp.json' , 'w') as f: json.dump(generalResultsPosNumRef_3_comp, f)
with open('generalF1_3_comp.json' , 'w') as f: json.dump(generalF1_3_comp, f)

plt.plot(generalResultsPosNumRef_3_comp, generalF1_3_comp, color="black", linewidth=2, linestyle="-", label="10-feat + doc vec + walk")
plt.legend(loc='lower center')
plt.xlabel('num of positive samples')
plt.ylabel('f1')
plt.title('f1 with different num of positive samples')

# varying training size, only 10-feature
(generalResults_10_feat, generalResultsPosNumRef_10_feat) = trialsWithVariedTrainSize(num_pos_sample=50,
                                                                                     num_pos_sample_cap=5000,
                                                                                     neg_pos_ratio=1,
                                                                                     pos_dataset=global_pos_10_40262,
                                                                                     neg_dataset=global_neg_10_402620,
                                                                                     train_test_split=0,
                                                                                     test_stratify=True,
                                                                                     scoring="f1",
                                                                                     plt_or_not=False)

generalF1_10_feat = []
for i in generalResults_10_feat:
    generalF1_10_feat.append(max(i[3]))

print generalF1_10_feat

with open('generalResults_10_feat.json' , 'w') as f: json.dump(generalResults_10_feat, f)
with open('generalResultsPosNumRef_10_feat.json' , 'w') as f: json.dump(generalResultsPosNumRef_10_feat, f)
with open('generalF1_10_feat.json' , 'w') as f: json.dump(generalF1_10_feat, f)

plt.plot(generalResultsPosNumRef_10_feat, generalF1_10_feat, color="red", linewidth=2, linestyle="-", label="10-feat")
plt.legend(loc='lower center')
plt.xlabel('num of positive samples')
plt.ylabel('f1')
plt.title('f1 with different num of positive samples')


# varying training size, 10-feature and doc vec
(generalResults_10_dv, generalResultsPosNumRef_10_dv) = trialsWithVariedTrainSize(num_pos_sample=50,
                                                                                     num_pos_sample_cap=5000,
                                                                                     neg_pos_ratio=1,
                                                                                     pos_dataset=combPos_10_dv,
                                                                                     neg_dataset=combNeg_10_dv,
                                                                                     train_test_split=0,
                                                                                     test_stratify=True,
                                                                                     scoring="f1",
                                                                                     plt_or_not=False)

generalF1_10_dv = []
for i in generalResults_10_dv:
    generalF1_10_dv.append(max(i[3]))

print generalF1_10_dv

with open('generalResults_10_dv.json' , 'w') as f: json.dump(generalResults_10_dv, f)
with open('generalResultsPosNumRef_10_dv.json' , 'w') as f: json.dump(generalResultsPosNumRef_10_dv, f)
with open('generalF1_10_dv.json' , 'w') as f: json.dump(generalF1_10_dv, f)

plt.plot(generalResultsPosNumRef_10_dv, generalF1_10_dv, color="green", linewidth=2, linestyle="-", label="10-feat + doc vec")
plt.legend(loc='lower center')
plt.xlabel('num of positive samples')
plt.ylabel('f1')
plt.title('f1 with different num of positive samples')

##################################################### each partition #####################################################

## each partition

# part1_pos_10
# part1_pos_200
# part1_pos_walk_200
# part1_pos_10_dv
# part1_pos_10_walk
# part1_pos_walk_dv
# part1_pos_10_walk_dv

## part 1
# 10+walk+dv
(part1_10_walk_dv, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=465,
                                                                     num_pos_sample_cap=465,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=part1_pos_10_walk_dv,
                                                                     neg_dataset=combNeg_10_walk_dv,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=False,
                                                                     save=True,
                                                                     saveName="part1_10_walk_dv")

print max(part1_10_walk_dv[0][3])

with open('part1_10_walk_dv.json', 'w') as f: json.dump(part1_10_walk_dv, f)
rawResults_part1_10_walk_dv = json.loads(open(data_directory + r'rawResults_part1_10_walk_dv.json').read()) # thres, preci, recal, fone

# 10
(part1_10, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=465,
                                                                     num_pos_sample_cap=465,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=part1_pos_10,
                                                                     neg_dataset=global_neg_10_402620,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=False,
                                                                     save=True,
                                                                     saveName="part1_10")

print max(part1_10[0][3])

with open('part1_10.json', 'w') as f: json.dump(part1_10, f)
rawResults_part1_10 = json.loads(open(data_directory + r'rawResults_part1_10.json').read()) # thres, preci, recal, fone

# dv
(part1_dv, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=465,
                                                                     num_pos_sample_cap=465,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=part1_pos_200,
                                                                     neg_dataset=global_neg_200_embed_402620,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=False,
                                                                     save=True,
                                                                     saveName="part1_dv")
print max(part1_dv[0][3])

with open('part1_dv.json', 'w') as f: json.dump(part1_dv, f)
rawResults_part1_dv = json.loads(open(data_directory + r'rawResults_part1_dv.json').read()) # thres, preci, recal, fone

# 10+dv
(part1_10_dv, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=465,
                                                                     num_pos_sample_cap=465,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=part1_pos_10_dv,
                                                                     neg_dataset=combNeg_10_dv,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=False,
                                                                     save=True,
                                                                     saveName="part1_10_dv")

print max(part1_10_dv[0][3])

with open('part1_10_dv.json', 'w') as f: json.dump(part1_10_dv, f)
rawResults_part1_10_dv = json.loads(open(data_directory + r'rawResults_part1_10_dv.json').read()) # thres, preci, recal, fone

# plot of part1
plt.plot(part1_10[0][2], part1_10[0][1], linewidth=2, linestyle="-", label="10-FEAT")
plt.plot(part1_dv[0][2], part1_dv[0][1], linewidth=2, linestyle="-", label="doc2vec-SUP")
plt.plot(part1_10_dv[0][2], part1_10_dv[0][1], linewidth=2, linestyle="-", label="10-FEAT + doc2vec-SUP")

plt.legend(loc='upper right')
plt.xlabel('recall', fontsize=20)
plt.ylabel('precision', fontsize=20)
plt.title('Recall-Precision Curve', fontsize=20)

## part 2
(part2_10_walk_dv, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=7275,
                                                                     num_pos_sample_cap=7275,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=part2_pos_10_walk_dv,
                                                                     neg_dataset=combNeg_10_walk_dv,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=False,
                                                                     save=True,
                                                                     saveName="part2_10_walk_dv")


print max(part2_10_walk_dv[0][3])

with open('part2_10_walk_dv.json', 'w') as f: json.dump(part2_10_walk_dv, f)
rawResults_part2_10_walk_dv = json.loads(open(data_directory + r'rawResults_part2_10_walk_dv.json').read()) # thres, preci, recal, fone

# 10
(part2_10, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=7275,
                                                                     num_pos_sample_cap=7275,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=part2_pos_10,
                                                                     neg_dataset=global_neg_10_402620,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=False,
                                                                     save=True,
                                                                     saveName="part2_10")

print max(part2_10[0][3])

with open('part2_10.json', 'w') as f: json.dump(part2_10, f)
rawResults_part2_10 = json.loads(open(data_directory + r'rawResults_part2_10.json').read()) # thres, preci, recal, fone

# dv
(part2_dv, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=7275,
                                                                     num_pos_sample_cap=7275,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=part2_pos_200,
                                                                     neg_dataset=global_neg_200_embed_402620,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=False,
                                                                     save=True,
                                                                     saveName="part2_dv")
print max(part2_dv[0][3])

with open('part2_dv.json', 'w') as f: json.dump(part2_dv, f)
rawResults_part2_dv = json.loads(open(data_directory + r'rawResults_part2_dv.json').read()) # thres, preci, recal, fone

# 10+dv
(part2_10_dv, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=7275,
                                                                     num_pos_sample_cap=7275,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=part2_pos_10_dv,
                                                                     neg_dataset=combNeg_10_dv,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=False,
                                                                     save=True,
                                                                     saveName="part2_10_dv")

print max(part2_10_dv[0][3])

with open('part2_10_dv.json', 'w') as f: json.dump(part2_10_dv, f)
rawResults_part2_10_dv = json.loads(open(data_directory + r'rawResults_part2_10_dv.json').read()) # thres, preci, recal, fone

# plot of part2
plt.plot(part2_10[0][2], part2_10[0][1], linewidth=2, linestyle="-", label="10-FEAT")
plt.plot(part2_dv[0][2], part2_dv[0][1], linewidth=2, linestyle="-", label="doc2vec-SUP")
plt.plot(part2_10_dv[0][2], part2_10_dv[0][1], linewidth=2, linestyle="-", label="10-FEAT + doc2vec-SUP")

plt.legend(loc='upper right')
plt.xlabel('recall', fontsize=20)
plt.ylabel('precision', fontsize=20)
plt.title('Recall-Precision Curve', fontsize=20)

# part 3
(part3_10_walk_dv, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=405,
                                                                     num_pos_sample_cap=405,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=part3_pos_10_walk_dv,
                                                                     neg_dataset=combNeg_10_walk_dv,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=False,
                                                                     save=True,
                                                                     saveName="part3_10_walk_dv")

print max(part3_10_walk_dv[0][3])

with open('part3_10_walk_dv.json', 'w') as f: json.dump(part3_10_walk_dv, f)
rawResults_part3_10_walk_dv = json.loads(open(data_directory + r'rawResults_part3_10_walk_dv.json').read()) # thres, preci, recal, fone

# 10
(part3_10, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=405,
                                                                     num_pos_sample_cap=405,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=part3_pos_10,
                                                                     neg_dataset=global_neg_10_402620,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=False,
                                                                     save=True,
                                                                     saveName="part3_10")

print max(part3_10[0][3])

with open('part3_10.json', 'w') as f: json.dump(part3_10, f)
rawResults_part3_10 = json.loads(open(data_directory + r'rawResults_part3_10.json').read()) # thres, preci, recal, fone

# dv
(part3_dv, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=405,
                                                                     num_pos_sample_cap=405,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=part3_pos_200,
                                                                     neg_dataset=global_neg_200_embed_402620,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=False,
                                                                     save=True,
                                                                     saveName="part3_dv")
print max(part3_dv[0][3])

with open('part3_dv.json', 'w') as f: json.dump(part3_dv, f)
rawResults_part3_dv = json.loads(open(data_directory + r'rawResults_part3_dv.json').read()) # thres, preci, recal, fone

# 10+dv
(part3_10_dv, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=405,
                                                                     num_pos_sample_cap=405,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=part3_pos_10_dv,
                                                                     neg_dataset=combNeg_10_dv,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=False,
                                                                     save=True,
                                                                     saveName="part3_10_dv")

print max(part3_10_dv[0][3])

with open('part3_10_dv.json', 'w') as f: json.dump(part3_10_dv, f)
rawResults_part3_10_dv = json.loads(open(data_directory + r'rawResults_part3_10_dv.json').read()) # thres, preci, recal, fone

# plot of part3
plt.plot(part3_10[0][2], part3_10[0][1], linewidth=2, linestyle="-", label="10-FEAT")
plt.plot(part3_dv[0][2], part3_dv[0][1], linewidth=2, linestyle="-", label="doc2vec-SUP")
plt.plot(part3_10_dv[0][2], part3_10_dv[0][1], linewidth=2, linestyle="-", label="10-FEAT + doc2vec-SUP")

plt.legend(loc='upper right')
plt.xlabel('recall', fontsize=20)
plt.ylabel('precision', fontsize=20)
plt.title('Recall-Precision Curve', fontsize=20)

# part 4
(part4_10_walk_dv, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=1019,
                                                                     num_pos_sample_cap=1019,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=part4_pos_10_walk_dv,
                                                                     neg_dataset=combNeg_10_walk_dv,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=False,
                                                                     save=True,
                                                                     saveName="part4_10_walk_dv")

print max(part4_10_walk_dv[0][3])

with open('part4_10_walk_dv.json', 'w') as f: json.dump(part4_10_walk_dv, f)
rawResults_part4_10_walk_dv = json.loads(open(data_directory + r'rawResults_part4_10_walk_dv.json').read()) # thres, preci, recal, fone

# 10
(part4_10, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=1019,
                                                                     num_pos_sample_cap=1019,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=part4_pos_10,
                                                                     neg_dataset=global_neg_10_402620,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=False,
                                                                     save=True,
                                                                     saveName="part4_10")

print max(part4_10[0][3])

with open('part4_10.json', 'w') as f: json.dump(part4_10, f)
rawResults_part4_10 = json.loads(open(data_directory + r'rawResults_part4_10.json').read()) # thres, preci, recal, fone

# dv
(part4_dv, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=1019,
                                                                     num_pos_sample_cap=1019,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=part4_pos_200,
                                                                     neg_dataset=global_neg_200_embed_402620,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=False,
                                                                     save=True,
                                                                     saveName="part4_dv")
print max(part4_dv[0][3])

with open('part4_dv.json', 'w') as f: json.dump(part4_dv, f)
rawResults_part4_dv = json.loads(open(data_directory + r'rawResults_part4_dv.json').read()) # thres, preci, recal, fone

# 10+dv
(part4_10_dv, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=1019,
                                                                     num_pos_sample_cap=1019,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=part4_pos_10_dv,
                                                                     neg_dataset=combNeg_10_dv,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=False,
                                                                     save=True,
                                                                     saveName="part4_10_dv")

print max(part4_10_dv[0][3])

with open('part4_10_dv.json', 'w') as f: json.dump(part4_10_dv, f)
rawResults_part4_10_dv = json.loads(open(data_directory + r'rawResults_part4_10_dv.json').read()) # thres, preci, recal, fone

# plot of part4
plt.plot(part4_10[0][2], part4_10[0][1], linewidth=2, linestyle="-", label="10-FEAT")
plt.plot(part4_dv[0][2], part4_dv[0][1], linewidth=2, linestyle="-", label="doc2vec-SUP")
plt.plot(part4_10_dv[0][2], part4_10_dv[0][1], linewidth=2, linestyle="-", label="10-FEAT + doc2vec-SUP")

plt.legend(loc='upper right')
plt.xlabel('recall', fontsize=20)
plt.ylabel('precision', fontsize=20)
plt.title('Recall-Precision Curve', fontsize=20)

# part 5
(part5_10_walk_dv, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=3507,
                                                                     num_pos_sample_cap=3507,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=part5_pos_10_walk_dv,
                                                                     neg_dataset=combNeg_10_walk_dv,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=False,
                                                                     save=True,
                                                                     saveName="part5_10_walk_dv")

print max(part5_10_walk_dv[0][3])

with open('part5_10_walk_dv.json', 'w') as f: json.dump(part5_10_walk_dv, f)
rawResults_part5_10_walk_dv = json.loads(open(data_directory + r'rawResults_part5_10_walk_dv.json').read()) # thres, preci, recal, fone

# 10
(part5_10, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=3507,
                                                                     num_pos_sample_cap=3507,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=part5_pos_10,
                                                                     neg_dataset=global_neg_10_402620,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=False,
                                                                     save=True,
                                                                     saveName="part5_10")

print max(part5_10[0][3])

with open('part5_10.json', 'w') as f: json.dump(part5_10, f)
rawResults_part5_10 = json.loads(open(data_directory + r'rawResults_part5_10.json').read()) # thres, preci, recal, fone

# dv
(part5_dv, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=3507,
                                                                     num_pos_sample_cap=3507,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=part5_pos_200,
                                                                     neg_dataset=global_neg_200_embed_402620,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=False,
                                                                     save=True,
                                                                     saveName="part5_dv")
print max(part5_dv[0][3])

with open('part5_dv.json', 'w') as f: json.dump(part5_dv, f)
rawResults_part5_dv = json.loads(open(data_directory + r'rawResults_part5_dv.json').read()) # thres, preci, recal, fone

# 10+dv
(part5_10_dv, generalResultsPosNumRef) = trialsWithVariedTrainSize(num_pos_sample=3507,
                                                                     num_pos_sample_cap=3507,
                                                                     neg_pos_ratio=1,
                                                                     pos_dataset=part5_pos_10_dv,
                                                                     neg_dataset=combNeg_10_dv,
                                                                     train_test_split=0,
                                                                     test_stratify=True,
                                                                     scoring="f1",
                                                                     plt_or_not=False,
                                                                     save=True,
                                                                     saveName="part5_10_dv")

print max(part5_10_dv[0][3])

with open('part5_10_dv.json', 'w') as f: json.dump(part5_10_dv, f)
rawResults_part5_10_dv = json.loads(open(data_directory + r'rawResults_part5_10_dv.json').read()) # thres, preci, recal, fone

# plot of part5
plt.plot(part5_10[0][2], part5_10[0][1], linewidth=2, linestyle="-", label="10-FEAT")
plt.plot(part5_dv[0][2], part5_dv[0][1], linewidth=2, linestyle="-", label="doc2vec-SUP")
plt.plot(part5_10_dv[0][2], part5_10_dv[0][1], linewidth=2, linestyle="-", label="10-FEAT + doc2vec-SUP")

plt.legend(loc='upper right')
plt.xlabel('recall', fontsize=20)
plt.ylabel('precision', fontsize=20)
plt.title('Recall-Precision Curve', fontsize=20)










# for pred, truth in pred_combine:  # (pred, truth)
#
#     if (pred < thres_new[th_index]):
#         if truth == 1:  # this item actually 1, predict as 0
#             FN_accu += 1
#             TP_accu -= 1
#
#         elif truth == 0:  # this item is actually 0, predict as 0
#             TN_accu += 1
#             FP_accu -= 1
#
#     else:
#         preci = float(TP_accu) / (TP_accu + FP_accu)
#         recal = float(TP_accu) / (FN_accu + TP_accu)
#         fone = 2 * preci * recal / (preci + recal)
#         result.append((thres_new[th_index], preci, recal, fone))
#         th_index += 1
#
#         if
#             if truth == 1:  # this item actually 1, predict as 0
#                 FN_accu += 1
#                 TP_accu -= 1
#
#             elif truth == 0:  # this item is actually 0, predict as 0
#                 TN_accu += 1
#                 FP_accu -= 1

# make it 0.1:0.01:0.99
# outArr = []
# prog = 0.1
# step = 0.01
# for i, j in enumerate(thresholds):
#     if j > prog:
#         f1tmp = 2 * prec[i] * rec[i] / (prec[i] + rec[i])
#         outArr.append([j, prec[i], rec[i], f1tmp])  # threshold, precision, recall, f1tmp
#         prog += step


# # unit test
# num_pos_sample=20
# neg_pos_ratio=1
# pos_dataset=global_pos_10_40262
# neg_dataset=global_neg_10_402620
# train_test_split=0
# test_stratify=True
# scoring="f1"
# plt_or_not=True
#
# num_neg_sample = int(num_pos_sample * neg_pos_ratio)
#
# # take sample of num_pos_sample number of positive examples
# (posPicked, posNotPicked) = takingSamples(pos_dataset, num=num_pos_sample)
# (negPicked, negNotPicked) = takingSamples(neg_dataset, num=num_neg_sample)
#
# # create train_X, train_y
# train_X = pd.DataFrame(posPicked + negPicked)
# train_y = np.array([1 for i in range(len(posPicked))] + [0 for i in range(len(negPicked))])
#
# test_X = pd.DataFrame(posNotPicked + negNotPicked)
# test_y = np.array([1 for i in range(len(posNotPicked))] + [0 for i in range(len(negNotPicked))])
#
# reg = LogisticRegressionCV(scoring=scoring)
# LogModel = reg.fit(train_X, train_y)
# y_predlog = LogModel.predict_proba(test_X)
# y_predlog_1 = y_predlog[:, 1]
#
# prec, rec, thresholds = precision_recall_curve(test_y, y_predlog_1)
# if plt_or_not:
#     plt.plot(rec, prec)
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title("Rec-Prec Curve of Logistic Regression Trials")
#
# # pred_combine sorted
# pred_combine = []
# for i in range(len(test_y)):
#     pred_combine.append((y_predlog_1[i], test_y[i]))
#
# pred_combine = sorted(pred_combine, key=operator.itemgetter(0))
#
# # create an array of 0.1:0.01:0.99
# thres_new = []
# initial = 0.1
# while initial <= 0.99:
#     thres_new.append(initial)
#     initial += 0.01
#     initial = round(initial, 2)
#
# # generate "threshold, prec, rec, f1" list
# # test_y is truth, y_predlog_1 is prob of being 1
# result = []
# item_index = 0
#
# FN_accu = 0
# TN_accu = 0
# TP_accu = list(test_y).count(1)
# FP_accu = list(test_y).count(0)
#
# for i in thres_new: # i is [0.1:0.01:0.99]
#     if (item_index < len(pred_combine)):
#         while pred_combine[item_index][0] < i:
#             if pred_combine[item_index][1] == 1:  # this item actually 1, predict as 0
#                 FN_accu += 1
#                 TP_accu -= 1
#             else:  # this item is actually 0, predict as 0, pred_combine[item_index][1] == 0
#                 TN_accu += 1
#                 FP_accu -= 1
#             item_index += 1
#             if (item_index == len(pred_combine)): break
#
#     # print "th: " + str(i) + ", TP: " + str(TP_accu) + ", FP: " + str(FP_accu) + ", FN: " + str(FN_accu) + ", TN: " + str(TN_accu)
#
#     if (TP_accu == 0): preci = 0
#     else: preci = float(TP_accu) / (TP_accu + FP_accu)
#
#     if (TP_accu == 0): recal = 0
#     else: recal = float(TP_accu) / (FN_accu + TP_accu)
#
#     if (2 * preci * recal == 0): fone = 0
#     else: fone = 2 * preci * recal / (preci + recal)
#
#     result.append((i, preci, recal, fone))
#
# tmp = [(i[0], i[3]) for i in result]
#
#
# outArr = []
# prog = 0.1
# step = 0.01
# for i, j in enumerate(thresholds):
#     if j > prog:
#         f1tmp = 2 * prec[i] * rec[i] / (prec[i] + rec[i])
#         outArr.append([j, f1tmp])  # threshold, precision, recall, f1tmp
#         prog += step

## varying training size, 3
# (generalResults_3, generalResultsPosNumRef_3) = trialsWithVariedTrainSize(num_pos_sample=50,
#                                                                                      num_pos_sample_cap=2000,
#                                                                                      neg_pos_ratio=1,
#                                                                                      pos_dataset=combPos_10_walk_dv,
#                                                                                      neg_dataset=combNeg_10_walk_dv,
#                                                                                      train_test_split=0,
#                                                                                      test_stratify=True,
#                                                                                      scoring="f1",
#                                                                                      plt_or_not=False)
#
# generalF1_3 = []
# for i in generalResults_3:
#     generalF1_3.append(max(i[3]))
#
# print generalF1_3
#
# with open('generalResults_3.json' , 'w') as f: json.dump(generalResults_3, f)
# with open('generalResultsPosNumRef_3.json' , 'w') as f: json.dump(generalResultsPosNumRef_3, f)
# with open('generalF1_3.json' , 'w') as f: json.dump(generalF1_3, f)
#
# plt.plot(generalResultsPosNumRef_3, generalF1_3, color="green", linewidth=2, linestyle="-", label="neg:pos = 1")
# plt.legend(loc='upper center')
# plt.xlabel('num of positive samples')
# plt.ylabel('f1')
# plt.title('f1 with different num of positive samples')
#
# (generalResults_3_plus, generalResultsPosNumRef_3_plus) = trialsWithVariedTrainSize(num_pos_sample=2050,
#                                                                                      num_pos_sample_cap=5000,
#                                                                                      neg_pos_ratio=1,
#                                                                                      pos_dataset=combPos_10_walk_dv,
#                                                                                      neg_dataset=combNeg_10_walk_dv,
#                                                                                      train_test_split=0,
#                                                                                      test_stratify=True,
#                                                                                      scoring="f1",
#                                                                                      plt_or_not=False,
#                                                                                      num_trial=10)
#
# with open('generalResults_3_plus.json' , 'w') as f: json.dump(generalResults_3_plus, f)
# with open('generalResultsPosNumRef_3_plus.json' , 'w') as f: json.dump(generalResultsPosNumRef_3_plus, f)
#
# generalResults_3_comp = generalResults_3 + generalResults_3_plus
# generalResultsPosNumRef_3_comp = generalResultsPosNumRef_3 + generalResultsPosNumRef_3_plus
#
# generalF1_3_comp = []
# for i in generalResults_3_comp:
#     generalF1_3_comp.append(max(i[3]))
#
# print generalF1_3_comp
#
# with open('generalResults_3_comp.json' , 'w') as f: json.dump(generalResults_3_comp, f)
# with open('generalResultsPosNumRef_3_comp.json' , 'w') as f: json.dump(generalResultsPosNumRef_3_comp, f)
# with open('generalF1_3_comp.json' , 'w') as f: json.dump(generalF1_3_comp, f)
#
# plt.plot(generalResultsPosNumRef_3_comp, generalF1_3_comp, color="black", linewidth=2, linestyle="-", label="10-feat + doc vec + walk")
# plt.legend(loc='lower center')
# plt.xlabel('num of positive samples')
# plt.ylabel('f1')
# plt.title('f1 with different num of positive samples')
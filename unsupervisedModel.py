from __future__ import division
import json
import operator
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


## data prep
# data directory
data_directory = r'/Users/ZhangHaotian/desktop/LO/'

# uuid & integer transfering
uuidAndIntMapping = json.loads(open(data_directory + r'uuidAndIntMapping.json').read())
intAndUuidMapping = json.loads(open(data_directory + r'intAndUuidMapping.json').read())

# unsupervised scores for proba of 1
y_predlog_1_dv = json.loads(open(data_directory + r'y_predlog_1_dv.json').read()) # 40262 + 402620, global_pos_ref + global_neg_ref
y_predlog_1_tfidf = json.loads(open(data_directory + r'y_predlog_1_tfidf.json').read()) # 40262 + 402620, global_pos_ref + global_neg_ref

# pos and neg reference
global_pos_ref = json.loads(open(data_directory + r'global_pos_ref.json').read()) # 40262
global_neg_ref = json.loads(open(data_directory + r'global_neg_ref.json').read()) # 402620

# "correct" reference of 0/1
test_y = [1 for i in range(len(global_pos_ref))] + [0 for j in range(len(global_neg_ref))]

## experiment
# generate threshold, precision, recall, f1
def genThPrecRecF1(y_predlog_1, test_y):
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
        if (TP_accu == 0): preci = 0
        else: preci = float(TP_accu) / (TP_accu + FP_accu)

        if (TP_accu == 0): recal = 0
        else: recal = float(TP_accu) / (FN_accu + TP_accu)

        if (2 * preci * recal == 0): fone = 0
        else: fone = 2 * preci * recal / (preci + recal)

        result.append([i, preci, recal, fone])

    return result

result_tfidf = genThPrecRecF1(y_predlog_1_tfidf, test_y) # [i, preci, recal, fone]
result_dv = genThPrecRecF1(y_predlog_1_dv, test_y)

with open('result_tfidf.json', 'w') as f: json.dump(result_tfidf, f)
with open('result_dv.json', 'w') as f: json.dump(result_dv, f)

result_tfidf = np.array(result_tfidf)
result_dv = np.array(result_dv)

plt.plot(list(result_tfidf[:,2]), list(result_tfidf[:,1]), linewidth=2, linestyle="-", label="tfidf")
plt.plot(list(result_dv[:,2]), list(result_dv[:,1]), linewidth=2, linestyle="-", label="doc vec")

plt.legend(loc='upper right')
plt.xlabel('recall', fontsize=20)
plt.ylabel('precision', fontsize=20)
plt.title('Recall-Precision Curve', fontsize=20)

f1_1 = list(result_tfidf[:,3])
f1_2 = list(result_dv[:,3])
maxf1_tfidf = max(result_tfidf[:,3]) # 0.3266
maxf1_dv = max(result_dv[:,3]) # 0.2750

index_tfidf = f1_1.index(max(result_tfidf[:,3])) # 0, th:0.1, prec:0.5946, rec:0.2251
index_dv = f1_2.index(max(result_dv[:,3])) # 10, th:0.2, prec:0.2075, rec:0.4074










# uuidAndIntMapping = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/uuidAndIntMapping.json').read())
# intAndUuidMapping = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/intAndUuidMapping.json').read())
#
# wc_predict_all = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/wc_predict_all.json').read())
# ety_predict_all = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/ety_predict_all.json').read())
# avg_predict_all = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/avg_predict_all.json').read()) # 1393
#
# groundTruthNoTime = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/groundTruthNoTime.json').read()) # truth
#
# # avg results
# avgPrecisionsWC = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/avgPrecisionsWC.json').read())
# avgRecallsWC = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/avgRecallsWC.json').read())
# avgPrecisionsEty = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/avgPrecisionsEty.json').read())
# avgRecallsEty = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/avgRecallsEty.json').read())
# avgPrecisionsAvg = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/avgPrecisionsAvg.json').read())
# avgRecallsAvg = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/avgRecallsAvg.json').read())
#
# def calcAvgPrecRec(truth, predict):
#     '''
#     Calculate average precision and recall over the whole corpus
#     :param truth: dict: {doc_uuid : list_of_correct_uuids}
#     :param predict: dict: {doc_uuid : list_of_predict_uuids}
#     :return: (precisions, recalls, F1s, avgPrecisions, avgRecalls, avgF1s)
#     '''
#     precisions = {}
#     recalls = {}
#     # F1s = {}
#     avgPrecisions = []
#     avgRecalls = []
#     # avgF1s = []
#     for doc_i, predictList_i in predict.items():
#         cnt = 0
#         tot = len(truth[doc_i]) # number of correct answers w.r.t doc_i
#         precision_i = []
#         recall_i = []
#         # F1_i = []
#         for index_j, predict_j in enumerate(predictList_i): # iterate over each prediction
#             if predict_j in truth[doc_i]:
#                 # if this prediction in truth
#                 # recall will grow
#                 # precision will grow
#                 cnt += 1
#                 tmp_recall = float(cnt) / tot
#                 tmp_precision = float(cnt) / (index_j + 1)
#                 recall_i.append(tmp_recall)
#                 precision_i.append(tmp_precision)
#                 # F1_i.append(2 * tmp_precision * tmp_recall / (tmp_precision + tmp_recall))
#             else:
#                 # if this prediction not in truth
#                 # recall stay the same
#                 # usual precision will surely decrease, but keep same as the highest value with same recall
#                 tmp_recall = float(cnt) / tot
#                 if (index_j > 0): tmp_precision = precision_i[index_j - 1]
#                 else: tmp_precision = 0
#                 recall_i.append(tmp_recall)
#                 precision_i.append(tmp_precision)
#                 # F1_i.append(2 * tmp_precision * tmp_recall / (tmp_precision + tmp_recall))
#
#         precisions[doc_i] = precision_i
#         recalls[doc_i] = recall_i
#         # F1s[doc_i] = F1_i
#
#     # compute avg
#     outcome_length = len(precisions.values()[0])
#     num_docs = len(truth)
#     for k in range(outcome_length):
#         precision_sum = 0
#         recall_sum = 0
#         # f1_sum = 0
#         for i,j in precisions.items(): # i is doc id, j is precision list, iterate through each doc here
#             precision_sum += precisions[i][k]
#             recall_sum += recalls[i][k]
#             # f1_sum += F1s[i][k]
#
#         avgPrecisions.append(precision_sum / num_docs)
#         avgRecalls.append(recall_sum / num_docs)
#         # avgF1s.append(f1_sum / num_docs)
#
#     return (precisions, recalls, avgPrecisions, avgRecalls)
#     # return (precisions, recalls, F1s, avgPrecisions, avgRecalls, avgF1s)
#
# # dummy test dataset
# truth = {1:[3,4], 2:[1,3,5], 3:[1], 4:[1,5], 5:[1,2,3,4]}
# predict = {1: [3,5,2,4], 2:[1,3,5,4], 3:[2,4,1,5], 4:[2,3,6,7], 5:[4,7,3,8]}
# (precisions, recalls, avgPrecisions, avgRecalls) = calcAvgPrecRec(truth, predict)
#
# def f1(precision, recall):
#     f1 = []
#     for i,j in enumerate(precision):
#         if (precision[i] + recall[i] != 0):
#             f1.append(2 * precision[i] * recall[i] / float((precision[i] + recall[i])))
#         else: f1.append(-1)
#     return f1
#
# ### DV starts
# ## global
# #
# pairWiseWCDV = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/pairWiseWCDV.json').read())
# num_doc_retrieve = 6278
# wc_dv_predict_all = genNextUuidsOnePhase(pairWiseWCDV, num_doc_retrieve, groundTruthNoTime.keys())
# (precisions1, recalls1, avgPrecisionsWCDV, avgRecallsWCDV) = calcAvgPrecRec(groundTruthNoTime, wc_dv_predict_all)
#
# with open('wc_dv_predict_all.json', 'w') as f:
#     json.dump(wc_dv_predict_all, f)
# with open('avgPrecisionsWCDV.json', 'w') as f:
#     json.dump(avgPrecisionsWCDV, f)
# with open('avgRecallsWCDV.json', 'w') as f:
#     json.dump(avgRecallsWCDV, f)
#
# #
# pairWiseEtyDV = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/pairWiseEtyDV.json').read())
# num_doc_retrieve = 6278
# ety_dv_predict_all = genNextUuidsOnePhase(pairWiseEtyDV, num_doc_retrieve, groundTruthNoTime.keys())
# (precisions1, recalls1, avgPrecisionsEtyDV, avgRecallsEtyDV) = calcAvgPrecRec(groundTruthNoTime, ety_dv_predict_all)
#
# with open('ety_dv_predict_all.json', 'w') as f:
#     json.dump(ety_dv_predict_all, f)
# with open('avgPrecisionsEtyDV.json', 'w') as f:
#     json.dump(avgPrecisionsEtyDV, f)
# with open('avgRecallsEtyDV.json', 'w') as f:
#     json.dump(avgRecallsEtyDV, f)
#
# ####
# pairWiseAvgDV = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/avgPairWiseDV.json').read())
# num_doc_retrieve = 6278
# avg_dv_predict_all = genNextUuidsOnePhase(pairWiseAvgDV, num_doc_retrieve, groundTruthNoTime.keys())
# (precisions1, recalls1, avgPrecisionsAvgDV, avgRecallsAvgDV) = calcAvgPrecRec(groundTruthNoTime, avg_dv_predict_all)
#
# with open('avg_dv_predict_all.json', 'w') as f:
#     json.dump(avg_dv_predict_all, f)
# with open('avgPrecisionsAvgDV.json', 'w') as f:
#     json.dump(avgPrecisionsAvgDV, f)
# with open('avgRecallsAvgDV.json', 'w') as f:
#     json.dump(avgRecallsAvgDV, f)
#
# #
# plt.plot(avgRecallsWCDV, avgPrecisionsWCDV, color="green", linewidth=2.5, linestyle="-", label="word cloud")
# plt.plot(avgRecallsEtyDV, avgPrecisionsEtyDV, color="blue", linewidth=2.5, linestyle="-", label="entity")
# plt.plot(avgRecallsAvgDV, avgPrecisionsAvgDV, color="red", linewidth=2.5, linestyle="-", label="average")
# plt.legend(loc='upper center')
# plt.xlabel('avg_recall')
# plt.ylabel('avg_precision')
# plt.title('doc vec recall - precision')
#
# # f1
# # wc f1
# f1_wc = f1(avgPrecisionsWCDV, avgRecallsWCDV)
# max_ind_wc = f1_wc.index(max(f1_wc))
# max(f1_wc) # 0.1228
#
# # ety f1
# f1_ety = f1(avgPrecisionsEtyDV, avgRecallsEtyDV)
# max_ind_ety = f1_ety.index(max(f1_ety))
# max(f1_ety) # 0.0792
#
# # avg f1
# f1_avg = f1(avgPrecisionsAvgDV, avgRecallsAvgDV)
# max_ind_avg = f1_avg.index(max(f1_avg))
# max(f1_avg) # 0.1386

# # pairWiseWCMTokTfidfInd = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/pairWiseWCMTokTfidfInd.json').read())
# # pairWiseEntityTfidfInd = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/pairWiseEntityTfidfInd.json').read())
# pairWiseTfidfAvgInd = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/avgPairWiseTfidfInd.json').read())
# # pairWiseWCDVInd = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/pairWiseWCDVInd.json').read())
# # pairWiseEtyDVInd = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/pairWiseEtyDVInd.json').read())
# pairWiseAvgDVInd = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/pairWiseAvgDVInd.json').read())

# pairWiseWCMTokTfidfInd = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/pairWiseWCMTokTfidfInd.json').read()) # [0,1]
# pairWiseWCDVInd = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/pairWiseWCDVInd.json').read()) # [-1,+1]

# #
# global_pos_tfidf_proba = []
# for i,j in global_pos_ref:
#     global_pos_tfidf_proba.append(pairWiseWCMTokTfidfInd[str(i)][str(j)])
#
# global_neg_tfidf_proba = []
# for i,j in global_neg_ref:
#     global_neg_tfidf_proba.append(pairWiseWCMTokTfidfInd[str(i)][str(j)])
#
# y_predlog_1_tfidf = global_pos_tfidf_proba + global_neg_tfidf_proba
#
# #
# global_pos_dv_proba = []
# for i,j in global_pos_ref:
#     global_pos_dv_proba.append((pairWiseWCDVInd[str(i)][str(j)] + 1) / 2.0) # w/ normalization
#
# global_neg_dv_proba = []
# for i,j in global_neg_ref:
#     global_neg_dv_proba.append((pairWiseWCDVInd[str(i)][str(j)] + 1) / 2.0) # w/ normalization
#
# y_predlog_1_dv = global_pos_dv_proba + global_neg_dv_proba

# with open('y_predlog_1_dv.json', 'w') as f: json.dump(y_predlog_1_dv, f) # 40262 + 402620, global_pos_ref + global_neg_ref
# with open('y_predlog_1_tfidf.json', 'w') as f: json.dump(y_predlog_1_tfidf, f) # 40262 + 402620, global_pos_ref + global_neg_ref
from __future__ import division
import json
import matplotlib.pyplot as plt

# global
global_10 = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/global_10.json').read())
global_dv = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/global_dv.json').read())
global_walk = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/global_walk.json').read())
global_walk_dv = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/global_walk_dv.json').read())
global_10_dv = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/global_10_dv.json').read())
global_10_walk = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/global_10_walk.json').read())
global_10_walk_dv = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/global_10_walk_dv.json').read())

# [thres, preci, recal, fone]
# plt.plot(global_walk[0][2], global_walk[0][1], linewidth=2, linestyle="-", label="graph2vec")
# plt.plot(global_walk_dv[0][2], global_walk_dv[0][1], linewidth=2, linestyle="-", label="graph2vec + doc2vec-SUP")
# plt.plot(global_10_walk[0][2], global_10_walk[0][1], linewidth=2, linestyle="-", label="10-FEAT + graph2vec")
# plt.plot(global_10_walk_dv[0][2], global_10_walk_dv[0][1], linewidth=2, linestyle="-", label="10-FEAT + graph2vec + doc2vec-SUP")

plt.plot(global_10[0][2], global_10[0][1], linewidth=2, linestyle="-", label="10-FEAT")
plt.plot(global_dv[0][2], global_dv[0][1], linewidth=2, linestyle="-", label="doc2vec-SUP")
plt.plot(global_10_dv[0][2], global_10_dv[0][1], linewidth=2, linestyle="-", label="10-FEAT + doc2vec-SUP")

plt.legend(loc='upper right')
plt.xlabel('recall', fontsize=20)
plt.ylabel('precision', fontsize=20)
plt.title('Recall-Precision Curve', fontsize=20)




















# partition
# part1_10_walk_dv = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/part1_10_walk_dv.json').read())
# part2_10_walk_dv = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/part2_10_walk_dv.json').read())
# part3_10_walk_dv = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/part3_10_walk_dv.json').read())
# part4_10_walk_dv = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/part4_10_walk_dv.json').read())
# part5_10_walk_dv = json.loads(open(r'/Users/ZhangHaotian/desktop/LO/part5_10_walk_dv.json').read())

# create x axis to be 0.1:0.01:0.99
# x_axis = []
# initial = 0.1
# while initial <= 0.99:
#     x_axis.append(initial)
#     initial += 0.01
#     initial = round(initial, 2)




















import os
import re
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



data_time_latency_pf_kf_ekf_num_humans = """{

2025-12-04 07:56:16 traning  worker:1, agv&box:1, env_len:1818, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00095 Fat_predict_loss:0.00123 Fat_coe_accu:0.0425 Rec_coe_accu:0.0229

2025-12-04 07:56:16 PF inference time step: 6.35247812806183e-05, KF inference time step: 5.010737575451271e-05, EKF inference time step: 3.18101268134626e-05

2025-12-04 07:56:26 evaluate  worker:1, agv&box:1, env_len:1554, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000943 Fat_predict_loss:0.00112 Fat_coe_accu:0.0317 Rec_coe_accu:0.0515

2025-12-04 07:56:26 PF inference time step: 6.41873751214419e-05, KF inference time step: 4.891385749807076e-05, EKF inference time step: 3.0941637940081546e-05

2025-12-04 07:56:34 evaluate  worker:1, agv&box:1, env_len:1587, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000809 Fat_predict_loss:0.00094 Fat_coe_accu:0.0315 Rec_coe_accu:0.0259

2025-12-04 07:56:34 PF inference time step: 6.369576937910439e-05, KF inference time step: 4.90986031410299e-05, EKF inference time step: 3.088234202547801e-05

2025-12-04 07:56:42 evaluate  worker:1, agv&box:1, env_len:1677, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000812 Fat_predict_loss:0.000932 Fat_coe_accu:0.0297 Rec_coe_accu:0.0214

2025-12-04 07:56:42 PF inference time step: 6.483293532187837e-05, KF inference time step: 4.92514210509344e-05, EKF inference time step: 3.105298919768723e-05

2025-12-04 07:56:52 evaluate  worker:1, agv&box:1, env_len:1613, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00113 Fat_predict_loss:0.00124 Fat_coe_accu:0.0309 Rec_coe_accu:0.0259

2025-12-04 07:56:52 PF inference time step: 6.499677730212653e-05, KF inference time step: 4.9592690370279446e-05, EKF inference time step: 3.122810392149176e-05

2025-12-04 07:57:00 evaluate  worker:1, agv&box:1, env_len:1562, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000789 Fat_predict_loss:0.000915 Fat_coe_accu:0.0286 Rec_coe_accu:0.0304

2025-12-04 07:57:00 PF inference time step: 6.396120244806463e-05, KF inference time step: 4.895914837279179e-05, EKF inference time step: 3.078301340608682e-05

2025-12-04 07:57:08 evaluate  worker:1, agv&box:1, env_len:1565, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00093 Fat_predict_loss:0.000848 Fat_coe_accu:0.0264 Rec_coe_accu:0.0207

2025-12-04 07:57:08 PF inference time step: 6.479683775490465e-05, KF inference time step: 4.9576591759824906e-05, EKF inference time step: 3.0968212091122956e-05

2025-12-04 07:57:16 evaluate  worker:1, agv&box:1, env_len:1574, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000956 Fat_predict_loss:0.000913 Fat_coe_accu:0.0324 Rec_coe_accu:0.0477

2025-12-04 07:57:16 PF inference time step: 6.429031207267636e-05, KF inference time step: 4.9247069158881256e-05, EKF inference time step: 3.107545336440982e-05

2025-12-04 07:57:26 evaluate  worker:1, agv&box:1, env_len:1693, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00122 Fat_predict_loss:0.00153 Fat_coe_accu:0.0414 Rec_coe_accu:0.0266

2025-12-04 07:57:26 PF inference time step: 6.467565163591523e-05, KF inference time step: 4.917661325507663e-05, EKF inference time step: 3.09647009771249e-05

2025-12-04 07:57:34 evaluate  worker:1, agv&box:1, env_len:1572, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00102 Fat_predict_loss:0.00105 Fat_coe_accu:0.0298 Rec_coe_accu:0.0469

2025-12-04 07:57:34 PF inference time step: 6.41750925369845e-05, KF inference time step: 4.9286519601448194e-05, EKF inference time step: 3.0939057279785776e-05

2025-12-04 07:57:45 evaluate  worker:1, agv&box:2, env_len:1644, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00126 Fat_predict_loss:0.00147 Fat_coe_accu:0.048 Rec_coe_accu:0.0318

2025-12-04 07:57:45 PF inference time step: 6.38760209373604e-05, KF inference time step: 4.944676610385124e-05, EKF inference time step: 3.107446823677007e-05

2025-12-04 07:57:53 evaluate  worker:1, agv&box:2, env_len:1532, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00108 Fat_predict_loss:0.00121 Fat_coe_accu:0.0312 Rec_coe_accu:0.0477

2025-12-04 07:57:53 PF inference time step: 6.46481626027558e-05, KF inference time step: 4.959371009009936e-05, EKF inference time step: 3.1040169240289196e-05

2025-12-04 07:58:03 evaluate  worker:1, agv&box:2, env_len:1508, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00109 Fat_predict_loss:0.00119 Fat_coe_accu:0.0241 Rec_coe_accu:0.0294

2025-12-04 07:58:03 PF inference time step: 6.375293832875056e-05, KF inference time step: 4.9092409149088975e-05, EKF inference time step: 3.10241385543378e-05

2025-12-04 07:58:13 evaluate  worker:1, agv&box:2, env_len:1499, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000912 Fat_predict_loss:0.000939 Fat_coe_accu:0.0426 Rec_coe_accu:0.0434

2025-12-04 07:58:13 PF inference time step: 6.516795702343228e-05, KF inference time step: 5.013788438304573e-05, EKF inference time step: 3.1447394678320705e-05

2025-12-04 07:58:21 evaluate  worker:1, agv&box:2, env_len:1523, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00112 Fat_predict_loss:0.00112 Fat_coe_accu:0.0403 Rec_coe_accu:0.0247

2025-12-04 07:58:21 PF inference time step: 6.471334596409738e-05, KF inference time step: 4.9501990144846084e-05, EKF inference time step: 3.0825502842723424e-05

2025-12-04 07:58:31 evaluate  worker:1, agv&box:2, env_len:1492, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00103 Fat_predict_loss:0.00125 Fat_coe_accu:0.0408 Rec_coe_accu:0.0667

2025-12-04 07:58:31 PF inference time step: 6.40093161657093e-05, KF inference time step: 4.9347372540839554e-05, EKF inference time step: 3.092074841660405e-05

2025-12-04 07:58:39 evaluate  worker:1, agv&box:2, env_len:1455, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000989 Fat_predict_loss:0.000929 Fat_coe_accu:0.0394 Rec_coe_accu:0.0379

2025-12-04 07:58:39 PF inference time step: 6.503305074685218e-05, KF inference time step: 4.9330524562560405e-05, EKF inference time step: 3.0992612805972806e-05

2025-12-04 07:58:49 evaluate  worker:1, agv&box:2, env_len:1484, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000956 Fat_predict_loss:0.000955 Fat_coe_accu:0.0412 Rec_coe_accu:0.0422

2025-12-04 07:58:49 PF inference time step: 6.867917078845906e-05, KF inference time step: 5.1857165570529e-05, EKF inference time step: 3.1846552846245366e-05

2025-12-04 07:58:57 evaluate  worker:1, agv&box:2, env_len:1485, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000731 Fat_predict_loss:0.000837 Fat_coe_accu:0.0403 Rec_coe_accu:0.0351

2025-12-04 07:58:57 PF inference time step: 7.154243160979916e-05, KF inference time step: 5.364289588799782e-05, EKF inference time step: 3.178962553390349e-05

2025-12-04 07:59:05 evaluate  worker:1, agv&box:2, env_len:1484, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000856 Fat_predict_loss:0.00102 Fat_coe_accu:0.0221 Rec_coe_accu:0.0693

2025-12-04 07:59:05 PF inference time step: 6.420576990132704e-05, KF inference time step: 4.983896193799947e-05, EKF inference time step: 3.120391516672633e-05

2025-12-04 07:59:15 evaluate  worker:1, agv&box:3, env_len:1565, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00103 Fat_predict_loss:0.000932 Fat_coe_accu:0.0341 Rec_coe_accu:0.0528

2025-12-04 07:59:15 PF inference time step: 6.297007917215268e-05, KF inference time step: 4.845427248043755e-05, EKF inference time step: 3.0424343511319388e-05

2025-12-04 07:59:25 evaluate  worker:1, agv&box:3, env_len:1593, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00126 Fat_predict_loss:0.00115 Fat_coe_accu:0.0349 Rec_coe_accu:0.018

2025-12-04 07:59:25 PF inference time step: 6.400094568093319e-05, KF inference time step: 4.918472017047544e-05, EKF inference time step: 3.0935297054235696e-05

2025-12-04 07:59:35 evaluate  worker:1, agv&box:3, env_len:1642, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00111 Fat_predict_loss:0.00148 Fat_coe_accu:0.0321 Rec_coe_accu:0.0425

2025-12-04 07:59:35 PF inference time step: 6.306839337738075e-05, KF inference time step: 4.878288064601215e-05, EKF inference time step: 3.06630686342934e-05

2025-12-04 07:59:43 evaluate  worker:1, agv&box:3, env_len:1533, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00119 Fat_predict_loss:0.00106 Fat_coe_accu:0.0341 Rec_coe_accu:0.0556

2025-12-04 07:59:43 PF inference time step: 6.308760894954555e-05, KF inference time step: 4.8584356388855975e-05, EKF inference time step: 3.07133830767866e-05

2025-12-04 07:59:53 evaluate  worker:1, agv&box:3, env_len:1525, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00106 Fat_predict_loss:0.0013 Fat_coe_accu:0.0428 Rec_coe_accu:0.0352

2025-12-04 07:59:53 PF inference time step: 6.407972242011399e-05, KF inference time step: 4.910656663238025e-05, EKF inference time step: 3.121845057753266e-05

2025-12-04 08:00:03 evaluate  worker:1, agv&box:3, env_len:1483, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000891 Fat_predict_loss:0.00113 Fat_coe_accu:0.0399 Rec_coe_accu:0.0448

2025-12-04 08:00:03 PF inference time step: 6.31490914466328e-05, KF inference time step: 4.869992879300365e-05, EKF inference time step: 3.0813551560014144e-05

2025-12-04 08:00:13 evaluate  worker:1, agv&box:3, env_len:1682, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00141 Fat_predict_loss:0.00124 Fat_coe_accu:0.0437 Rec_coe_accu:0.0272

2025-12-04 08:00:13 PF inference time step: 6.341650709952809e-05, KF inference time step: 4.879884005443378e-05, EKF inference time step: 3.070099883924341e-05

2025-12-04 08:00:21 evaluate  worker:1, agv&box:3, env_len:1533, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00111 Fat_predict_loss:0.00118 Fat_coe_accu:0.0255 Rec_coe_accu:0.0289

2025-12-04 08:00:21 PF inference time step: 6.330067708612033e-05, KF inference time step: 4.8661496385747285e-05, EKF inference time step: 3.063562098314617e-05

2025-12-04 08:00:31 evaluate  worker:1, agv&box:3, env_len:1526, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00129 Fat_predict_loss:0.00125 Fat_coe_accu:0.044 Rec_coe_accu:0.0424

2025-12-04 08:00:31 PF inference time step: 6.391289662252404e-05, KF inference time step: 4.905626314495682e-05, EKF inference time step: 3.0825522436539434e-05

2025-12-04 08:00:41 evaluate  worker:1, agv&box:3, env_len:1578, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00123 Fat_predict_loss:0.00125 Fat_coe_accu:0.0515 Rec_coe_accu:0.0367

2025-12-04 08:00:41 PF inference time step: 6.335497809000342e-05, KF inference time step: 4.910606848423019e-05, EKF inference time step: 3.088321371525715e-05

2025-12-04 08:00:47 evaluate  worker:2, agv&box:1, env_len:1095, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00111 Fat_predict_loss:0.0013 Fat_coe_accu:0.0618 Rec_coe_accu:0.0591

2025-12-04 08:00:47 PF inference time step: 6.297995510710973e-05, KF inference time step: 5.063096137895976e-05, EKF inference time step: 3.129880722254923e-05

2025-12-04 08:00:47 PF inference time step: 4.697346796183826e-05, KF inference time step: 3.680573206513984e-05, EKF inference time step: 2.636343376821579e-05

2025-12-04 08:00:55 evaluate  worker:2, agv&box:1, env_len:1260, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00167 Fat_predict_loss:0.00216 Fat_coe_accu:0.0786 Rec_coe_accu:0.0501

2025-12-04 08:00:55 PF inference time step: 6.510322056119404e-05, KF inference time step: 5.0950050354003904e-05, EKF inference time step: 3.13870490543426e-05

2025-12-04 08:00:55 PF inference time step: 4.6057928176153276e-05, KF inference time step: 3.6902654738653275e-05, EKF inference time step: 2.6533717200869604e-05

2025-12-04 08:01:01 evaluate  worker:2, agv&box:1, env_len:1097, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00129 Fat_predict_loss:0.0011 Fat_coe_accu:0.0476 Rec_coe_accu:0.0536

2025-12-04 08:01:01 PF inference time step: 6.412525229162809e-05, KF inference time step: 5.10980785164707e-05, EKF inference time step: 3.1394315178870285e-05

2025-12-04 08:01:01 PF inference time step: 4.6173007029669876e-05, KF inference time step: 3.660192463543596e-05, EKF inference time step: 2.6143673016140434e-05

2025-12-04 08:01:09 evaluate  worker:2, agv&box:1, env_len:1249, max_env_len:2500, finished:True, over_work:False Comp_loss:0.0012 Fat_predict_loss:0.00123 Fat_coe_accu:0.0722 Rec_coe_accu:0.072

2025-12-04 08:01:09 PF inference time step: 6.462594239209727e-05, KF inference time step: 5.041798942083354e-05, EKF inference time step: 3.128456439467981e-05

2025-12-04 08:01:09 PF inference time step: 4.6475397481070794e-05, KF inference time step: 3.7449087688119244e-05, EKF inference time step: 2.667214796006727e-05

2025-12-04 08:01:15 evaluate  worker:2, agv&box:1, env_len:1082, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00145 Fat_predict_loss:0.00186 Fat_coe_accu:0.0708 Rec_coe_accu:0.0645

2025-12-04 08:01:15 PF inference time step: 6.452417638077093e-05, KF inference time step: 5.048524429969999e-05, EKF inference time step: 3.134807685386672e-05

2025-12-04 08:01:15 PF inference time step: 4.578165558481833e-05, KF inference time step: 3.671624083175236e-05, EKF inference time step: 2.634371054150481e-05

2025-12-04 08:01:23 evaluate  worker:2, agv&box:1, env_len:1527, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00134 Fat_predict_loss:0.00156 Fat_coe_accu:0.0838 Rec_coe_accu:0.0735

2025-12-04 08:01:23 PF inference time step: 6.49635200025837e-05, KF inference time step: 5.200304232474317e-05, EKF inference time step: 3.1825367266162206e-05

2025-12-04 08:01:23 PF inference time step: 4.60775521508837e-05, KF inference time step: 3.707978798816933e-05, EKF inference time step: 2.6494284119571668e-05

2025-12-04 08:01:29 evaluate  worker:2, agv&box:1, env_len:1101, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00145 Fat_predict_loss:0.00121 Fat_coe_accu:0.0674 Rec_coe_accu:0.0539

2025-12-04 08:01:29 PF inference time step: 6.600924776858573e-05, KF inference time step: 5.159131187833947e-05, EKF inference time step: 3.1707289000189814e-05

2025-12-04 08:01:29 PF inference time step: 4.7084096342081596e-05, KF inference time step: 3.753197831960292e-05, EKF inference time step: 2.6850999213694227e-05

2025-12-04 08:01:37 evaluate  worker:2, agv&box:1, env_len:1097, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00131 Fat_predict_loss:0.00136 Fat_coe_accu:0.0747 Rec_coe_accu:0.0571

2025-12-04 08:01:37 PF inference time step: 6.499047048980796e-05, KF inference time step: 5.188983684251171e-05, EKF inference time step: 3.210500684127877e-05

2025-12-04 08:01:37 PF inference time step: 4.876735760281058e-05, KF inference time step: 3.821304307379501e-05, EKF inference time step: 2.7212318552550992e-05

2025-12-04 08:01:43 evaluate  worker:2, agv&box:1, env_len:1098, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00134 Fat_predict_loss:0.00163 Fat_coe_accu:0.0628 Rec_coe_accu:0.0693

2025-12-04 08:01:43 PF inference time step: 6.327472749303599e-05, KF inference time step: 5.025920103590779e-05, EKF inference time step: 3.110602470912134e-05

2025-12-04 08:01:43 PF inference time step: 4.562263280316129e-05, KF inference time step: 3.66176629544173e-05, EKF inference time step: 2.6137016727190417e-05

2025-12-04 08:01:49 evaluate  worker:2, agv&box:1, env_len:1089, max_env_len:2500, finished:True, over_work:False Comp_loss:0.0016 Fat_predict_loss:0.00132 Fat_coe_accu:0.0599 Rec_coe_accu:0.038

2025-12-04 08:01:49 PF inference time step: 6.438790383527211e-05, KF inference time step: 5.033872234810373e-05, EKF inference time step: 3.104827292366133e-05

2025-12-04 08:01:49 PF inference time step: 4.617811015595856e-05, KF inference time step: 3.687869090350863e-05, EKF inference time step: 2.6392433159498034e-05

2025-12-04 08:01:57 evaluate  worker:2, agv&box:2, env_len:1216, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00215 Fat_predict_loss:0.00306 Fat_coe_accu:0.0732 Rec_coe_accu:0.0601

2025-12-04 08:01:57 PF inference time step: 6.447144244846545e-05, KF inference time step: 5.0937070658332426e-05, EKF inference time step: 3.145223385409305e-05

2025-12-04 08:01:57 PF inference time step: 4.678984221659208e-05, KF inference time step: 3.6838807557758534e-05, EKF inference time step: 2.636368337430452e-05

2025-12-04 08:02:05 evaluate  worker:2, agv&box:2, env_len:1177, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00171 Fat_predict_loss:0.00138 Fat_coe_accu:0.0566 Rec_coe_accu:0.0826

2025-12-04 08:02:05 PF inference time step: 6.439550103225644e-05, KF inference time step: 5.041509740111579e-05, EKF inference time step: 3.104201797316818e-05

2025-12-04 08:02:05 PF inference time step: 4.702376022922253e-05, KF inference time step: 3.6998431157944254e-05, EKF inference time step: 2.6489377730765784e-05

2025-12-04 08:02:11 evaluate  worker:2, agv&box:2, env_len:1186, max_env_len:2500, finished:True, over_work:False Comp_loss:0.0019 Fat_predict_loss:0.00198 Fat_coe_accu:0.0758 Rec_coe_accu:0.0615

2025-12-04 08:02:11 PF inference time step: 6.456379142619708e-05, KF inference time step: 5.0452065105985027e-05, EKF inference time step: 3.121172597798288e-05

2025-12-04 08:02:11 PF inference time step: 4.5833056933964284e-05, KF inference time step: 3.6950859211347555e-05, EKF inference time step: 2.646064275834854e-05

2025-12-04 08:02:19 evaluate  worker:2, agv&box:2, env_len:1190, max_env_len:2500, finished:True, over_work:False Comp_loss:0.0014 Fat_predict_loss:0.00095 Fat_coe_accu:0.0739 Rec_coe_accu:0.0373

2025-12-04 08:02:19 PF inference time step: 6.433114284226874e-05, KF inference time step: 5.0568981330935696e-05, EKF inference time step: 3.120218004499163e-05

2025-12-04 08:02:19 PF inference time step: 4.690254435819738e-05, KF inference time step: 3.690539287919758e-05, EKF inference time step: 2.638251841569147e-05

2025-12-04 08:02:27 evaluate  worker:2, agv&box:2, env_len:1179, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00209 Fat_predict_loss:0.00228 Fat_coe_accu:0.0858 Rec_coe_accu:0.0462

2025-12-04 08:02:27 PF inference time step: 6.424521267464648e-05, KF inference time step: 5.070449337300049e-05, EKF inference time step: 3.143707708952486e-05

2025-12-04 08:02:27 PF inference time step: 4.691851149179654e-05, KF inference time step: 3.7010288319413794e-05, EKF inference time step: 2.6398740449732294e-05

2025-12-04 08:02:35 evaluate  worker:2, agv&box:2, env_len:1205, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00174 Fat_predict_loss:0.00185 Fat_coe_accu:0.0789 Rec_coe_accu:0.0444

2025-12-04 08:02:35 PF inference time step: 6.459758489458393e-05, KF inference time step: 5.052416156436398e-05, EKF inference time step: 3.1277945427479094e-05

2025-12-04 08:02:35 PF inference time step: 4.6258348646994945e-05, KF inference time step: 3.67170547548666e-05, EKF inference time step: 2.638888062283211e-05

2025-12-04 08:02:41 evaluate  worker:2, agv&box:2, env_len:1229, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00183 Fat_predict_loss:0.00211 Fat_coe_accu:0.0843 Rec_coe_accu:0.0546

2025-12-04 08:02:41 PF inference time step: 6.453053945443608e-05, KF inference time step: 5.167494759121205e-05, EKF inference time step: 3.139811671779251e-05

2025-12-04 08:02:41 PF inference time step: 4.564561525721973e-05, KF inference time step: 3.7058666530326676e-05, EKF inference time step: 2.6461746364807675e-05

2025-12-04 08:02:49 evaluate  worker:2, agv&box:2, env_len:1173, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00163 Fat_predict_loss:0.00141 Fat_coe_accu:0.089 Rec_coe_accu:0.066

2025-12-04 08:02:49 PF inference time step: 6.432850342577376e-05, KF inference time step: 5.0089446683267395e-05, EKF inference time step: 3.097124416809863e-05

2025-12-04 08:02:49 PF inference time step: 4.4737935371106235e-05, KF inference time step: 3.647194494066946e-05, EKF inference time step: 2.6224824168797955e-05

2025-12-04 08:02:57 evaluate  worker:2, agv&box:2, env_len:1216, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00164 Fat_predict_loss:0.00146 Fat_coe_accu:0.0813 Rec_coe_accu:0.0443

2025-12-04 08:02:57 PF inference time step: 6.426184585219936e-05, KF inference time step: 5.062806763147053e-05, EKF inference time step: 3.111382064066435e-05

2025-12-04 08:02:57 PF inference time step: 4.664867332107142e-05, KF inference time step: 3.715820218387403e-05, EKF inference time step: 2.6630727868331106e-05

2025-12-04 08:03:05 evaluate  worker:2, agv&box:2, env_len:1192, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00193 Fat_predict_loss:0.00192 Fat_coe_accu:0.05 Rec_coe_accu:0.0467

2025-12-04 08:03:05 PF inference time step: 6.41756009735517e-05, KF inference time step: 5.0594542650568405e-05, EKF inference time step: 3.136904447670751e-05

2025-12-04 08:03:05 PF inference time step: 4.7054666800786984e-05, KF inference time step: 3.741511562526626e-05, EKF inference time step: 2.673948371170351e-05

2025-12-04 08:03:13 evaluate  worker:2, agv&box:3, env_len:1180, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00182 Fat_predict_loss:0.00178 Fat_coe_accu:0.0766 Rec_coe_accu:0.0601

2025-12-04 08:03:13 PF inference time step: 6.499128826593948e-05, KF inference time step: 5.072638139886371e-05, EKF inference time step: 3.131708856356346e-05

2025-12-04 08:03:13 PF inference time step: 4.761582714016154e-05, KF inference time step: 3.7290282168630824e-05, EKF inference time step: 2.675703016378112e-05

2025-12-04 08:03:21 evaluate  worker:2, agv&box:3, env_len:1164, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00163 Fat_predict_loss:0.00194 Fat_coe_accu:0.0982 Rec_coe_accu:0.0378

2025-12-04 08:03:21 PF inference time step: 6.512329750454303e-05, KF inference time step: 5.082043585498718e-05, EKF inference time step: 3.132213841598878e-05

2025-12-04 08:03:21 PF inference time step: 4.692053057483791e-05, KF inference time step: 3.668368886836206e-05, EKF inference time step: 2.6506042152745616e-05

2025-12-04 08:03:29 evaluate  worker:2, agv&box:3, env_len:1171, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00165 Fat_predict_loss:0.00179 Fat_coe_accu:0.0699 Rec_coe_accu:0.0355

2025-12-04 08:03:29 PF inference time step: 6.483193649174718e-05, KF inference time step: 5.0833243574473104e-05, EKF inference time step: 3.139245825685262e-05

2025-12-04 08:03:29 PF inference time step: 4.754546969289153e-05, KF inference time step: 3.750379221954476e-05, EKF inference time step: 2.6737696894410328e-05

2025-12-04 08:03:37 evaluate  worker:2, agv&box:3, env_len:1169, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00164 Fat_predict_loss:0.00172 Fat_coe_accu:0.0909 Rec_coe_accu:0.0485

2025-12-04 08:03:37 PF inference time step: 6.362349687222235e-05, KF inference time step: 5.087493524477969e-05, EKF inference time step: 3.1463502307953844e-05

2025-12-04 08:03:37 PF inference time step: 4.753972649880206e-05, KF inference time step: 3.734238643336235e-05, EKF inference time step: 2.6718381126279805e-05

2025-12-04 08:03:45 evaluate  worker:2, agv&box:3, env_len:1176, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00175 Fat_predict_loss:0.00175 Fat_coe_accu:0.0781 Rec_coe_accu:0.0603

2025-12-04 08:03:45 PF inference time step: 6.463454694164042e-05, KF inference time step: 5.0879660106840586e-05, EKF inference time step: 3.154383224694907e-05

2025-12-04 08:03:45 PF inference time step: 4.561539409922905e-05, KF inference time step: 3.71373429590342e-05, EKF inference time step: 2.6352146044880353e-05

2025-12-04 08:03:53 evaluate  worker:2, agv&box:3, env_len:1163, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00171 Fat_predict_loss:0.00134 Fat_coe_accu:0.0852 Rec_coe_accu:0.0737

2025-12-04 08:03:53 PF inference time step: 6.401077591952638e-05, KF inference time step: 5.046540257854905e-05, EKF inference time step: 3.1234268883847004e-05

2025-12-04 08:03:53 PF inference time step: 4.658387430966772e-05, KF inference time step: 3.6473532534834645e-05, EKF inference time step: 2.6188118131839296e-05

2025-12-04 08:03:59 evaluate  worker:2, agv&box:3, env_len:1162, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00159 Fat_predict_loss:0.00125 Fat_coe_accu:0.0622 Rec_coe_accu:0.0644

2025-12-04 08:03:59 PF inference time step: 6.4147934445826e-05, KF inference time step: 5.034879234694778e-05, EKF inference time step: 3.1249863760811945e-05

2025-12-04 08:03:59 PF inference time step: 4.694178674799645e-05, KF inference time step: 3.6828489188688344e-05, EKF inference time step: 2.6426504072757267e-05

2025-12-04 08:04:07 evaluate  worker:2, agv&box:3, env_len:1191, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00178 Fat_predict_loss:0.0018 Fat_coe_accu:0.0774 Rec_coe_accu:0.0548

2025-12-04 08:04:07 PF inference time step: 6.435139653664692e-05, KF inference time step: 5.070428503950937e-05, EKF inference time step: 3.121141422505543e-05

2025-12-04 08:04:07 PF inference time step: 4.683293583611297e-05, KF inference time step: 3.653309307210491e-05, EKF inference time step: 2.6131757260569996e-05

2025-12-04 08:04:15 evaluate  worker:2, agv&box:3, env_len:1152, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00222 Fat_predict_loss:0.00274 Fat_coe_accu:0.0799 Rec_coe_accu:0.038

2025-12-04 08:04:15 PF inference time step: 6.391397780842251e-05, KF inference time step: 4.955381155014038e-05, EKF inference time step: 3.06458936797248e-05

2025-12-04 08:04:15 PF inference time step: 4.636558393637339e-05, KF inference time step: 3.6585662100050184e-05, EKF inference time step: 2.6220041844579908e-05

2025-12-04 08:04:23 evaluate  worker:2, agv&box:3, env_len:1179, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00169 Fat_predict_loss:0.00175 Fat_coe_accu:0.0818 Rec_coe_accu:0.0642

2025-12-04 08:04:23 PF inference time step: 6.419627518852046e-05, KF inference time step: 5.0579318563446745e-05, EKF inference time step: 3.127954691724316e-05

2025-12-04 08:04:23 PF inference time step: 4.630173738979505e-05, KF inference time step: 3.636904547434929e-05, EKF inference time step: 2.614151531604511e-05

2025-12-04 08:04:31 evaluate  worker:3, agv&box:1, env_len:1066, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00204 Fat_predict_loss:0.00256 Fat_coe_accu:0.0948 Rec_coe_accu:0.0848

2025-12-04 08:04:31 PF inference time step: 6.630295436780404e-05, KF inference time step: 5.128549142805318e-05, EKF inference time step: 3.154953246268725e-05

2025-12-04 08:04:31 PF inference time step: 4.6895995149021975e-05, KF inference time step: 3.773946923118148e-05, EKF inference time step: 2.695516618510348e-05

2025-12-04 08:04:31 PF inference time step: 4.462587453187295e-05, KF inference time step: 3.577262778219541e-05, EKF inference time step: 2.6080442861589212e-05

2025-12-04 08:04:37 evaluate  worker:3, agv&box:1, env_len:1148, max_env_len:2500, finished:True, over_work:False Comp_loss:0.0017 Fat_predict_loss:0.0021 Fat_coe_accu:0.0858 Rec_coe_accu:0.0483

2025-12-04 08:04:37 PF inference time step: 6.55603325740801e-05, KF inference time step: 5.1074534758458156e-05, EKF inference time step: 3.166605786579411e-05

2025-12-04 08:04:37 PF inference time step: 4.708995387113884e-05, KF inference time step: 3.804541631027381e-05, EKF inference time step: 2.733174101400874e-05

2025-12-04 08:04:37 PF inference time step: 4.507481844167676e-05, KF inference time step: 3.5451887376632424e-05, EKF inference time step: 2.6080874200481987e-05

2025-12-04 08:04:45 evaluate  worker:3, agv&box:1, env_len:1088, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00185 Fat_predict_loss:0.0019 Fat_coe_accu:0.0851 Rec_coe_accu:0.0604

2025-12-04 08:04:45 PF inference time step: 6.429588093477137e-05, KF inference time step: 5.144538248286528e-05, EKF inference time step: 3.184224752818837e-05

2025-12-04 08:04:45 PF inference time step: 4.658628912533031e-05, KF inference time step: 3.7917538600809436e-05, EKF inference time step: 2.712734481867622e-05

2025-12-04 08:04:45 PF inference time step: 4.536724265883951e-05, KF inference time step: 3.539069610483506e-05, EKF inference time step: 2.5933717980104336e-05

2025-12-04 08:04:51 evaluate  worker:3, agv&box:1, env_len:1055, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00224 Fat_predict_loss:0.00322 Fat_coe_accu:0.103 Rec_coe_accu:0.0566

2025-12-04 08:04:51 PF inference time step: 6.462544626534267e-05, KF inference time step: 5.177592779222823e-05, EKF inference time step: 3.1657693510371924e-05

2025-12-04 08:04:51 PF inference time step: 4.637026673809612e-05, KF inference time step: 3.721860912738818e-05, EKF inference time step: 2.6475535749824126e-05

2025-12-04 08:04:51 PF inference time step: 4.490088512547208e-05, KF inference time step: 3.4335665228242557e-05, EKF inference time step: 2.5116888832706975e-05

2025-12-04 08:04:57 evaluate  worker:3, agv&box:1, env_len:1060, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00177 Fat_predict_loss:0.00214 Fat_coe_accu:0.0764 Rec_coe_accu:0.0921

2025-12-04 08:04:57 PF inference time step: 6.463730110312408e-05, KF inference time step: 5.1679251328954153e-05, EKF inference time step: 3.1573817415057487e-05

2025-12-04 08:04:57 PF inference time step: 4.6606108827411005e-05, KF inference time step: 3.6406966875184256e-05, EKF inference time step: 2.6002694975655034e-05

2025-12-04 08:04:57 PF inference time step: 4.41198079091198e-05, KF inference time step: 3.5738270237760726e-05, EKF inference time step: 2.5998646358274064e-05

2025-12-04 08:05:05 evaluate  worker:3, agv&box:1, env_len:1060, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00143 Fat_predict_loss:0.00141 Fat_coe_accu:0.0821 Rec_coe_accu:0.0764

2025-12-04 08:05:05 PF inference time step: 6.429541785761995e-05, KF inference time step: 5.139000010940264e-05, EKF inference time step: 3.1395003480731315e-05

2025-12-04 08:05:05 PF inference time step: 4.65215377087863e-05, KF inference time step: 3.667754947014575e-05, EKF inference time step: 2.6076919627639484e-05

2025-12-04 08:05:05 PF inference time step: 4.364477013641933e-05, KF inference time step: 3.504955543662017e-05, EKF inference time step: 2.5499091958099942e-05

2025-12-04 08:05:11 evaluate  worker:3, agv&box:1, env_len:1069, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00158 Fat_predict_loss:0.00166 Fat_coe_accu:0.106 Rec_coe_accu:0.0736

2025-12-04 08:05:11 PF inference time step: 6.409400640189704e-05, KF inference time step: 5.094641497248907e-05, EKF inference time step: 3.138850848177418e-05

2025-12-04 08:05:11 PF inference time step: 4.577190659668425e-05, KF inference time step: 3.720155043285288e-05, EKF inference time step: 2.659939515244972e-05

2025-12-04 08:05:11 PF inference time step: 4.426511901912564e-05, KF inference time step: 3.449999360529701e-05, EKF inference time step: 2.5233339215127857e-05

2025-12-04 08:05:17 evaluate  worker:3, agv&box:1, env_len:1044, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00173 Fat_predict_loss:0.00174 Fat_coe_accu:0.101 Rec_coe_accu:0.0607

2025-12-04 08:05:17 PF inference time step: 6.425951632503349e-05, KF inference time step: 5.036295602148063e-05, EKF inference time step: 3.112344449507322e-05

2025-12-04 08:05:17 PF inference time step: 4.629476773784535e-05, KF inference time step: 3.73250679951518e-05, EKF inference time step: 2.6518814408459425e-05

2025-12-04 08:05:17 PF inference time step: 4.364772774707312e-05, KF inference time step: 3.474745257147427e-05, EKF inference time step: 2.546602738771402e-05

2025-12-04 08:05:25 evaluate  worker:3, agv&box:1, env_len:1111, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00207 Fat_predict_loss:0.00246 Fat_coe_accu:0.0917 Rec_coe_accu:0.0846

2025-12-04 08:05:25 PF inference time step: 6.330024005055535e-05, KF inference time step: 5.183146946286425e-05, EKF inference time step: 3.187705760169523e-05

2025-12-04 08:05:25 PF inference time step: 4.587615534166942e-05, KF inference time step: 3.65241812114561e-05, EKF inference time step: 2.616745839775628e-05

2025-12-04 08:05:25 PF inference time step: 4.444950663431822e-05, KF inference time step: 3.5266850468921406e-05, EKF inference time step: 2.5584824336315467e-05

2025-12-04 08:05:31 evaluate  worker:3, agv&box:1, env_len:1051, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00207 Fat_predict_loss:0.00252 Fat_coe_accu:0.0893 Rec_coe_accu:0.0691

2025-12-04 08:05:31 PF inference time step: 6.67526651631073e-05, KF inference time step: 5.2694135569709464e-05, EKF inference time step: 3.2053574508536325e-05

2025-12-04 08:05:31 PF inference time step: 4.776107142019226e-05, KF inference time step: 3.8020390765537656e-05, EKF inference time step: 2.696920189825724e-05

2025-12-04 08:05:31 PF inference time step: 4.5897050997283095e-05, KF inference time step: 3.573556495325323e-05, EKF inference time step: 2.5881686514610342e-05

2025-12-04 08:05:39 evaluate  worker:3, agv&box:2, env_len:1147, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00219 Fat_predict_loss:0.0024 Fat_coe_accu:0.086 Rec_coe_accu:0.0428

2025-12-04 08:05:39 PF inference time step: 7.157920232106004e-05, KF inference time step: 5.59300470061165e-05, EKF inference time step: 3.2840860129026296e-05

2025-12-04 08:05:39 PF inference time step: 4.98136854629051e-05, KF inference time step: 3.913785440566338e-05, EKF inference time step: 2.7594819937775005e-05

2025-12-04 08:05:39 PF inference time step: 4.774420388179337e-05, KF inference time step: 3.7177292907768054e-05, EKF inference time step: 2.6624100875521913e-05

2025-12-04 08:05:47 evaluate  worker:3, agv&box:2, env_len:949, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00197 Fat_predict_loss:0.00207 Fat_coe_accu:0.104 Rec_coe_accu:0.0662

2025-12-04 08:05:47 PF inference time step: 6.797642803292631e-05, KF inference time step: 5.41757858967756e-05, EKF inference time step: 3.268168522761418e-05

2025-12-04 08:05:47 PF inference time step: 4.912804502079183e-05, KF inference time step: 3.95963264843184e-05, EKF inference time step: 2.7834929455193378e-05

2025-12-04 08:05:47 PF inference time step: 4.708251912927979e-05, KF inference time step: 3.737770217236277e-05, EKF inference time step: 2.706616143406505e-05

2025-12-04 08:05:55 evaluate  worker:3, agv&box:2, env_len:1094, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00212 Fat_predict_loss:0.00211 Fat_coe_accu:0.106 Rec_coe_accu:0.0523

2025-12-04 08:05:55 PF inference time step: 6.596261884002407e-05, KF inference time step: 5.275980863972164e-05, EKF inference time step: 3.246982093267075e-05

2025-12-04 08:05:55 PF inference time step: 4.888626949442806e-05, KF inference time step: 3.8722969278339055e-05, EKF inference time step: 2.733554003225602e-05

2025-12-04 08:05:55 PF inference time step: 4.6322289071091786e-05, KF inference time step: 3.635098772903247e-05, EKF inference time step: 2.6413683917448332e-05

2025-12-04 08:06:03 evaluate  worker:3, agv&box:2, env_len:1150, max_env_len:2500, finished:True, over_work:False Comp_loss:0.0022 Fat_predict_loss:0.00268 Fat_coe_accu:0.0792 Rec_coe_accu:0.0479

2025-12-04 08:06:03 PF inference time step: 7.578746132228686e-05, KF inference time step: 5.891053572944973e-05, EKF inference time step: 3.317812214726987e-05

2025-12-04 08:06:03 PF inference time step: 5.3394566411557404e-05, KF inference time step: 4.0633989417034646e-05, EKF inference time step: 2.827478491741678e-05

2025-12-04 08:06:03 PF inference time step: 4.91351666657821e-05, KF inference time step: 3.76154028851053e-05, EKF inference time step: 2.7199828106424082e-05

2025-12-04 08:06:11 evaluate  worker:3, agv&box:2, env_len:1149, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00255 Fat_predict_loss:0.00272 Fat_coe_accu:0.0662 Rec_coe_accu:0.0577

2025-12-04 08:06:11 PF inference time step: 6.66916318516611e-05, KF inference time step: 5.3214860854717415e-05, EKF inference time step: 3.2012622391689955e-05

2025-12-04 08:06:11 PF inference time step: 4.655304320279778e-05, KF inference time step: 3.722732849386695e-05, EKF inference time step: 2.6536050104285654e-05

2025-12-04 08:06:11 PF inference time step: 4.575229707855676e-05, KF inference time step: 3.530649230000867e-05, EKF inference time step: 2.596915753848455e-05

2025-12-04 08:06:19 evaluate  worker:3, agv&box:2, env_len:1140, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00211 Fat_predict_loss:0.00244 Fat_coe_accu:0.0905 Rec_coe_accu:0.0622

2025-12-04 08:06:19 PF inference time step: 7.167489905106393e-05, KF inference time step: 5.583512155633224e-05, EKF inference time step: 3.2526150084378424e-05

2025-12-04 08:06:19 PF inference time step: 5.012102294386479e-05, KF inference time step: 3.9120306048476904e-05, EKF inference time step: 2.7504511046827885e-05

2025-12-04 08:06:19 PF inference time step: 4.632284766749332e-05, KF inference time step: 3.697453883656284e-05, EKF inference time step: 2.66139967399731e-05

2025-12-04 08:06:27 evaluate  worker:3, agv&box:2, env_len:1148, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00215 Fat_predict_loss:0.00239 Fat_coe_accu:0.0802 Rec_coe_accu:0.0725

2025-12-04 08:06:27 PF inference time step: 6.91440878014116e-05, KF inference time step: 5.38595462094616e-05, EKF inference time step: 3.2007694244384766e-05

2025-12-04 08:06:27 PF inference time step: 4.913935677930453e-05, KF inference time step: 3.865786961146763e-05, EKF inference time step: 2.721564694979465e-05

2025-12-04 08:06:27 PF inference time step: 4.6780508154360674e-05, KF inference time step: 3.613038345496413e-05, EKF inference time step: 2.622064397725494e-05

2025-12-04 08:06:35 evaluate  worker:3, agv&box:2, env_len:1137, max_env_len:2500, finished:True, over_work:False Comp_loss:0.0018 Fat_predict_loss:0.00159 Fat_coe_accu:0.1 Rec_coe_accu:0.0721

2025-12-04 08:06:35 PF inference time step: 7.150879845665334e-05, KF inference time step: 5.591827835445354e-05, EKF inference time step: 3.26052609513282e-05

2025-12-04 08:06:35 PF inference time step: 4.958791908928461e-05, KF inference time step: 3.909666284512509e-05, EKF inference time step: 2.7433234343113558e-05

2025-12-04 08:06:35 PF inference time step: 4.72089755713363e-05, KF inference time step: 3.637529405041966e-05, EKF inference time step: 2.6528837393540086e-05

2025-12-04 08:06:45 evaluate  worker:3, agv&box:2, env_len:1139, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00198 Fat_predict_loss:0.00235 Fat_coe_accu:0.0853 Rec_coe_accu:0.0548

2025-12-04 08:06:45 PF inference time step: 7.225590070367802e-05, KF inference time step: 5.701092694075721e-05, EKF inference time step: 3.3028613484879145e-05

2025-12-04 08:06:45 PF inference time step: 5.1042508199824064e-05, KF inference time step: 3.92925959077187e-05, EKF inference time step: 2.766450943917533e-05

2025-12-04 08:06:45 PF inference time step: 4.679807138819774e-05, KF inference time step: 3.677695544797983e-05, EKF inference time step: 2.6513025151521516e-05

2025-12-04 08:06:53 evaluate  worker:3, agv&box:2, env_len:1136, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00225 Fat_predict_loss:0.00276 Fat_coe_accu:0.0891 Rec_coe_accu:0.0505

2025-12-04 08:06:53 PF inference time step: 6.546059124906299e-05, KF inference time step: 5.234987802908454e-05, EKF inference time step: 3.1825522301902237e-05

2025-12-04 08:06:53 PF inference time step: 4.694537377693284e-05, KF inference time step: 3.681888043040961e-05, EKF inference time step: 2.6152377397241726e-05

2025-12-04 08:06:53 PF inference time step: 4.5652120885714676e-05, KF inference time step: 3.5429924306735184e-05, EKF inference time step: 2.5685194512488136e-05

2025-12-04 08:07:01 evaluate  worker:3, agv&box:3, env_len:1165, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00216 Fat_predict_loss:0.00225 Fat_coe_accu:0.0995 Rec_coe_accu:0.0649

2025-12-04 08:07:01 PF inference time step: 6.875623449235516e-05, KF inference time step: 5.383818957938657e-05, EKF inference time step: 3.200375470992322e-05

2025-12-04 08:07:01 PF inference time step: 4.790187393646895e-05, KF inference time step: 3.8029912203678247e-05, EKF inference time step: 2.6958694785449637e-05

2025-12-04 08:07:01 PF inference time step: 4.62284415576591e-05, KF inference time step: 3.579409848978591e-05, EKF inference time step: 2.575022979867305e-05

2025-12-04 08:07:09 evaluate  worker:3, agv&box:3, env_len:1159, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00278 Fat_predict_loss:0.00332 Fat_coe_accu:0.0974 Rec_coe_accu:0.0787

2025-12-04 08:07:09 PF inference time step: 6.800833398458565e-05, KF inference time step: 5.289025096876854e-05, EKF inference time step: 3.186045688632442e-05

2025-12-04 08:07:09 PF inference time step: 4.72144899541083e-05, KF inference time step: 3.7531280846715076e-05, EKF inference time step: 2.6711109283157626e-05

2025-12-04 08:07:09 PF inference time step: 4.5653146541355187e-05, KF inference time step: 3.56124124206069e-05, EKF inference time step: 2.5823262357012376e-05

2025-12-04 08:07:19 evaluate  worker:3, agv&box:3, env_len:1155, max_env_len:2500, finished:True, over_work:False Comp_loss:0.0021 Fat_predict_loss:0.0025 Fat_coe_accu:0.0851 Rec_coe_accu:0.0573

2025-12-04 08:07:19 PF inference time step: 6.930549423415939e-05, KF inference time step: 5.386773642007407e-05, EKF inference time step: 3.175343269909615e-05

2025-12-04 08:07:19 PF inference time step: 4.677256464442133e-05, KF inference time step: 3.731157872583959e-05, EKF inference time step: 2.6490678002823998e-05

2025-12-04 08:07:19 PF inference time step: 4.537115881453345e-05, KF inference time step: 3.533177561574168e-05, EKF inference time step: 2.5758082732493744e-05

2025-12-04 08:07:27 evaluate  worker:3, agv&box:3, env_len:1165, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00193 Fat_predict_loss:0.00215 Fat_coe_accu:0.0967 Rec_coe_accu:0.0639

2025-12-04 08:07:27 PF inference time step: 6.490408606795283e-05, KF inference time step: 5.129192008481005e-05, EKF inference time step: 3.122219200297998e-05

2025-12-04 08:07:27 PF inference time step: 4.696518566475406e-05, KF inference time step: 3.7132312299867556e-05, EKF inference time step: 2.6315885552009287e-05

2025-12-04 08:07:27 PF inference time step: 4.554899977000486e-05, KF inference time step: 3.533608923654188e-05, EKF inference time step: 2.560656469778953e-05

2025-12-04 08:07:35 evaluate  worker:3, agv&box:3, env_len:1146, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00222 Fat_predict_loss:0.00224 Fat_coe_accu:0.0877 Rec_coe_accu:0.0655

2025-12-04 08:07:35 PF inference time step: 6.677571807648292e-05, KF inference time step: 5.2184542644294354e-05, EKF inference time step: 3.144919976306003e-05

2025-12-04 08:07:35 PF inference time step: 4.786429812978908e-05, KF inference time step: 3.744898458217867e-05, EKF inference time step: 2.6622575823133113e-05

2025-12-04 08:07:35 PF inference time step: 4.460549479379704e-05, KF inference time step: 3.528678188357262e-05, EKF inference time step: 2.569969203875743e-05

2025-12-04 08:07:45 evaluate  worker:3, agv&box:3, env_len:1140, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00182 Fat_predict_loss:0.00229 Fat_coe_accu:0.0915 Rec_coe_accu:0.0495

2025-12-04 08:07:45 PF inference time step: 6.52037168803968e-05, KF inference time step: 5.157914078026487e-05, EKF inference time step: 3.1643164785284744e-05

2025-12-04 08:07:45 PF inference time step: 4.703015611882795e-05, KF inference time step: 3.7684984374464605e-05, EKF inference time step: 2.684760511967174e-05

2025-12-04 08:07:45 PF inference time step: 4.605828670033237e-05, KF inference time step: 3.5663445790608726e-05, EKF inference time step: 2.6036354533413e-05

2025-12-04 08:07:53 evaluate  worker:3, agv&box:3, env_len:1169, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00197 Fat_predict_loss:0.00215 Fat_coe_accu:0.0929 Rec_coe_accu:0.0712

2025-12-04 08:07:53 PF inference time step: 6.514048352009013e-05, KF inference time step: 5.146272168188038e-05, EKF inference time step: 3.16570516933836e-05

2025-12-04 08:07:53 PF inference time step: 4.7724098094617125e-05, KF inference time step: 3.7603239609299814e-05, EKF inference time step: 2.6810362973307012e-05

2025-12-04 08:07:53 PF inference time step: 4.562564748693879e-05, KF inference time step: 3.581295878179467e-05, EKF inference time step: 2.5979059185096602e-05

2025-12-04 08:08:01 evaluate  worker:3, agv&box:3, env_len:1154, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00206 Fat_predict_loss:0.00218 Fat_coe_accu:0.0885 Rec_coe_accu:0.0542

2025-12-04 08:08:01 PF inference time step: 6.53454914456637e-05, KF inference time step: 5.184364484129167e-05, EKF inference time step: 3.173570285843398e-05

2025-12-04 08:08:01 PF inference time step: 4.744818966764718e-05, KF inference time step: 3.76645572470627e-05, EKF inference time step: 2.6823123158367297e-05

2025-12-04 08:08:01 PF inference time step: 4.591933569420561e-05, KF inference time step: 3.564254456624208e-05, EKF inference time step: 2.5879365732599585e-05

2025-12-04 08:08:11 evaluate  worker:3, agv&box:3, env_len:1139, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00253 Fat_predict_loss:0.0024 Fat_coe_accu:0.087 Rec_coe_accu:0.0535

2025-12-04 08:08:11 PF inference time step: 6.742808146891204e-05, KF inference time step: 5.266141012324064e-05, EKF inference time step: 3.188026903803878e-05

2025-12-04 08:08:11 PF inference time step: 4.6907547171226815e-05, KF inference time step: 3.761947939958983e-05, EKF inference time step: 2.6778655684339675e-05

2025-12-04 08:08:11 PF inference time step: 4.6618053848226415e-05, KF inference time step: 3.625364864573759e-05, EKF inference time step: 2.637675606021764e-05

2025-12-04 08:08:25 evaluate  worker:3, agv&box:3, env_len:1161, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00203 Fat_predict_loss:0.00169 Fat_coe_accu:0.0821 Rec_coe_accu:0.0428

2025-12-04 08:08:25 PF inference time step: 6.540842245202142e-05, KF inference time step: 5.244531064686541e-05, EKF inference time step: 3.206678549859361e-05

2025-12-04 08:08:25 PF inference time step: 4.806629447049873e-05, KF inference time step: 3.771387637431615e-05, EKF inference time step: 2.6962451129818046e-05

2025-12-04 08:08:25 PF inference time step: 4.586446500871841e-05, KF inference time step: 3.583301869473716e-05, EKF inference time step: 2.5934232503150067e-05


}"""


data_num_particles_100_to_1000_pf = """{

100


2025-12-04 08:24:00 traning  worker:1, agv&box:1, env_len:1827, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00125 Fat_predict_loss:0.00155 Fat_coe_accu:0.0399 Rec_coe_accu:0.0272

2025-12-04 08:24:00 PF inference time step: 5.845039609878782e-05, KF inference time step: nan, EKF inference time step: nan

2025-12-04 08:24:10 evaluate  worker:1, agv&box:1, env_len:1700, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000852 Fat_predict_loss:0.00094 Fat_coe_accu:0.0309 Rec_coe_accu:0.0214

2025-12-04 08:24:10 PF inference time step: 5.8562475092270794e-05, KF inference time step: nan, EKF inference time step: nan

2025-12-04 08:24:18 evaluate  worker:1, agv&box:1, env_len:1575, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00107 Fat_predict_loss:0.00104 Fat_coe_accu:0.0343 Rec_coe_accu:0.041

2025-12-04 08:24:18 PF inference time step: 5.916050502232143e-05, KF inference time step: nan, EKF inference time step: nan

2025-12-04 08:24:26 evaluate  worker:1, agv&box:1, env_len:1698, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00126 Fat_predict_loss:0.00164 Fat_coe_accu:0.0252 Rec_coe_accu:0.034

2025-12-04 08:24:26 PF inference time step: 5.922370861219714e-05, KF inference time step: nan, EKF inference time step: nan

2025-12-04 08:24:36 evaluate  worker:1, agv&box:1, env_len:1617, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00095 Fat_predict_loss:0.000902 Fat_coe_accu:0.0247 Rec_coe_accu:0.0539

2025-12-04 08:24:36 PF inference time step: 5.8872432862083045e-05, KF inference time step: nan, EKF inference time step: nan

2025-12-04 08:24:44 evaluate  worker:1, agv&box:1, env_len:1586, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000938 Fat_predict_loss:0.000875 Fat_coe_accu:0.0268 Rec_coe_accu:0.0321

2025-12-04 08:24:44 PF inference time step: 5.7906400961954087e-05, KF inference time step: nan, EKF inference time step: nan

2025-12-04 08:24:52 evaluate  worker:1, agv&box:1, env_len:1589, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000853 Fat_predict_loss:0.00107 Fat_coe_accu:0.0281 Rec_coe_accu:0.03

2025-12-04 08:24:52 PF inference time step: 5.911250321500477e-05, KF inference time step: nan, EKF inference time step: nan

2025-12-04 08:25:00 evaluate  worker:1, agv&box:1, env_len:1568, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000807 Fat_predict_loss:0.000788 Fat_coe_accu:0.0276 Rec_coe_accu:0.0345

2025-12-04 08:25:00 PF inference time step: 5.904828407326523e-05, KF inference time step: nan, EKF inference time step: nan

2025-12-04 08:25:08 evaluate  worker:1, agv&box:1, env_len:1692, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00106 Fat_predict_loss:0.00137 Fat_coe_accu:0.0398 Rec_coe_accu:0.0336

2025-12-04 08:25:08 PF inference time step: 5.904114838187576e-05, KF inference time step: nan, EKF inference time step: nan



200

2025-12-04 17:44:25
traning  worker:1, agv&box:1, env_len:1799, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000867 Fat_predict_loss:0.00115 Fat_coe_accu:0.0425 Rec_coe_accu:0.0216
2025-12-04 17:44:25
PF inference time step: 6.029564781676669e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 17:44:33
evaluate  worker:1, agv&box:1, env_len:1578, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000842 Fat_predict_loss:0.000986 Fat_coe_accu:0.0398 Rec_coe_accu:0.0483
2025-12-04 17:44:33
PF inference time step: 5.9931450470318815e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 17:44:41
evaluate  worker:1, agv&box:1, env_len:1551, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000935 Fat_predict_loss:0.000677 Fat_coe_accu:0.0408 Rec_coe_accu:0.0641
2025-12-04 17:44:41
PF inference time step: 6.051389422438208e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 17:44:51
evaluate  worker:1, agv&box:1, env_len:1707, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00108 Fat_predict_loss:0.00105 Fat_coe_accu:0.0301 Rec_coe_accu:0.0187
2025-12-04 17:44:51
PF inference time step: 6.165166451488242e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 17:44:59
evaluate  worker:1, agv&box:1, env_len:1606, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00106 Fat_predict_loss:0.00126 Fat_coe_accu:0.0503 Rec_coe_accu:0.0267
2025-12-04 17:44:59
PF inference time step: 6.063462491350186e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 17:45:07
evaluate  worker:1, agv&box:1, env_len:1548, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000716 Fat_predict_loss:0.000665 Fat_coe_accu:0.0357 Rec_coe_accu:0.0582
2025-12-04 17:45:07
PF inference time step: 6.0996959991849364e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 17:45:15
evaluate  worker:1, agv&box:1, env_len:1573, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000854 Fat_predict_loss:0.000949 Fat_coe_accu:0.0333 Rec_coe_accu:0.0421
2025-12-04 17:45:15
PF inference time step: 5.959812729357157e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 17:45:23
evaluate  worker:1, agv&box:1, env_len:1681, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000882 Fat_predict_loss:0.000804 Fat_coe_accu:0.037 Rec_coe_accu:0.0623
2025-12-04 17:45:23
PF inference time step: 6.022005120889549e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 17:45:33
evaluate  worker:1, agv&box:1, env_len:1693, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00123 Fat_predict_loss:0.00113 Fat_coe_accu:0.025 Rec_coe_accu:0.0331
2025-12-04 17:45:33
PF inference time step: 6.058732926739931e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 17:45:39
evaluate  worker:1, agv&box:1, env_len:1579, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00082 Fat_predict_loss:0.000683 Fat_coe_accu:0.0286 Rec_coe_accu:0.034
2025-12-04 17:45:39
PF inference time step: 6.023141838010917e-05, KF inference time step: nan, EKF inference time step: nan

300

2025-12-04 16:29:46
traning  worker:1, agv&box:1, env_len:1818, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00112 Fat_predict_loss:0.00141 Fat_coe_accu:0.0396 Rec_coe_accu:0.0289
2025-12-04 16:29:46
PF inference time step: 6.12375104125708e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:29:54
evaluate  worker:1, agv&box:1, env_len:1623, max_env_len:2500, finished:True, over_work:False Comp_loss:0.001 Fat_predict_loss:0.000965 Fat_coe_accu:0.0189 Rec_coe_accu:0.0186
2025-12-04 16:29:54
PF inference time step: 6.173748479385988e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:30:04
evaluate  worker:1, agv&box:1, env_len:1566, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00081 Fat_predict_loss:0.000619 Fat_coe_accu:0.0302 Rec_coe_accu:0.048
2025-12-04 16:30:04
PF inference time step: 6.268520769336062e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:30:12
evaluate  worker:1, agv&box:1, env_len:1702, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000895 Fat_predict_loss:0.000915 Fat_coe_accu:0.0396 Rec_coe_accu:0.0469
2025-12-04 16:30:12
PF inference time step: 6.234043493394146e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:30:20
evaluate  worker:1, agv&box:1, env_len:1626, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00115 Fat_predict_loss:0.00131 Fat_coe_accu:0.029 Rec_coe_accu:0.0289
2025-12-04 16:30:20
PF inference time step: 6.175525074075508e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:30:28
evaluate  worker:1, agv&box:1, env_len:1582, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00103 Fat_predict_loss:0.001 Fat_coe_accu:0.0294 Rec_coe_accu:0.0638
2025-12-04 16:30:28
PF inference time step: 6.0956336454254035e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:30:36
evaluate  worker:1, agv&box:1, env_len:1559, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000864 Fat_predict_loss:0.000981 Fat_coe_accu:0.0374 Rec_coe_accu:0.0476
2025-12-04 16:30:36
PF inference time step: 6.12280015535581e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:30:46
evaluate  worker:1, agv&box:1, env_len:1674, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00102 Fat_predict_loss:0.00116 Fat_coe_accu:0.0313 Rec_coe_accu:0.0345
2025-12-04 16:30:46
PF inference time step: 6.022120034822854e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:30:54
evaluate  worker:1, agv&box:1, env_len:1704, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000848 Fat_predict_loss:0.000881 Fat_coe_accu:0.0304 Rec_coe_accu:0.0275
2025-12-04 16:30:54
PF inference time step: 6.17094163043958e-05, KF inference time step: nan, EKF inference time step: nan

400

2025-12-04 16:31:44
traning  worker:1, agv&box:1, env_len:1823, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00129 Fat_predict_loss:0.00166 Fat_coe_accu:0.0408 Rec_coe_accu:0.0231
2025-12-04 16:31:44
PF inference time step: 6.137924801172036e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:31:52
evaluate  worker:1, agv&box:1, env_len:1557, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000825 Fat_predict_loss:0.000892 Fat_coe_accu:0.029 Rec_coe_accu:0.0339
2025-12-04 16:31:52
PF inference time step: 6.116317033614574e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:32:00
evaluate  worker:1, agv&box:1, env_len:1561, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000952 Fat_predict_loss:0.000935 Fat_coe_accu:0.0293 Rec_coe_accu:0.0292
2025-12-04 16:32:00
PF inference time step: 6.22368714200007e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:32:10
evaluate  worker:1, agv&box:1, env_len:1818, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000965 Fat_predict_loss:0.00117 Fat_coe_accu:0.0412 Rec_coe_accu:0.0297
2025-12-04 16:32:10
PF inference time step: 6.126688651912676e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:32:18
evaluate  worker:1, agv&box:1, env_len:1613, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00113 Fat_predict_loss:0.0011 Fat_coe_accu:0.0404 Rec_coe_accu:0.0259
2025-12-04 16:32:18
PF inference time step: 6.158545530145389e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:32:26
evaluate  worker:1, agv&box:1, env_len:1572, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000871 Fat_predict_loss:0.000872 Fat_coe_accu:0.0291 Rec_coe_accu:0.05
2025-12-04 16:32:26
PF inference time step: 6.121290852398666e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:32:34
evaluate  worker:1, agv&box:1, env_len:1569, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000884 Fat_predict_loss:0.000812 Fat_coe_accu:0.036 Rec_coe_accu:0.0349
2025-12-04 16:32:34
PF inference time step: 6.118528853562929e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:32:42
evaluate  worker:1, agv&box:1, env_len:1665, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000878 Fat_predict_loss:0.000849 Fat_coe_accu:0.0164 Rec_coe_accu:0.0412
2025-12-04 16:32:42
PF inference time step: 6.004439459906684e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:32:50
evaluate  worker:1, agv&box:1, env_len:1701, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00105 Fat_predict_loss:0.000856 Fat_coe_accu:0.0398 Rec_coe_accu:0.0317
2025-12-04 16:32:50
PF inference time step: 6.138234191751004e-05, KF inference time step: nan, EKF inference time step: nan

500

2025-12-04 16:33:45
traning  worker:1, agv&box:1, env_len:1796, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000812 Fat_predict_loss:0.0011 Fat_coe_accu:0.0402 Rec_coe_accu:0.0255
2025-12-04 16:33:45
PF inference time step: 6.697013277253489e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:33:53
evaluate  worker:1, agv&box:1, env_len:1557, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000942 Fat_predict_loss:0.000973 Fat_coe_accu:0.0354 Rec_coe_accu:0.0224
2025-12-04 16:33:53
PF inference time step: 6.428588715285618e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:34:01
evaluate  worker:1, agv&box:1, env_len:1587, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000903 Fat_predict_loss:0.00089 Fat_coe_accu:0.0358 Rec_coe_accu:0.0271
2025-12-04 16:34:01
PF inference time step: 6.36153951259101e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:34:11
evaluate  worker:1, agv&box:1, env_len:1697, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00108 Fat_predict_loss:0.00158 Fat_coe_accu:0.0189 Rec_coe_accu:0.0116
2025-12-04 16:34:11
PF inference time step: 6.407755715185288e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:34:19
evaluate  worker:1, agv&box:1, env_len:1664, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000963 Fat_predict_loss:0.00101 Fat_coe_accu:0.028 Rec_coe_accu:0.0497
2025-12-04 16:34:19
PF inference time step: 6.406066509393546e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:34:27
evaluate  worker:1, agv&box:1, env_len:1634, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000679 Fat_predict_loss:0.000552 Fat_coe_accu:0.0419 Rec_coe_accu:0.0445
2025-12-04 16:34:27
PF inference time step: 6.306083870634455e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:34:35
evaluate  worker:1, agv&box:1, env_len:1565, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000884 Fat_predict_loss:0.000791 Fat_coe_accu:0.0394 Rec_coe_accu:0.0304
2025-12-04 16:34:35
PF inference time step: 6.256347266249002e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:34:43
evaluate  worker:1, agv&box:1, env_len:1564, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000919 Fat_predict_loss:0.000789 Fat_coe_accu:0.0307 Rec_coe_accu:0.0479
2025-12-04 16:34:43
PF inference time step: 6.415060414072803e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:34:51
evaluate  worker:1, agv&box:1, env_len:1707, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00124 Fat_predict_loss:0.00114 Fat_coe_accu:0.0232 Rec_coe_accu:0.0305
2025-12-04 16:34:51
PF inference time step: 6.481982422768895e-05, KF inference time step: nan, EKF inference time step: nan

600

2025-12-04 16:37:03
traning  worker:1, agv&box:1, env_len:1792, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000946 Fat_predict_loss:0.00122 Fat_coe_accu:0.0417 Rec_coe_accu:0.0259
2025-12-04 16:37:03
PF inference time step: 6.317906081676483e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:37:11
evaluate  worker:1, agv&box:1, env_len:1678, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000893 Fat_predict_loss:0.000685 Fat_coe_accu:0.0322 Rec_coe_accu:0.0551
2025-12-04 16:37:11
PF inference time step: 6.524413930645148e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:37:21
evaluate  worker:1, agv&box:1, env_len:1633, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000795 Fat_predict_loss:0.000985 Fat_coe_accu:0.0473 Rec_coe_accu:0.0549
2025-12-04 16:37:21
PF inference time step: 6.480576140852844e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:37:29
evaluate  worker:1, agv&box:1, env_len:1703, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00118 Fat_predict_loss:0.00114 Fat_coe_accu:0.0338 Rec_coe_accu:0.0276
2025-12-04 16:37:29
PF inference time step: 6.298464463167308e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:37:37
evaluate  worker:1, agv&box:1, env_len:1639, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000965 Fat_predict_loss:0.00118 Fat_coe_accu:0.0409 Rec_coe_accu:0.02
2025-12-04 16:37:37
PF inference time step: 6.317015898671363e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:37:45
evaluate  worker:1, agv&box:1, env_len:1558, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000882 Fat_predict_loss:0.000953 Fat_coe_accu:0.0344 Rec_coe_accu:0.0671
2025-12-04 16:37:45
PF inference time step: 6.497411274940579e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:37:53
evaluate  worker:1, agv&box:1, env_len:1553, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000953 Fat_predict_loss:0.000923 Fat_coe_accu:0.0325 Rec_coe_accu:0.0392
2025-12-04 16:37:53
PF inference time step: 6.47750886424618e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:38:01
evaluate  worker:1, agv&box:1, env_len:1593, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000838 Fat_predict_loss:0.000839 Fat_coe_accu:0.0249 Rec_coe_accu:0.0179
2025-12-04 16:38:01
PF inference time step: 6.356975677547706e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:38:09
evaluate  worker:1, agv&box:1, env_len:1714, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00106 Fat_predict_loss:0.000981 Fat_coe_accu:0.0406 Rec_coe_accu:0.0302
2025-12-04 16:38:09
PF inference time step: 6.40710287639391e-05, KF inference time step: nan, EKF inference time step: nan

700
2025-12-04 16:39:22
traning  worker:1, agv&box:1, env_len:1818, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00112 Fat_predict_loss:0.00145 Fat_coe_accu:0.0413 Rec_coe_accu:0.0275
2025-12-04 16:39:22
PF inference time step: 6.421446406801936e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:39:30
evaluate  worker:1, agv&box:1, env_len:1560, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000992 Fat_predict_loss:0.0012 Fat_coe_accu:0.0312 Rec_coe_accu:0.0451
2025-12-04 16:39:30
PF inference time step: 6.475326342460436e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:39:38
evaluate  worker:1, agv&box:1, env_len:1559, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000901 Fat_predict_loss:0.000918 Fat_coe_accu:0.0399 Rec_coe_accu:0.0257
2025-12-04 16:39:38
PF inference time step: 6.427009416131196e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:39:46
evaluate  worker:1, agv&box:1, env_len:1693, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00109 Fat_predict_loss:0.00107 Fat_coe_accu:0.0241 Rec_coe_accu:0.0518
2025-12-04 16:39:46
PF inference time step: 6.615953620058929e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:39:56
evaluate  worker:1, agv&box:1, env_len:1625, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000875 Fat_predict_loss:0.00132 Fat_coe_accu:0.0398 Rec_coe_accu:0.0263
2025-12-04 16:39:56
PF inference time step: 6.585766718937801e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:40:04
evaluate  worker:1, agv&box:1, env_len:1559, max_env_len:2500, finished:True, over_work:False Comp_loss:0.001 Fat_predict_loss:0.00097 Fat_coe_accu:0.0245 Rec_coe_accu:0.0474
2025-12-04 16:40:04
PF inference time step: 6.456372063461216e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:40:12
evaluate  worker:1, agv&box:1, env_len:1551, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00106 Fat_predict_loss:0.000926 Fat_coe_accu:0.0331 Rec_coe_accu:0.0374
2025-12-04 16:40:12
PF inference time step: 6.538633374841962e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:40:20
evaluate  worker:1, agv&box:1, env_len:1563, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00078 Fat_predict_loss:0.000756 Fat_coe_accu:0.035 Rec_coe_accu:0.0516
2025-12-04 16:40:20
PF inference time step: 6.613469017062977e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:40:28
evaluate  worker:1, agv&box:1, env_len:1682, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000957 Fat_predict_loss:0.00116 Fat_coe_accu:0.0438 Rec_coe_accu:0.0274
2025-12-04 16:40:28
PF inference time step: 6.466047942425776e-05, KF inference time step: nan, EKF inference time step: nan


800
2025-12-04 16:41:36
traning  worker:1, agv&box:1, env_len:1808, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000955 Fat_predict_loss:0.00118 Fat_coe_accu:0.0405 Rec_coe_accu:0.0286
2025-12-04 16:41:36
PF inference time step: 6.532510824963055e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:41:44
evaluate  worker:1, agv&box:1, env_len:1587, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000831 Fat_predict_loss:0.000918 Fat_coe_accu:0.0261 Rec_coe_accu:0.0272
2025-12-04 16:41:44
PF inference time step: 6.598215397604622e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:41:52
evaluate  worker:1, agv&box:1, env_len:1560, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000883 Fat_predict_loss:0.000771 Fat_coe_accu:0.027 Rec_coe_accu:0.0303
2025-12-04 16:41:52
PF inference time step: 6.64049234145727e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:42:00
evaluate  worker:1, agv&box:1, env_len:1717, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00133 Fat_predict_loss:0.00142 Fat_coe_accu:0.0439 Rec_coe_accu:0.0358
2025-12-04 16:42:00
PF inference time step: 6.529350369630613e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:42:08
evaluate  worker:1, agv&box:1, env_len:1629, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00102 Fat_predict_loss:0.00085 Fat_coe_accu:0.0335 Rec_coe_accu:0.06
2025-12-04 16:42:08
PF inference time step: 6.598457373893648e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:42:18
evaluate  worker:1, agv&box:1, env_len:1680, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000946 Fat_predict_loss:0.000809 Fat_coe_accu:0.0308 Rec_coe_accu:0.0495
2025-12-04 16:42:18
PF inference time step: 6.422726880936396e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:42:26
evaluate  worker:1, agv&box:1, env_len:1558, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00115 Fat_predict_loss:0.00105 Fat_coe_accu:0.0416 Rec_coe_accu:0.0205
2025-12-04 16:42:26
PF inference time step: 6.570069260407473e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:42:32
evaluate  worker:1, agv&box:1, env_len:1568, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00105 Fat_predict_loss:0.00105 Fat_coe_accu:0.0428 Rec_coe_accu:0.0356
2025-12-04 16:42:32
PF inference time step: 6.617438428255976e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:42:42
evaluate  worker:1, agv&box:1, env_len:1741, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000911 Fat_predict_loss:0.000986 Fat_coe_accu:0.0309 Rec_coe_accu:0.0339
2025-12-04 16:42:42
PF inference time step: 6.450448208742618e-05, KF inference time step: nan, EKF inference time step: nan

900

2025-12-04 16:43:26
traning  worker:1, agv&box:1, env_len:1807, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00109 Fat_predict_loss:0.00141 Fat_coe_accu:0.0415 Rec_coe_accu:0.0193
2025-12-04 16:43:26
PF inference time step: 7.229043953182556e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:43:34
evaluate  worker:1, agv&box:1, env_len:1568, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000877 Fat_predict_loss:0.000854 Fat_coe_accu:0.0351 Rec_coe_accu:0.0567
2025-12-04 16:43:34
PF inference time step: 6.547068454781357e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:43:42
evaluate  worker:1, agv&box:1, env_len:1551, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000882 Fat_predict_loss:0.00103 Fat_coe_accu:0.0395 Rec_coe_accu:0.0406
2025-12-04 16:43:42
PF inference time step: 6.743172228067326e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:43:52
evaluate  worker:1, agv&box:1, env_len:1808, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00116 Fat_predict_loss:0.0016 Fat_coe_accu:0.0399 Rec_coe_accu:0.0412
2025-12-04 16:43:52
PF inference time step: 6.434097226742095e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:44:01
evaluate  worker:1, agv&box:1, env_len:1610, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00112 Fat_predict_loss:0.00113 Fat_coe_accu:0.0293 Rec_coe_accu:0.0157
2025-12-04 16:44:01
PF inference time step: 6.573792570125983e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:44:09
evaluate  worker:1, agv&box:1, env_len:1556, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000789 Fat_predict_loss:0.00104 Fat_coe_accu:0.0427 Rec_coe_accu:0.0528
2025-12-04 16:44:09
PF inference time step: 6.551944811117373e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:44:17
evaluate  worker:1, agv&box:1, env_len:1587, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00088 Fat_predict_loss:0.000893 Fat_coe_accu:0.032 Rec_coe_accu:0.0375
2025-12-04 16:44:17
PF inference time step: 6.544988111797638e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:44:25
evaluate  worker:1, agv&box:1, env_len:1572, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000978 Fat_predict_loss:0.000813 Fat_coe_accu:0.026 Rec_coe_accu:0.0386
2025-12-04 16:44:25
PF inference time step: 6.525404277345303e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:44:33
evaluate  worker:1, agv&box:1, env_len:1737, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00114 Fat_predict_loss:0.00141 Fat_coe_accu:0.0291 Rec_coe_accu:0.0319
2025-12-04 16:44:33
PF inference time step: 6.54916757815311e-05, KF inference time step: nan, EKF inference time step: nan


1000

2025-12-04 16:45:50
traning  worker:1, agv&box:1, env_len:1818, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00106 Fat_predict_loss:0.00139 Fat_coe_accu:0.0411 Rec_coe_accu:0.0314
2025-12-04 16:45:50
PF inference time step: 6.660560045567545e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:45:58
evaluate  worker:1, agv&box:1, env_len:1583, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00092 Fat_predict_loss:0.000957 Fat_coe_accu:0.0259 Rec_coe_accu:0.0589
2025-12-04 16:45:58
PF inference time step: 6.767593451263177e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:46:06
evaluate  worker:1, agv&box:1, env_len:1572, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00091 Fat_predict_loss:0.000794 Fat_coe_accu:0.0233 Rec_coe_accu:0.0483
2025-12-04 16:46:06
PF inference time step: 6.756527733256798e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:46:16
evaluate  worker:1, agv&box:1, env_len:1712, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000993 Fat_predict_loss:0.00118 Fat_coe_accu:0.0425 Rec_coe_accu:0.0248
2025-12-04 16:46:16
PF inference time step: 6.732093953640661e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:46:24
evaluate  worker:1, agv&box:1, env_len:1685, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00105 Fat_predict_loss:0.00084 Fat_coe_accu:0.0397 Rec_coe_accu:0.0321
2025-12-04 16:46:24
PF inference time step: 6.816054663955281e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:46:32
evaluate  worker:1, agv&box:1, env_len:1568, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000914 Fat_predict_loss:0.000805 Fat_coe_accu:0.0282 Rec_coe_accu:0.0293
2025-12-04 16:46:32
PF inference time step: 6.829247790939954e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:46:40
evaluate  worker:1, agv&box:1, env_len:1671, max_env_len:2500, finished:True, over_work:False Comp_loss:0.000877 Fat_predict_loss:0.00105 Fat_coe_accu:0.0303 Rec_coe_accu:0.0252
2025-12-04 16:46:40
PF inference time step: 6.635284652116411e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:46:48
evaluate  worker:1, agv&box:1, env_len:1570, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00102 Fat_predict_loss:0.000934 Fat_coe_accu:0.0246 Rec_coe_accu:0.0291
2025-12-04 16:46:48
PF inference time step: 6.773760364313794e-05, KF inference time step: nan, EKF inference time step: nan
2025-12-04 16:46:58
evaluate  worker:1, agv&box:1, env_len:1710, max_env_len:2500, finished:True, over_work:False Comp_loss:0.00109 Fat_predict_loss:0.000951 Fat_coe_accu:0.0348 Rec_coe_accu:0.0241
2025-12-04 16:46:58
PF inference time step: 6.782199904235482e-05, KF inference time step: nan, EKF inference time step: nan

}
"""

PF_PATTERN = re.compile(r'PF inference time step:\s*([0-9.eE+-]+|nan)', re.IGNORECASE)
KF_PATTERN = re.compile(r'KF inference time step:\s*([0-9.eE+-]+|nan)', re.IGNORECASE)
EKF_PATTERN = re.compile(r'EKF inference time step:\s*([0-9.eE+-]+|nan)', re.IGNORECASE)
FAT_PATTERN = re.compile(r'Fat_coe_accu:([0-9.eE+-]+)', re.IGNORECASE)
REC_PATTERN = re.compile(r'Rec_coe_accu:([0-9.eE+-]+)', re.IGNORECASE)


def _to_float(value: str) -> float:
    try:
        if value is None:
            return np.nan
        value = value.strip()
        if value.lower() == "nan":
            return np.nan
        return float(value)
    except (AttributeError, ValueError):
        return np.nan


def _extract_with_pattern(pattern, text: str) -> float:
    match = pattern.search(text)
    return _to_float(match.group(1)) if match else np.nan


def _nanmean(values):
    if not values:
        return np.nan
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else np.nan


def aggregate_latency_by_humans(raw_text: str):
    human_stats = defaultdict(lambda: {'PF': [], 'KF': [], 'EKF': []})
    current_worker = None

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line in {'{', '}'}:
            continue

        worker_match = re.search(r'worker:(\d+)', line)
        if worker_match:
            current_worker = int(worker_match.group(1))

        if 'PF inference time step' in line and current_worker is not None:
            pf = _extract_with_pattern(PF_PATTERN, line)
            kf = _extract_with_pattern(KF_PATTERN, line)
            ekf = _extract_with_pattern(EKF_PATTERN, line)

            if not np.isnan(pf):
                human_stats[current_worker]['PF'].append(pf)
            if not np.isnan(kf):
                human_stats[current_worker]['KF'].append(kf)
            if not np.isnan(ekf):
                human_stats[current_worker]['EKF'].append(ekf)

    return human_stats


def plot_filter_latency_vs_humans(raw_text: str, save_path: str | None = None):
    human_stats = aggregate_latency_by_humans(raw_text)
    if not human_stats:
        print("No latency data found for PF/KF/EKF comparison.")
        return None

    humans = sorted(human_stats.keys())
    fig, ax = plt.subplots(figsize=(7, 4))
    style_map = {
        'PF': ('#1f77b4', 'o'),
        'KF': ('#ff7f0e', 's'),
        'EKF': ('#2ca02c', '^'),
    }

    for filter_name, (color, marker) in style_map.items():
        means = [_nanmean(human_stats[h][filter_name]) * 1e6 for h in humans]
        ax.plot(
            humans,
            means,
            label=f'{filter_name} latency',
            color=color,
            marker=marker,
            linewidth=2,
        )

    ax.set_xlabel('Number of humans', fontsize=14)
    ax.set_ylabel('Step update latency (µs)', fontsize=14)
    ax.set_title('Filter latency vs. number of humans', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xticks(humans)
    ax.legend(fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"Latency vs. humans plot saved to: {save_path}")

    return fig


def aggregate_pf_particle_metrics(raw_text: str):
    particle_stats = defaultdict(lambda: {'latency': [], 'fat': [], 'rec': []})
    current_particle = None
    pending_metrics = None

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line in {'{', '}'}:
            continue

        if re.fullmatch(r'\d+', line):
            current_particle = int(line)
            pending_metrics = None
            continue

        if 'Fat_coe_accu:' in line and 'Rec_coe_accu:' in line:
            fat_val = _extract_with_pattern(FAT_PATTERN, line)
            rec_val = _extract_with_pattern(REC_PATTERN, line)
            pending_metrics = (fat_val, rec_val)

        if 'PF inference time step' in line and current_particle is not None:
            latency = _extract_with_pattern(PF_PATTERN, line)
            stats = particle_stats[current_particle]
            if not np.isnan(latency):
                stats['latency'].append(latency)
            if pending_metrics:
                fat_val, rec_val = pending_metrics
                if not np.isnan(fat_val):
                    stats['fat'].append(fat_val)
                if not np.isnan(rec_val):
                    stats['rec'].append(rec_val)
            pending_metrics = None

    return particle_stats


def plot_pf_particles_metrics(raw_text: str, save_path: str | None = None):
    particle_stats = aggregate_pf_particle_metrics(raw_text)
    if not particle_stats:
        print("No PF particle data found.")
        return None

    particles = sorted(particle_stats.keys())
    latency_us = [_nanmean(particle_stats[p]['latency']) * 1e6 for p in particles]
    fat_values = [_nanmean(particle_stats[p]['fat']) for p in particles]
    rec_values = [_nanmean(particle_stats[p]['rec']) for p in particles]

    fig, ax_latency = plt.subplots(figsize=(7, 4))
    ax_latency.plot(
        particles,
        latency_us,
        color='#1f77b4',
        marker='o',
        linewidth=2,
        label='Latency (µs)',
    )
    ax_latency.set_xlabel('Number of particles', fontsize=14)
    ax_latency.set_ylabel('PF latency (µs)', fontsize=14, color='#1f77b4')
    ax_latency.tick_params(axis='y', labelcolor='#1f77b4')
    ax_latency.grid(True, linestyle='--', alpha=0.3)

    ax_acc = ax_latency.twinx()
    ax_acc.plot(
        particles,
        fat_values,
        color='#ff7f0e',
        marker='s',
        linewidth=2,
        label='Fatigue coeff. accuracy',
    )
    ax_acc.plot(
        particles,
        rec_values,
        color='#2ca02c',
        marker='^',
        linewidth=2,
        label='Recovery coeff. accuracy',
    )
    ax_acc.set_ylabel('Accuracy', fontsize=14)

    lines, labels = ax_latency.get_legend_handles_labels()
    lines2, labels2 = ax_acc.get_legend_handles_labels()
    ax_latency.legend(lines + lines2, labels + labels2, fontsize=11, loc='best')
    ax_latency.set_title('PF latency & accuracy vs. particle count', fontsize=16)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"PF particle analysis plot saved to: {save_path}")

    return fig


def create_combined_figure(save_path: str | None = None):
    """Create a single figure with (left) PF/KF/EKF latency vs humans and (right) PF latency & accuracy vs particles."""
    human_stats = aggregate_latency_by_humans(data_time_latency_pf_kf_ekf_num_humans)
    particle_stats = aggregate_pf_particle_metrics(data_num_particles_100_to_1000_pf)

    if not human_stats or not particle_stats:
        print("Insufficient data to create combined figure.")
        return None

    particles = sorted(particle_stats.keys())
    latency_us = [_nanmean(particle_stats[p]['latency']) * 1e6 for p in particles]
    fat_values = [_nanmean(particle_stats[p]['fat']) for p in particles]
    rec_values = [_nanmean(particle_stats[p]['rec']) for p in particles]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left subplot: latency vs humans for three filters
    humans = sorted(human_stats.keys())
    style_map = {
        'PF': ('#1f77b4', 'o'),
        'KF': ('#ff7f0e', 's'),
        'EKF': ('#2ca02c', '^'),
    }
    for filter_name, (color, marker) in style_map.items():
        means = [_nanmean(human_stats[h][filter_name]) * 1e6 for h in humans]
        ax1.plot(
            humans,
            means,
            label=f'{filter_name}',
            color=color,
            marker=marker,
            linewidth=2,
        )
    ax1.set_xlabel('Number of humans', fontsize=12)
    ax1.set_ylabel('Step update latency (µs)', fontsize=12)
    ax1.set_title('Filter latency vs. humans', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_xticks(humans)
    ax1.legend(fontsize=10)

    # Right subplot: PF latency + accuracy vs particles (latency on left y, accuracies on right y)
    ax2_lat = ax2
    ax2_lat.plot(
        particles,
        latency_us,
        color='#1f77b4',
        marker='o',
        linewidth=2,
        label='Latency (µs)',
    )
    ax2_lat.set_xlabel('Number of particles', fontsize=12)
    ax2_lat.set_ylabel('PF latency (µs)', fontsize=12, color='#1f77b4')
    ax2_lat.tick_params(axis='y', labelcolor='#1f77b4')
    ax2_lat.grid(True, linestyle='--', alpha=0.3)

    ax2_acc = ax2_lat.twinx()
    ax2_acc.plot(
        particles,
        fat_values,
        color='#ff7f0e',
        marker='s',
        linewidth=2,
        label='Fatigue coeff. accuracy',
    )
    ax2_acc.plot(
        particles,
        rec_values,
        color='#2ca02c',
        marker='^',
        linewidth=2,
        label='Recovery coeff. accuracy',
    )
    ax2_acc.set_ylabel('Accuracy', fontsize=12)

    lines, labels = ax2_lat.get_legend_handles_labels()
    lines2, labels2 = ax2_acc.get_legend_handles_labels()
    ax2_lat.legend(lines + lines2, labels + labels2, fontsize=9, loc='best')
    ax2_lat.set_title('PF latency & accuracy vs. particles', fontsize=14)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"Combined latency figure saved to: {save_path}")

    return fig


if __name__ == '__main__':
    figs_dir = os.path.dirname(__file__)
    combined_path = os.path.join(figs_dir, "filter_latency_combined.pdf")
    create_combined_figure(combined_path)
    plt.show()

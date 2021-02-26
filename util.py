# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 08:48:55 2021

@author: User
"""
import time


'''
utility function that outputs a progress-bar. Suitable for computations, where the amount of iterations is large and
known in advance. "iteration" is the current iterarion, while "total" is the total number of iterations.

An estimated time until the computation is finished will be displayed.
'''


def printProgressBar(iteration, total, starttime=0, decimals=1, length=50, fill='█', printEnd="\r"):
    
    '''
    from: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    '''
    
    if iteration > 0 :
        timeLeft = (total-iteration)*(time.perf_counter() - starttime)/iteration 
        suffix= "   Time left: %3dm %2ds" % (timeLeft//60, timeLeft%60)
    else:
        suffix = ""
        
    percent =  ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r|%s| %s%% %s' % (bar, percent,suffix), end = printEnd)
    if iteration == total:
        print()


model_paths = ["e_coli_core","iAB_RBC_283","iAF692","iAM_Pb448","iCN718","iCN900","iEK1008","iHN637","iIS312","iIT341","iJB785","iLJ478","iNF517","iYO844"]

all_bigg_names = ["e_coli_core","iAB_RBC_283","iAF692","iAF987","iAF1260","iAF1260b","iAM_Pb448","iAM_Pc455","iAM_Pf480","iAM_Pk459","iAM_Pv461","iAPECO1_1312","iAT_PLT_636","iB21_1397","iBWG_1329","ic_1306","iCHOv1","iCHOv1_DG44","iCN718","iCN900","iE2348C_1286","iEC042_1314","iEC1344_C","iEC1349_Crooks","iEC1356_Bl21DE3","iEC1364_W","iEC1368_DH5a","iEC1372_W3110","iEC55989_1330","iECABU_c1320","iECB_1328","iECBD_1354","iECD_1391","iEcDH1_1363","iECDH1ME8569_1439","iECDH10B_1368","iEcE24377_1341","iECED1_1282","iECH74115_1262","iEcHS_1320","iECIAI1_1343","iECIAI39_1322","iECNA114_1301","iECO26_1355","iECO103_1326","iECO111_1330","iECOK1_1307","iEcolC_1368","iECP_1309","iECs_1301","iECS88_1305","iECSE_1348","iECSF_1327","iEcSMS35_1347","iECSP_1301","iECUMN_1333","iEK1008","iEKO11_1354","iETEC_1333","iG2583_1286","iHN637","iIS312","iIS312_Amastigote","iIS312_Epimastigote","iIS312_Trypomastigote","iIT341","iJB785","iJN678","iJN746","iJN1463","iJO1366","iJR904","iLB1027_lipid","iLF82_1304","iLJ478","iML1515","iMM904","iMM1415","iND750","iNF517","iNJ661","iNRG857_1313","iPC815","iRC1080","iS_1188","iSB619","iSbBS512_1146","iSBO_1134","iSDY_1059","iSF_1195","iSFV_1184","iSFxv_1172","iSSON_1240","iSynCJ816","iUMN146_1321","iUMNK88_1353","iUTI89_1310","iWFL_1372","iY75_1357","iYL1228","iYO844","iYS854","iYS1720","iZ_1308","RECON1","Recon3D","STM_v1_0"]


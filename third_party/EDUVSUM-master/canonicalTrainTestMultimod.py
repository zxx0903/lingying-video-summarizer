

from src import utils, models
import json
import numpy as np
import math
import pickle
import time
from scipy.stats import kendalltau, spearmanr
from scipy.stats import rankdata

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import sys

def getOneHotClasses(gt,numClasses):
    oneHot=[]
    for i in range(len(gt)):
        curr=np.zeros(numClasses).astype(int)
        curr[int(gt[i]-1)]=1
        oneHot.append(curr)
    return np.array(oneHot)   

def getVisualFeatures(dataBase,visualFeatureTag,videoName,rsp,inShape):
#    featureDict={}
#    featureDict["vgg16Features"]=[7,7,512]
#    featureDict["inceptionV3Features"]=[8,8,2048]
#    featureDict["efficientNetB0Features"]=[7,7,1280]
    videoFreatures=pickle.load(open(dataBase+visualFeatureTag+"/"+videoName+"_"+visualFeatureTag+".p","rb"))
#    fts1=fts.reshape((videoFreatures.shape[0],featureDict[visualFeatureTag][0],featureDict[visualFeatureTag][1],featureDict[visualFeatureTag][2]))
    videoFreatures=videoFreatures.reshape((videoFreatures.shape[0],inShape[1],rsp))
    return np.mean(videoFreatures,axis=2)

#def getVisualFeatures2(dataBase,visualFeatureTag,videoName,rsp,inShape):
    
def getCombinedTxtFeatures(textFreaturesAll2):
    textFreaturesAll2[0]=textFreaturesAll2[0].reshape((textFreaturesAll2[0].shape[1],textFreaturesAll2[0].shape[2]))
    stackedAll=textFreaturesAll2[0]#textFreaturesAll2[0][:-int(textFreaturesAll2[0].shape[0]*0.10)]
    for i in range(1,len(textFreaturesAll2)):
        textFreaturesAll2[i]=textFreaturesAll2[i].reshape((textFreaturesAll2[i].shape[1],textFreaturesAll2[i].shape[2]))
        part=textFreaturesAll2[i][int(textFreaturesAll2[i].shape[0]*0.10):]#-int(textFreaturesAll2[i].shape[0]*0.10)
        stackedAll=np.concatenate([stackedAll,part],axis=0)  
    return stackedAll

def getCorreshist(hist,feature,featureDim,vidFps,vidFrs):
    ttime=vidFrs/vidFps
    return math.floor(featureDim/ttime)
#    if(feature=="v"):
#        return math.floor(featureDim//ttime)
#    if(feature=="t"):
#        return featureDim//ttime
#    if(feature=="a"):
#        return featureDim//ttime


def getClassesFromOneHot(y,yHat):
    yc=[]
    yHatc=[]
    for i in range(len(y)):
        yc.append(np.argmax(y[i]))
        yHatc.append(np.argmax(yHat[i]))
    return np.array(yc),np.array(yHatc)

def getAccuracy(y,yHat):
    accCnt=0
    for i in range(len(y)):
        if(y[i]==yHat[i]):
            accCnt=+1
    total=len(y)
    if(len(y)==0):
        total=999999999999
    acc=accCnt/total
    return acc

def getUniformSampled(visualFeatures,otherFeatures,vidFps,mod=""):
    nm=visualFeatures.shape[0]//vidFps
    anm=otherFeatures.shape[0]//nm
    asmpr=anm*nm
    if(mod=="a"):
        asmpr=visualFeatures.shape[0]*10
    if(mod=="t"):
        asmpr=visualFeatures.shape[0]
    otherFeaturesArange=np.arange(otherFeatures.shape[0])
    otherFeaturesSmp=otherFeatures[np.random.choice(otherFeaturesArange,asmpr)]
    return otherFeaturesSmp

def getTestResults(model,allFilesTvSumTest,histV,histA,histT,gtbase,modelName,params):
    testMetricsout=[]
    yNyhatByVideo=[]
    testMaeLossLs=[]
    tMseLs=[]
    accLs=[]
    prcLs=[]
    recLs=[]
    fscoreLs=[]
    allMetric=[]
    vidFps=params["vidFps"]
    visualIndicator=params["visualIndicator"]
    audioIndicator=params["audioIndicator"]
    textIndicator=params["textIndicator"]
    for j in range(len(allFilesTvSumTest)):
#        visualFeatures=getVisualFeatures(dataBase,visualFeatureTag,allFilesTvSumTest[j],rsp,inShape)
        visualFeatures=pickle.load(open(dataBase+visualFeatureTag+"/"+allFilesTvSumTest[j]+"_"+visualFeatureTag+".p","rb"))
        visualFeatures=visualFeatures.reshape((visualFeatures.shape[0],visualFeatures.shape[2]))
        gt=pickle.load(open(gtbase+allFilesTvSumTest[j]+gtpostfix,"rb"))
        gtoH=getOneHotClasses(gt,numClasses)
        audioFeatures=pickle.load(open(dataBase+audioFeatureTag+"/"+allFilesTvSumTest[j]+"_0.05_0.025_"+audioFeatureTag+".p","rb"))
        audioFeatures[0]=audioFeatures[0].reshape((audioFeatures[0].shape[1],audioFeatures[0].shape[0]))
        audFeatures=audioFeatures[0]
        audFeatures= getUniformSampled(visualFeatures,audFeatures,vidFps,mod="a")
        
        textFreaturesAll2=pickle.load(open(dataBase+textFeatureTag+"/"+allFilesTvSumTest[j]+"_"+textFeatureTag+".p","rb"))
        txtFeatures=getCombinedTxtFeatures(textFreaturesAll2)
        txtFeatures= getUniformSampled(visualFeatures,txtFeatures,vidFps,mod="t")
        yhatLs=[]
        Y=[]
        maxPoss=(visualFeatures.shape[0]//vidFps)*vidFps
        batch=1
#        maxPoss=(maxPoss//(histV*batch))*(histV*batch)
#        yOrg=0
        for h in range(histV,maxPoss,histV):
            xtest=[]
            b=0
            vInputbatch=[]
            aInputbatch=[]
            tInputbatch=[]
            for b in range(batch):
                if(h+b-1>=len(gtoH)):
                    break
                if(visualIndicator in modelName):
                    vInput=visualFeatures[h+b-histV:h+b]
                    vInput=vInput.reshape((1,vInput.shape[0],vInput.shape[1]))
                    vInputbatch.append(vInput)
                if(audioIndicator in modelName):
                    aInput=audFeatures[((h//histV)*histA)+b-histA:((h//histV)*histA)+b]
                    aInput=aInput.reshape((1,aInput.shape[0],aInput.shape[1]))
                    aInputbatch.append(aInput)
                if(textIndicator in modelName):
                    tInput=txtFeatures[((h//histV)*histT)+b-histT:((h//histV)*histT)+b]
                    tInput=tInput.reshape((1,tInput.shape[0],tInput.shape[1]))
                    tInputbatch.append(tInput)
            if(len(vInputbatch)==0 and len(aInputbatch)==0 and len(tInputbatch)==0):
                break
            if(visualIndicator in modelName):
                vInputbatch=np.array(vInputbatch).reshape((batch,vInput.shape[1],vInput.shape[2]))
            if(audioIndicator in modelName):
                aInputbatch=np.array(aInputbatch).reshape((batch,aInput.shape[1],aInput.shape[2]))
            if(textIndicator in modelName):
                tInputbatch=np.array(tInputbatch).reshape((batch,tInput.shape[1],tInput.shape[2]))
            allInputs=[vInputbatch,aInputbatch,tInputbatch]
            xtest=[x for x in allInputs if len(x)>0]
            if(h+b-1>=len(gtoH) or len(xtest)==0):
                break
            yOrg=gtoH[h+b-1]
            yhat=model.predict(xtest)
            yhatLs.append(yhat[0])
            
            Y.append(yOrg)
    #    print("test: ",j," name: ",allFilesTvSumTest[j])
        y_true, y_pred=getClassesFromOneHot(Y,yhatLs)
        tMse=(np.square(np.array(y_true) - np.array(y_pred))).mean(axis=0)
        tMseLs.append(tMse)
        acc=getAccuracy(y_true, y_pred)
        accLs.append(acc)
        macRes=precision_recall_fscore_support(y_true, y_pred, average='macro')
        micRes=precision_recall_fscore_support(y_true, y_pred, average='micro')
        wRes=precision_recall_fscore_support(y_true, y_pred, average='weighted')
        pS=spearmanr(np.array(y_true),np.array(y_pred))[0]
        kT=kendalltau(rankdata(-np.array(y_true)), rankdata(-np.array(y_pred)))[0]
        allMetric.append([tMse,acc,macRes[0],macRes[1],macRes[2],micRes[0],micRes[1],micRes[2],wRes[0],wRes[1],wRes[2],kT,pS])
        yNyhatByVideo.append([allFilesTvSumTest[j],Y,yhatLs])
    return allMetric,yNyhatByVideo


#params=json.load(open("parameters.json","r"))
print("all args on console: ",sys.argv)
params=[]
if(len(sys.argv)>1):
    params=json.load(open(sys.argv[1],"r"))
else:
    params=json.load(open("parameters.json","r"))


gtbase=params["gtbase"]#"./data/gtAvg/"
gtpostfix=params["gtpostfix"]#"_smpGt.p"

baseout=params["output"]

dataBase=params["dataBase"]#"./data/"


histLs=params["histLs"]

epochs=params["epochs"]
#epochs=5
savePoint=epochs-1

batch=params["batch"]
#batch=2
data=params["dataName"]#"TvSum"





optimizerNameLs=params["optimizerNameLs"]#["adam"]
activationNameLs=params["activationNameLs"]#["softmax"] #sigmoid #softmax



totalMod=len(optimizerNameLs)*len(activationNameLs)
modeCount=0

allFilesTvSumTrain=utils.getVideoList(params["trainVideoList"])
allFilesTvSumTest=utils.getVideoList(params["testVideoList"])




eta=params["eta"] #0.00000001
stopCount=params["stopCount"] #5

lsAllTrainMse0=[]
lsAllTrainMse1=[]
lsAllyNyhatbyvideo=[]
lsAllTestMse=[]
lsAllTestLsMse=[]
lsAllTime=[]
paramLs=[]
maeByVidLs=[]

acth=params["activationHidden"]#"tanh" #relu # tanh
actf=params["activationFinal"]

gt=pickle.load(open(gtbase+allFilesTvSumTrain[0]+gtpostfix,"rb"))
numClasses=params["numClasses"]
gtoH=getOneHotClasses(gt,numClasses)


#allFilesTvSumTest=allFilesTvSumTrain[:2]
vidFps=params["vidFps"]
visualIndicator=params["visualIndicator"]
audioIndicator=params["audioIndicator"]
textIndicator=params["textIndicator"]
visualFeatureTags=params["visualFeatureTags"]
audioFeatureTags=params["audioFeatureTags"]
textFeatureTags=params["textFeatureTags"]


modelNames=params["models"]

 
for modelName in modelNames:
    for visualFeatureTag in visualFeatureTags:
        for audioFeatureTag in audioFeatureTags:
            for textFeatureTag in textFeatureTags:
                visualFeatures=pickle.load(open(dataBase+visualFeatureTag+"/"+allFilesTvSumTrain[0]+"_"+visualFeatureTag+".p","rb"))
                visualFeatures=visualFeatures.reshape((visualFeatures.shape[0],visualFeatures.shape[2]))
                audioFeatures=pickle.load(open(dataBase+audioFeatureTag+"/"+allFilesTvSumTrain[0]+"_0.05_0.025_"+audioFeatureTag+".p","rb"))
                audioFeatures[0]=audioFeatures[0].reshape((audioFeatures[0].shape[1],audioFeatures[0].shape[0]))
                audFeatures=audioFeatures[0]
                audFeatures= getUniformSampled(visualFeatures,audFeatures,vidFps,mod="a")
                textFreaturesAll2=pickle.load(open(dataBase+textFeatureTag+"/"+allFilesTvSumTrain[0]+"_"+textFeatureTag+".p","rb"))
                txtFeatures=textFreaturesAll2
                txtFeatures=getCombinedTxtFeatures(textFreaturesAll2)
                txtFeatures= getUniformSampled(visualFeatures,txtFeatures,vidFps,mod="t")
                for hist in histLs:
                    histV=getCorreshist(hist,feature="v",featureDim=visualFeatures.shape[0],vidFps=vidFps,vidFrs=visualFeatures.shape[0])
                    histA=getCorreshist(hist,feature="a",featureDim=audFeatures.shape[0],vidFps=vidFps,vidFrs=visualFeatures.shape[0])
                    histT=getCorreshist(hist,feature="t",featureDim=txtFeatures.shape[0],vidFps=vidFps,vidFrs=visualFeatures.shape[0])
                    
                    inputShapeV=[histV,visualFeatures.shape[1]]
                    inputShapeA=[histA,audFeatures.shape[-1]]
                    inputShapeT=[histT,txtFeatures.shape[-1]]
    
                    for opt in optimizerNameLs:
                        modeCount+=1
                        model=models.getModel(params,modelName,inputShapeV,inputShapeT,inputShapeA,opt,actf,acth,numClasses)
                
                        lossEpMaeMse=[[],[]]
                        fscoreTestLs=[]
                        timeTakenLs=[]
                        ls=[]
                        batch=params["batch"]
                        modelType=modelName+"_" +data+"_"+str(epochs)+"_"+str(hist)+"_"+str(batch)+"_"+visualFeatureTag+"_"+audioFeatureTag+"_"+textFeatureTag+"_"+opt+"_"+actf+"_"+acth
                        paramLs.append([modelName,data,str(epochs),str(hist),str(batch),visualFeatureTag,audioFeatureTag,textFeatureTag,opt,actf])
                        print("now in model :",modelType," :",modeCount," out: ",totalMod)
                        print("***************************************************")
                        epCnt=0
                        maxAcc=0
                        for i in range(epochs):
                            strTime=time.time()
                            lossLsMaeMse=[[],[]]
                            lossLsMae=[]
                            for j in range(len(allFilesTvSumTrain)):
                                gt=pickle.load(open(gtbase+allFilesTvSumTrain[j]+gtpostfix,"rb"))
                                gtoH=getOneHotClasses(gt,numClasses)
    #                            if(visualIndicator in modelName):
                                visualFeatures=pickle.load(open(dataBase+visualFeatureTag+"/"+allFilesTvSumTrain[j]+"_"+visualFeatureTag+".p","rb"))
                                visualFeatures=visualFeatures.reshape((visualFeatures.shape[0],visualFeatures.shape[2]))
                                if(audioIndicator in modelName):
                                    audioFeatures=pickle.load(open(dataBase+audioFeatureTag+"/"+allFilesTvSumTrain[j]+"_0.05_0.025_"+audioFeatureTag+".p","rb"))
                                    audioFeatures[0]=audioFeatures[0].reshape((audioFeatures[0].shape[1],audioFeatures[0].shape[0]))
                                    audFeatures=audioFeatures[0]
                                    audFeatures= getUniformSampled(visualFeatures,audFeatures,vidFps,mod="a")
                                if(textIndicator in modelName):
                                    txtFeatures=pickle.load(open(dataBase+textFeatureTag+"/"+allFilesTvSumTrain[j]+"_"+textFeatureTag+".p","rb"))
                                    txtFeatures=getCombinedTxtFeatures(txtFeatures)
                                    txtFeatures= getUniformSampled(visualFeatures,txtFeatures,vidFps,mod="t")
                                maxPoss=(visualFeatures.shape[0]//vidFps)*vidFps
#                                maxPoss=(maxPoss//(histV*batch))*(histV*batch)
                                for h in range(histV,maxPoss,histV*batch):#len(gt)
                                    xtrain=[]
                                    ytrain=[]
                                    b=0
                                    vInputbatch=[]
                                    aInputbatch=[]
                                    tInputbatch=[]
                                    remain=visualFeatures.shape[0]-h
                                    if(remain<=batch):
                                        batch=remain
                                    for b in range(batch):
                                        if(h+b-1>=len(gtoH)):
                                            break
                                        if(visualIndicator in modelName):
                                            vInput=visualFeatures[h+b-histV:h+b]
                                            vInput=vInput.reshape((1,vInput.shape[0],vInput.shape[1]))
                                            vInputbatch.append(vInput)
                                        if(audioIndicator in modelName):
                                            aInput=audFeatures[((h//histV)*histA)+b-histA:((h//histV)*histA)+b]
                                            aInput=aInput.reshape((1,aInput.shape[0],aInput.shape[1]))
                                            aInputbatch.append(aInput)
                                        if(textIndicator in modelName):
                                            tInput=txtFeatures[((h//histV)*histT)+b-histT:((h//histV)*histT)+b]
                                            tInput=tInput.reshape((1,tInput.shape[0],tInput.shape[1]))
                                            tInputbatch.append(tInput)
                                        
                                        ytrain.append(gtoH[h+b-1])#[h+b-3]
                                    if(visualIndicator in modelName):
                                        vInputbatch=np.array(vInputbatch).reshape((len(vInputbatch),vInput.shape[1],vInput.shape[2]))
                                    if(audioIndicator in modelName):
                                        aInputbatch=np.array(aInputbatch).reshape((len(aInputbatch),aInput.shape[1],aInput.shape[2]))
                                    if(textIndicator in modelName):
                                        tInputbatch=np.array(tInputbatch).reshape((len(tInputbatch),tInput.shape[1],tInput.shape[2]))
                                    allInputs=[vInputbatch,aInputbatch,tInputbatch]
                                    if(len(ytrain)==0):
                                        break
                                    xtrain=[x for x in allInputs if len(x)>0]
                    #                    ytrain=np.array([[gt[h+b-1]]])
                                    ytrainB=np.array(ytrain) # cause index -1 which here 1=3
                                    #xtrain=utils.getReshape(xtrain,inShape,modelName,batch)
                                    ls=model.train_on_batch(xtrain,ytrainB)
                                    lossLsMaeMse[0].append(ls[0])
                                    lossLsMaeMse[1].append(ls[1])
                            lossEpMaeMse[0].append(np.mean(lossLsMaeMse[0]))
                            lossEpMaeMse[1].append(np.mean(lossLsMaeMse[1]))
                            if(i>3 and (lossEpMaeMse[0][-1]-lossEpMaeMse[0][-2])<eta):
                                epCnt+=1
                            else:
                                epCnt=0
                            if(lossEpMaeMse[1][-1]>=maxAcc):
                                maxAcc=lossEpMaeMse[1][-1]
                                model.save(baseout+modelType+"_modelMxScr_whole.h5")
                            if(epCnt>=stopCount):
                                model.save(baseout+modelType+"_model_whole.h5")
                                break
                            print("train at epoch: ",i," loss:",[lossEpMaeMse[0][-1],lossEpMaeMse[1][-1]], "timetaken: ",utils.getTimeElapsed(strTime)[0])
                            if(i%savePoint==0 and i>0):
                                model.save(baseout+modelType+"_model_whole.h5")
                            timeTakenLs.append(utils.getTimeElapsed(strTime)[0])
                        allMetric,yNyhatByVideo=getTestResults(model,allFilesTvSumTest,histV,histA,histT,gtbase,modelName,params)
                        allMetricArr=np.array(allMetric)
                        outName=modelType
                        np.savetxt(baseout+outName+"_allMetric.csv", np.array(allMetric), delimiter=",", fmt='%s')
                        #        np.savetxt(baseout+outName+"_allMetric.csv", np.array(allMetric), delimiter=",", fmt='%s')
                        pickle.dump(np.array([lsAllTrainMse0,lsAllTrainMse1]),open(baseout+outName+"_lsAllTrainloss.p","wb"))
                        pickle.dump(np.array(allMetric),open(baseout+outName+"_allMetric.p","wb"))
                        pickle.dump(np.array(yNyhatByVideo),open(baseout+outName+"_yNyhatByVideo.p","wb"))


    



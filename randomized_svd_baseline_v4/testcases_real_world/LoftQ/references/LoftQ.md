LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large
Language Models
∗∗ ∗∗
Yixiao Li , Yifan Yu , Chen Liang, Pengcheng He,
∗
Nikos Karampatziakis, Weizhu Chen, Tuo Zhao
November 29, 2023
Abstract
QuantizationisanindispensabletechniqueforservingLargeLanguageModels(LLMs)
andhasrecentlyfounditswayintoLoRAfine-tuning(Dettmersetal.,2023). Inthisworkwe
focusonthescenariowherequantizationandLoRAfine-tuningareappliedtogetheronapre-
trained model. In such cases it is common to observe a consistent gap in the performance on
downstreamtasksbetweenfullfine-tuningandquantizationplusLoRAfine-tuningapproach.
Inresponse,weproposeLoftQ(LoRA-Fine-Tuning-awareQuantization),anovelquantization
frameworkthatsimultaneouslyquantizesanLLMandfindsaproperlow-rankinitializationfor
LoRAfine-tuning. Suchaninitializationalleviatesthediscrepancybetweenthequantizedand
full-precisionmodelandsignificantlyimprovesgeneralizationindownstreamtasks. Weevaluate
ourmethodonnaturallanguageunderstanding,questionanswering,summarization,andnatural
languagegenerationtasks. Experimentsshowthatourmethodishighlyeffectiveandoutperforms
existingquantizationmethods,especiallyinthechallenging2-bitand2/4-bitmixedprecision
regimes. Thecodeisavailableonhttps://github.com/yxli2123/LoftQ.
1 Introduction
TheadventofPre-trainedLanguageModels(PLMs)hasmarkedatransformativeshiftinthefield
of Natural Language Processing (NLP), offering versatile solutions across various applications
(He et al., 2021b; Lewis et al., 2019; Touvron et al., 2023). They have showcased unparalleled
proficiencyinexecutingavarietyoflanguagetasks,includingNaturalLanguageUnderstanding
(NLU)andNaturalLanguageGeneration(NLG).Thesemodelstypicallyhavemillionsorevenbil-
lionsofparameters,necessitatingsubstantialcomputationalandmemoryrequirements. However,
theextensivecomputationalandmemorydemandsofthesemodelsposesignificantchallenges,
∗ Li, Yu, Liang and Zhao are affiliated with Georgia Tech. He, Karampatziakisand and Chen are affiliated with
MicrosoftAzure.Correspondencetoyixiaoli@gatech.edu,yyu429@gatech.eduandtourzhao@gatech.edu.
**Equalcontributions
1
3202
voN
82
]LC.sc[
4v95680.0132:viXra

especiallyinreal-worlddeploymentswhereresourcesareoftenconstrainedandneedtobeshared
amongmanyusers.
Tomitigatetheextensivestoragerequirementsofpre-trainedmodels,quantizationservesasa
pivotalcompressiontechnique(Zafriretal.,2019;Shenetal.,2020;Baietal.,2022;Dettmersetal.,
2022),convertinghigh-precisionnumericalvaluesintoadiscretesetofvalues. Typically,model
parameters,originallystoredina16-bitfloatformat,aretransformedintoa4-bitintegerformat
throughquantization,resultinginasubstantial75%reductioninstorageoverhead. Additionally,to
facilitatetheadaptationofquantizedpre-trainedmodelstodownstreamtasksefficiently,Low-Rank
Adaptation(LoRA)isaviableapproach(Huetal.,2021). Thistechniqueisaparameter-efficient
fine-tuningmethodtraditionallyappliedtohigh-precisionpre-trainedmodels. Itisbasedonthe
hypothesisthatthedifferencesbetweenfullyfine-tunedweightsandpre-trainedweightsexhibit
low-rankproperties. Thisallowsthesedifferencestoberepresentedusinglow-rankmatrices. Asa
result,theoriginalpre-trainedweightsremainunaltered,withadaptationsconfinedsolelytothese
low-rankmatrices,enablingeffectivetaskadaptation.
Whenquantizingpre-trainedmodels,practitionersoftenconcentrateprimarilyonthequantiza-
tiontechnique,inadvertentlyneglectingtheimportanceofsubsequentLoRAfine-tuning(Dettmers
etal.,2023;Diaoetal.,2023). Forexample,QLoRAinheritsthefixupinitialization(Zhangetal.,
2019)usedinLoRA,which(Dettmersetal.,2023)attacheszeroinitializedlow-rankadapters(see
Section2.3)tothequantizedpre-trainedmodel. Theinevitablediscrepancyintroducedbyquanti-
zationduringtheapproximationoftheoriginalhigh-precisionnumbers,ascenarioparticularly
pronouncedinlow-bitsituationssuchasthe2-bitregime,canadverselyimpacttheinitializationof
LoRAfine-tuning. AsillustratedinFigure1a,thequantizedpre-trainedmodelobtainedbyQLoRA
exhibitsseveredegradationbelowthe3-bitlevel. Thisdeviationininitializationoftenresultsinan
inferiorfine-tuningperformance. AsillustratedinFigure1b,thefine-tuningperformancedropsas
thequantizationbitdecreaseswhenapplyingQLoRA.Moreover,itisnoteworthythatQLoRAfails
belowthe3-bitlevel.
Inthispaper,weintroduceanovelquantizationframework,calledLoRA-Fine-Tuning-aware
Quantization(LoftQ).Itisdesignedspecificallyforpre-trainedmodelsthatrequirequantization
andLoRAfine-tuning. Thisframeworkactivelyintegrateslow-rankapproximation,workingin
tandemwithquantizationtojointlyapproximatetheoriginalhigh-precisionpre-trainedweights.
Thissynergysignificantlyenhancesalignmentwiththeoriginalpre-trainedweightsasillustratedin
Figure2. Consequently,ourmethodprovidesanadvantageousinitializationpointforsubsequent
LoRAfine-tuning,leadingtoimprovementsindownstreamtasks.
Weevaluateourquantizationframeworkbyconductingextensiveexperimentsondownstream
tasks,suchasNLU,questionanswering,summarization,andNLG.ExperimentsshowthatLoftQ
consistentlyoutperformsQLoRAacrossallprecisionlevels. Forinstance,with4-bitquantization,
we achieve a 1.1 and 0.8 gain in Rouge-1 for XSum (Narayan et al., 2018) and CNN/DailyMail
(Hermann et al., 2015), respectively. LoftQ excels particularly in low-bit scenarios and works
2

| 12  |     |             |       |     |     | 7.19 |
| --- | --- | ----------- | ----- | --- | --- | ---- |
|     |     | 11.37 11.48 | 11.50 |     |     | 6.80 |
ytixelpreP fo goL
ytixelpreP fo goL
| 10  |     |     |     | 6   |     |     |
| --- | --- | --- | --- | --- | --- | --- |
8
4
2.99
6
|     |     |     |     | 2 1.63 1.64 | 1.65 1.65 |     |
| --- | --- | --- | --- | ----------- | --------- | --- |
4
| 2.49 2.50 | 2.53 2.53      |          |     |      |                |        |
| --------- | -------------- | -------- | --- | ---- | -------------- | ------ |
| 2         |                |          |     | 0    |                |        |
| 16 8      | 4 3            | 2.5 2.25 | 2   | 16 8 | 4 3 2.5        | 2.25 2 |
|           | Number of Bits |          |     |      | Number of Bits |        |
(a)Pre-trainedLLAMA-2-13bonWikiText-2 (b)Fine-tunedLLAMA-2-13bonWikiText-2
QLoRAperformancewithdifferentbits.
Figure1: Left: QLoRAinitializationofLLAMA-2-13b
on WikiText-2. Right: Apply QLoRA to LLAMA-2-13b on WikiText-2 language modeling task.
Smallerperplexityindicatesbetterperformance.
effectivelywithdifferentquantizationmethods. Forexample,weachieveoveran8%gainonMNLI
(Wang et al., 2019) and more than 10% on SQuADv1.1 (Rajpurkar et al., 2016) with both 2-bit
NormalFloatandthe2-bituniformquantization. Wehavenotseenourapproachperformsworse
thanQLoRA.
14
| LoftQ |     |     |     | LoftQ |     |     |
| ----- | --- | --- | --- | ----- | --- | --- |
60
12
| QLoRA |     |     |     | QLoRA |     |     |
| ----- | --- | --- | --- | ----- | --- | --- |
50
| ycnapercsiD 10 |     |     | ycnapercsiD |     |     |     |
| -------------- | --- | --- | ----------- | --- | --- | --- |
40
8
30
6
20
4
| 2   |     |     |     | 10  |     |     |
| --- | --- | --- | --- | --- | --- | --- |
| 0   |     |     |     | 0   |     |     |
Uniform NormalFloat Uniform NormalFloat Uniform NormalFloat Uniform NormalFloat
| 4bit | 4bit | 2bit 2bit |     | 4bit | 4bit 2bit | 2bit |
| ---- | ---- | --------- | --- | ---- | --------- | ---- |
(a)Spectralnormoftheinitializationdifference (b)Frobeniusnormoftheinitializationdifference
Figure2: InitializationdiscrepancybetweentheLoRAinitializationandtheoriginalpre-trained
weightmatrix,describedbythespectralnormandFrobeniusnormofthedifference.
Theweight
matrixintheabovefiguresisrandomlyselectedinBART-large. Theinitializationisobtainedby
QLoRAandLoftQ,withUniformandNormalFloatquantizationmethodsappliedatboth2-bitand
4-bitlevels. LoftQsuccessfullymitigatesthediscrepancy,especiallyatthe2-bitlevel.
3

2 Background
2.1 TransformerModels
Atransformermodelcontainsasequenceoflayers,whereeachlayerconsistsoftwosub-layers: a
multi-headself-attention(MHA)andafullyconnectedfeedforwardnetwork(FFN)(Vaswanietal.,
∈Rn×d,wherenisthesequencelengthandd
2017). GiventheinputX isthehiddendimensionof
themodel,MHAcomputesthehattentionheadsinparallel:
|     | MHA(X)=Concat(head |     |     | 1 ,...,head | )W  | ,   |     |
| --- | ------------------ | --- | --- | ----------- | --- | --- | --- |
h o
(cid:112)
⊤
| where head | i =Softmax(XW |     | q (XW | k ) / d h )XW | v for | i =1,...,h, |     |
| ---------- | ------------- | --- | ----- | ------------- | ----- | ----------- | --- |
|            |               |     | i     | i             | i     |             |     |
| ∈Rd×d      |               |     |       |               | ∈Rd×d |             |     |
whereW q ,W k ,W v h arequery,key,andvaluematrices,W o istheoutputmatrix,and
i i i
d =d/h. FFNcomprisestwolineartransformationsandanactivationfunction,andisdefinedas
h
FFN(X)=σ(XW +b )W +b ,whereW ∈Rd×d m,W ∈Rd ×d,andσ(·)istheactivationfunction.
| f 1 f | 2   | f   |     | f   | m   |     |     |
| ----- | --- | --- | --- | --- | --- | --- | --- |
| 1     | 2   | 1   |     | 2   |     |     |     |
Aresidualconnectionisusedandfollowedbylayernormalization.
2.2 Quantization
Quantization. Givenahigh-precisionnumber,e.g.,suchas32-bitfloatingpointnumber,XHP∈R,
N-bitquantizationencodesittoanintegerXINT∈{0,1,...,2N −1}. Thisprocesscanbeexpressedas
|     |            |     | (cid:16) | (cid:16) | (cid:17)(cid:17) |     |     |
| --- | ---------- | --- | -------- | -------- | ---------------- | --- | --- |
|     | XINT=round |     | (2N      | −1)F XHP |                  |     | (1) |
,
where F(·): R (cid:55)→ [0,1] is a normalization function. Uniform quantization assumes F(X) = (X−
−X
X )/(X ). Dettmers et al. (2023) proposes 4-bit NormalFloat Quantization (NF4). It
min max min
assumesX ∼N(0,σ2)andhenceF(X)=Φ(X/σ),whereΦ(·)isthecumulativedistributionfunction
ofthestandardnormaldistribution.
AlookuptableT
Dequantization. ,where
(cid:18) (cid:19)
−1 i
|     | T [i]=F |     | ,i  | =0,1,...,2N | −1, |     | (2) |
| --- | ------- | --- | --- | ----------- | --- | --- | --- |
−1
2N
isusedtodecodetheintegerXINT toitssimulatedhigh-precisioncounterpartXD∈R.
Therefore,
thedequantizationcanbeexpressedas
|     |     | XD=T | [XINT]. |     |     |     | (3) |
| --- | --- | ---- | ------- | --- | --- | --- | --- |
Simulated Quantization for Matrices. While it is possible to perform multiplication directly
betweenquantizedrepresentations,itiscommontoapplysimulatedquantizationformatrices(Bai
etal.,2020;Shenetal.,2020). There,quantizedweightmatricesarestoredasencodedintegers
inmemory,andaretemporarilydequantizedtosimulatedhigh-precisionmatricesbythelookup
tablewhenengagedinmultiplicationoperations. Insimulatedquantization,itisonlynecessaryto
analyzethemapfromahigh-precisionmatrixtoasimulatedhigh-precisionmatrix. Wedenote
thisend-to-endprocessbyq (·): Rm×n(cid:55)→Rm×n,whereR :{T [i]∈R|0≤i <2N}.
|     | N   |     |     | N   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
N
4

2.3 Low-RankAdaptation
LoRA(Huetal.,2021)updatestwosmallweightmatricesAandBthatareattachedtoafrozen
pre-trainedweightmatrixW. Hence,alineartransformation,Y =XW,isreformulatedas
⊤
Y =XW +XAB , (4)
whereX ∈Rn×d 1,W ∈Rd 1 ×d 2,A∈Rd 1 ×r,B∈Rd 2 ×r,andr ≪min{d 1 ,d 2 }. Initially,
A∼N(0,σ2), B=0, (5)
so as to align to the pre-trained weights. During the fine-tuning, W is fixed while A and B are
updatedbysomeSGD-typeoptimizationmethod.
It is worth noting that if low-rank adapters A and B are attached to a quantized backbone
⊤
Q =q (W) and are initialized by (5), the starting weight Q+AB is no longer equal to the pre-
N
trainedweightW duetothediscrepancyintroducedbythequantization.
3 Method
WeproposeLoRA-Fine-Tuning-awareQuantization(LoftQ),aquantizationframeworkforLLMs.
It alternatively applies quantization and low-rank approximation to approximate original pre-
trainedweights. ThisquantizationframeworkprovidesapromisinginitializationforLoRAfine-
tuning,whichalleviatesthequantizationdiscrepancyinQLoRAandimprovesgeneralizationin
downstreamtaskssignificantly.
3.1 LoRA-AwareQuantization
We use an N-bit quantized weight Q∈Rd 1 ×d 2 and low-rank approximations A∈Rd 1 ×r,B∈Rd 2 ×r
N
toapproximatetheoriginalhigh-precisionpre-trainedweightW ∈Rd 1 ×d 2 astheinitializationof
LoRA fine-tuning. Specifically, beforefine-tuning, we initializethe network byminimizing the
followingobjective:
(cid:13) (cid:13)
min (cid:13) (cid:13)W −Q−AB ⊤(cid:13) (cid:13) , (6)
Q,A,B F
where∥·∥ denotestheFrobeniousnorm. Thisobjectivein(6)takesLoRAfine-tuningintoconsider-
F
ationbyjointlyoptimizingtheinitialvaluesofthequantizedbackboneQ andlow-rankadapters
A,B. Contrarily,practitionerstypicallyconvertthepre-trainedweightW intoaquantizedweight
Q outright,neglectingthesubsequentLoRAfine-tuningprocess. Thisoversightleadstonotable
performancedegradationindownstreamtasksarisingfromthequantizationdiscrepancy.
5

3.2 AlternatingOptimization
Wesolvetheminimizationproblemin(6)byalternatingbetweenquantizationandsingularvalue
| decomposition(SVD).Tobeginwith,wesetA |     |     |     | ,andB | equalto0. |     |     |     |
| ------------------------------------- | --- | --- | --- | ----- | --------- | --- | --- | --- |
0 0
Atthet-thstep,wequantizethedifferencebetweentheoriginalpre-trainedweight
Quantization.
⊤
W andthelow-rankapproximationA t−1 B fromthelaststeptoobtainthequantizedweightQ
|     |     |     |     | t−1 |     |     |     | t   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
by
⊤
|                                                              |     |     | Q =q | (W −A | B ), |     |     | (7) |
| ------------------------------------------------------------ | --- | --- | ---- | ----- | ---- | --- | --- | --- |
|                                                              |     |     | t    | N t−1 | t−1  |     |     |     |
| whereq (·)mapsahigh-precisionweightmatrixtoaquantizedmatrix. |     |     |      |       |      |     |     |     |
N
Weremarkthatouralgorithmiscompatiblewithdifferentquantizationfunctionsq (·).
We
N
apply NF4 and the uniform quantization in Section 4 as examples. We also remark that Q is
t
not an exact solution of the minimization in (6), given the fixed A B ⊤ , but it is an efficient
t−1 t−1
approximation.
SVD.Afterobtainingthet-thquantizedweightQ ,SVDisappliedtotheresidualofthequantiza-
t
−Q
| tiondenotedbyR | t =W | t by |     |     |     |     |     |     |
| -------------- | ---- | ---- | --- | --- | --- | --- | --- | --- |
(cid:88)d
⊤
|     |     |     | R   | = σ u   | v , |     |     | (8) |
| --- | --- | --- | --- | ------- | --- | --- | --- | --- |
|     |     |     | t   | t,i t,i | t,i |     |     |     |
i=1
where d = min{d ,d }, σ ≥ σ ≥ ... ≥ σ are the singular values of R , u ’s and v ’s are the
|     | 1 2 | t,1 t,2 | t,d |     |     | t t,i | t,i |     |
| --- | --- | ------- | --- | --- | --- | ----- | --- | --- |
associatedleftandrightsingularvectorsofR . Wethenobtainarank-r approximationofR by
|     |     |     |     | t   |     |     |     | t   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
⊤
A B ,where
t t
|     |     |     | √    | √         |         |     |     |     |
| --- | --- | --- | ---- | --------- | ------- | --- | --- | --- |
|     |     |     | A =[ | σ u ,..., | σ u ],  |     |     |     |
|     |     |     | t    | t,1 t,1   | t,r t,r |     |     |     |
|     |     |     | √    | √         |         |     |     |     |
|     |     |     | B =[ | σ v ,..., | σ v ].  |     |     | (9) |
|     |     |     | t    | t,1 t,1   | t,r t,r |     |     |     |
We summarize our method in Algorithm 1. It is worth noting that T = 1 is a special case
whereQ 1 istheexactquantizedweightobtainedbyQLoRA,andlow-rankapproximationsA 1 ,B 1
|     |     |     |     |     | −Q  | sufficient |     |     |
| --- | --- | --- | --- | --- | --- | ---------- | --- | --- |
are obtained by the SVD of the quantization residual W . T =1 is to mitigate the
1
quantizationdiscrepancy,andalternatingoptimizationhelpstofindacloserinitializationtothe
pre-trainedweightW,whichfurtherimprovestheperformance(seeSection3).
WeremarkthatthecomputationalcostofLoftQisnegligiblebecauseitisappliedtoindividual
weightmatricesandthereforecanbeexecutedinparallel. WealsoremarkonecanapplyLoftQonly
oncetoapre-trainedmodelandreusetheinitializationobtainedbyLoftQfordifferentdownstream
tasks.
3.3 ApplyingtoLoRAFine-tuning
×d
|             | ∈Rd 1 | 2                                    |     |     |     | by(1)andalookuptableT |     |     |
| ----------- | ----- | ------------------------------------ | --- | --- | --- | --------------------- | --- | --- |
| WestoretheQ | T     | obtainedbyLoftQusinganintegermatrixM |     |     |     |                       |     |     |
N
by(2). WeinitializethebackbonewiththeintegermatrixM andinitializethelow-rankadapters
| withA ,B | obtainedbyLoftQ. |     |     |     |     |     |     |     |
| -------- | ---------------- | --- | --- | --- | --- | --- | --- | --- |
| T T      |                  |     |     |     |     |     |     |     |
6

Algorithm1LoftQ
input Pre-trainedweightW,targetrankr,N-bitquantizationfunctionq (·),alternatingstepT
N
1: InitializeA 0 ←0,B 0 ←0
2: fort=1toT do
3: ObtainquantizedweightQ t ←q N (W −A t−1 B ⊤ t−1 )
4: Obtainlow-rankapproximationA t ,B t ←SVD(W −Q t )by(9)
5: endfor
output Q ,A ,B
T T T
DuringLoRAfine-tuning,wefreezetheintegerweightM andoptimizethelow-rankadapters
withanefficientoptimizationalgorithm,e.g.,AdamW(LoshchilovandHutter,2017). Inforward
propagation, the integer weight M is temporarily dequantized to the simulated high-precision
weightQ byitslookuptable,asdescribedin(3). Inbackpropagation,gradientsandoptimizer
T
stateareonlyrelatedtolow-rankadaptersA,B,whichreducesconsiderabletrainingcost.
4 Experiments
WeevaluateourmethodonNLUandNLGtasks. WeapplyLoftQforquantizingDeBERTaV3-base
(Heetal.,2021b),BART-large(Lewisetal.,2019),andLLAMA-2series(Touvronetal.,2023).
Implementation Details. Following the prior works of LoRA variants (Zhang et al., 2023; He
et al., 2021a), we freeze all the backbone weight matrices and add low-rank adapters to weight
matrices in MHA and FFN of all layers. We quantize the weight matrices that are attached by
low-rank adapters. All the quantized models and adapters used in this paper are available on
https://huggingface.co/LoftQ. OurimplementationisbasedonpubliclyavailableHuggingface
Transformerscode-base(Paszkeetal.,2019). AlltheexperimentsareconductedonNVIDIAA100
GPUs.
QuantizationMethods. WeapplytwoquantizationmethodstodemonstrateLoftQiscompatible
withdifferentquantizationfunctions:
• Uniform quantization is a classic quantization method. It uniformly divides a continuous
intervalinto2N categoriesandstoresalocalmaximumabsolutevaluefordequantization.
• NF4 and its 2-bit variant NF2 are quantization methods used in QLoRA (Dettmers et al.,
2023). Theyassumethatthehigh-precisionvaluesaredrawnfromaGaussiandistribution
andmapthesevaluestodiscreteslotsthathaveequalprobability.
Weperform2-bitand4-bitquantizationonallmodels,achievingcompressionratiosof25-30%and
15-20%atthe4-bitand2-bitlevels,respectively. Thecompressionratiosandtrainableparameter
ratiosforallmodelsaredetailedintheAppendixA.
Baselines. WecompareLoftQwiththefollowingbaselinemethods:
7

• Fullfine-tuningisthemostcommonapproachforadaptingapre-trainedmodeltodownstream
tasks. The model is initialized with pre-trained weights and all parameters are updated
throughanSGD-typeoptimizationmethod.
• FullprecisionLoRA(LoRA)isalightweightmethodfortaskadaptation,whereitstoresthe
backboneusing16-bitnumbersandoptimizesthelow-rankadaptorsonly. Theadaptorsare
appliedtothesamematricesasinLoftQ.
• QLoRAissimilartoLoRAexceptthebackboneisquantizedintolow-bitregime. Thelow-rank
adaptersareinitializedusing(5)andareappliedtothesamematricesasinLoftQ.
4.1 Encoder-onlyModel: DeBERTaV3
Models and Datasets. We quantize the DeBERTaV3-base (He et al., 2021b) with LoftQ, then
finetune and evaluate the model on the General Language Understanding Evaluation (GLUE)
benchmark(Wangetal.,2019),SQuADv1.1(Rajpurkaretal.,2016),andANLI(Nieetal.,2019).
ThespecifictasksofGLUEaregiveninAppendixC.Followingpreviousworks(Zhangetal.,2023),
weexcludeWNLIintheexperiments.
ImplementationDetails. Weselectthelearningratesfrom{1×10 −5,5×10 −5,1×10 −45×10 −4}.
We quantize the entire backbone. Given that GLUE, SQuADv1.1, and ANLI are relatively easy
NLUtasks,wealsoquantizetheembeddinglayerforhighercompressionefficiency. Weapplythe
NormalFloatandtheuniformquantizationforLoftQandQLoRAatboth2-bitand4-bitlevels.
Weuserank16and32forlow-rankadapters. Moreimplementationdetails,suchasthetraining
epochsandbatchsizes,arepresentedinAppendixD.2.
Main Results. Table 1 and Table 2 summarize the results for 2-bit quantization on the GLUE,
SQuADv1.1,andANLIdatasets,byNF2andtheuniformquantization,respectively. Ourmethod
consistently outperforms QLoRA on all settings with respect to different ranks, quantization
methods,anddatasets. Whenusingtheuniformquantization(Table2),ourmethodachieves88.0%
accuracyonMNLI-m,surpassingtheQLoRAbaselineby8%. FortaskslikeSSTandSQuADv1.1,
ourmethodevenapproachesthefullfine-tuningperformanceat2-bitlevel. The4-bitquantization
experimentresultsarepresentedinAppendixD.1asbothLoftQandQLoRAachieveperformance
closetofullfine-tuning.
OurmethodisalsomorestablecomparedtoQLoRAinthelow-bitregime. Forinstance,while
QLoRAfailstoconvergeonCoLAforbothquantizationmethodsandranks,LoftQconvergesin
allcasesandachievesascoreof60.5usinguniformquantizationatrank32. LoftQstandsoutin
itsabilitytoconsistentlyattainrobustandimprovedperformancebyeffectivelypreservingthe
startingpointofpre-trainedweights.
8

Table 1: Results with 2-bit LoftQ of DeBERTaV3-base models on GLUE development set,
SQuADv1.1 development set, ANLI test set using NF2quantization. We report the median
overfourseeds. N.A.indicatesthemodeldoesnotconverge. Thebestresultsoneachdatasetare
showninbold.
Rank Method MNLI QNLI RTE SST MRPC CoLA QQP STSB SQuAD ANLI
m/mm Acc Acc Acc Acc Matt Acc P/SCorr EM/F1 Acc
- FullFT 90.5/90.6 94.0 82.0 95.3 89.5/93.3 69.2 92.4/89.8 91.6/91.1 88.5/92.8 59.8
16 LoRA 90.4/90.5 94.6 85.1 95.1 89.9/93.6 69.9 92.0/89.4 91.7/91.1 87.3/93.1 60.2
QLoRA 75.4/75.6 82.4 55.9 86.5 73.8/82.8 N.A. 86.8/82.3 83.0/82.8 61.5/71.2 N.A.
16
LoftQ 84.7/85.1 86.6 61.4 90.2 83.8/88.6 37.4 90.3/86.9 87.1/86.9 81.5/88.6 47.1
QLoRA 78.5/78.7 80.4 56.7 86.9 73.8/82.7 N.A. 87.1/82.7 83.6/83.3 64.6/73.8 N.A.
32
LoftQ 86.0/86.1 89.9 61.7 92.0 83.6/87.2 47.5 91.0/87.9 87.5/87.0 82.9/89.8 49.0
Table 2: Results with 2-bit LoftQ of DeBERTaV3-base models on GLUE development set,
SQuADv1.1 development set using Uniform quantization . We report the median over four
seeds. N.A.indicatesthemodeldoesnotconverge. Thebestresultsoneachtaskareshowninbold.
Rank Method MNLI QNLI RTE SST MRPC CoLA QQP STSB SQuAD
m/mm Acc Acc Acc Acc Matt Acc P/SCorr Em/F1
- FullFT 90.5/90.6 94.0 82.0 95.3 89.5/93.3 69.2 92.4/89.8 91.6/91.1 88.5/92.8
16 LoRA 90.4/90.5 94.6 85.1 95.1 89.9/93.6 69.9 92.0/89.4 91.7/91.1 87.3/93.1
QLoRA 76.5/76.3 83.8 56.7 86.6 75.7/84.7 N.A. 87.1/82.6 83.5/83.4 69.5/77.6
16
LoftQ 87.3/87.1 90.6 61.1 94.0 87.0/90.6 59.1 90.9/88.0 87.9/87.6 84.4/91.2
QLoRA 79.9/79.5 83.7 57.8 86.9 76.5/84.5 N.A. 88.6/84.7 84.1/84.0 71.6/80.2
32
LoftQ 88.0/88.1 92.2 63.2 94.7 87.5/91.2 60.5 91.3/88.3 89.5/89.2 85.2/91.6
4.2 Encoder-DecoderModel: BART
ModelsandDatasets. WequantizeBART-largemodel(Lewisetal.,2020)withLoftQ,thenfinetune
and evaluate the model on two commonly used summarization datasets: XSum (Narayan et al.,
2018)andCNN/DailyMail(Hermannetal.,2015).
ImplementationDetails. WeapplyLoftQtoweightmatricesinMHAandFFNofbothencoder
anddecoderlayers. WereportROUGE1/2/Lscores,whicharethemetricsforsummarizationtasks
(Lin,2004). Weconductquantizationexperimentsinboth2-bitand4-bitscenarios. Weexperiment
with both NormalFloat and the uniform quantization in both 2-bit and 4-bit scenarios. In each
precision, we choose rank equal to 8 and 16 for a fair comparison with the full precision LoRA
baseline(Zhangetal.,2023). PleaseseeAppendixEfordetailedconfigurations.
MainResults. Table3summarizesour4-bitquantizationexperimentresultsontheXSumand
CNN/DailyMail test sets. Our method consistently outperforms QLoRA at both ranks on both
9

datasets. ItevensurpassesfullprecisionLoRAatbothranksonXsum. Wewilldiscussthisunex-
pectedresultsinSection5. The2-bitquantizationresultsareshowninTable4. Ourobservation
isconsistentwiththeNLUexperiments,thatLoftQdemonstratestheconvergencetoreasonable
results,whileQLoRAdoesnotconverge. Thisindicatesourmethodisrobusterbynarrowingthe
initializationgap.
Table3: Resultswith4-bitLoftQofBART-largeonXSumandCNN/DailyMail. WereportROUGE-
1/2/L, the higher the better. Lead-3 means choosing the first 3 sentences as the summary. N.A.
indicatesthemodeldoesnotconverge. FullFTreferstothefullfine-tuningwhereallparameters
aretuned. Wereportthemedianoverfiveseeds.
| Quantization | Rank Method |                  | XSum | CNN/DailyMail     |
| ------------ | ----------- | ---------------- | ---- | ----------------- |
|              | Lead-3      | 16.30/1.60/11.95 |      | 40.42/17.62/36.67 |
-
|     | FullFT | 45.14/22.27/37.25 |     | 44.16/21.28/40.90 |
| --- | ------ | ----------------- | --- | ----------------- |
FullPrecision
|     | 8 LoRA  | 43.40/20.20/35.20 |     | 44.72/21.58/41.84 |
| --- | ------- | ----------------- | --- | ----------------- |
|     | 16 LoRA | 43.95/20.72/35.68 |     | 45.03/21.84/42.15 |
|     | QLoRA   | 42.91/19.72/34.82 |     | 43.10/20.22/40.06 |
8
|     | LoftQ | 44.08/20.72/35.89 |     | 43.81/20.95/40.84 |
| --- | ----- | ----------------- | --- | ----------------- |
NF4
|     | QLoRA | 43.29/20.05/35.15 |     | 43.42/20.62/40.44 |
| --- | ----- | ----------------- | --- | ----------------- |
16
|     | LoftQ | 44.51/21.14/36.18 |     | 43.96/21.06/40.96 |
| --- | ----- | ----------------- | --- | ----------------- |
|     | QLoRA | 41.84/18.71/33.74 |     | N.A.              |
8
|     | LoftQ | 43.86/20.51/35.69 |     | 43.73/20.91/40.77 |
| --- | ----- | ----------------- | --- | ----------------- |
Uniform
|     | QLoRA | 42.45/19.36/34.38 |     | 43.00/20.19/40.02 |
| --- | ----- | ----------------- | --- | ----------------- |
16
|     | LoftQ | 44.29/20.90/36.00 |     | 43.87/20.99/40.92 |
| --- | ----- | ----------------- | --- | ----------------- |
Table4: Resultswith2-bitLoftQofBART-largeonXSumandCNN/DailyMailusingNF2quan-
tization. N.A. indicates the model does not converge. We report ROUGE-1/2/L, the higher the
better. Wereportthemedianoverfiveseeds.
| Rank | Method | XSum | CNN/DailyMail |      |
| ---- | ------ | ---- | ------------- | ---- |
|      | QLoRA  | N.A. |               | N.A. |
8
|     | LoftQ 39.63/16.65/31.62 |      | 42.24/19.44/29.04 |      |
| --- | ----------------------- | ---- | ----------------- | ---- |
|     | QLoRA                   | N.A. |                   | N.A. |
16
|     | LoftQ 40.81/17.85/32.80 |     | 42.52/19.81/39.51 |     |
| --- | ----------------------- | --- | ----------------- | --- |
10

4.3 Decoder-onlyModel: LLAMA-2
ModelsandDatasets. WequantizeLLAMA-2-7bandLLAMA-2-13b(Touvronetal.,2023)with
LoftQ. We then fine-tune and evaluate the models on two NLG datasets: GSM8K (Cobbe et al.,
2021) and WikiText-2 (Merity et al., 2016). Please see Appendix F for more details about the
datasets.
ImplementationDetails. Similarly,weapplyLoftQtoweightmatricesinMHAandFFNofall
layers. InWikiText-2evaluation,wereportperplexity. InGSM8Kevaluation,weextractnumerical
answersinthegeneratedsolutionsandthencalculatetheaccuracyusingthosenumericalanswers.
WeconductexperimentswithbothNF2andNF4. PleaseseeAppendixFfordetailedconfigurations.
Main Results. Table 5 presents a summary of our experiments on LLAMA-2-7b and LLAMA-
2-13busing2-bit,4-bit,andmixed-precisionNormalFloatquantizationmethodsonWikiText-2
and GSM8K datasets. In WikiText-2, our method consistently outperforms QLoRA across all
quantizationprecisionsettingsonbothmodels. Whendealingwiththechallenging2-bitprecision,
whereQLoRAfailstoconverge, LoftQmanagestoachieveaperplexityof7.85. InGSM8K,our
methodachievesbetteroronparperformancecomparedtoQLoRAacrossdifferentmodelsizes
andquantizationprecisionlevels. Forexample,ourmethodachieves20.9%accuracyusing2-bit
precision,whereQLoRAdoesn’tconverge.
WefindLoftQoutperformsfullprecisionLoRAinGSM8KwithLLAMA-2-13b. Onepossible
explanationisthatthelackofregularizationcausesoverfittingonfullprecisionLoRAfine-tuning.
Therefore,weconductfullprecisionLoRAwithweightdecayonGSM8K.FromTable5,regulariza-
tionhelpsLLAMA-2-13bfullprecisionLoRAfine-tuning,butfailsinLLAMA-2-7b. Thisindicates
LLAMA-2-13b is prone to overfitting and quantization has implicit regularization to overcome
suchoverfitting.
To provide a customized trade-off between the performance and precision, we also explore
mixed-precisionquantizationwherematricesinthefirst4layersarequantizedusing4bits,andthe
restmatricesremain2bits. Wewitnessaremarkable5.9%accuracyboostontheGSM8Kdataset
usingLLAMA-2-7banda12.7%boostusingLLAMA-2-13b. Thisresultunderscoresthepotential
ofLoftQforcomplexmixed-precisionquantizationscenarios.
4.4 Analysis
EffectivenessofAlternatingOptimization. Weconductexperimentswithdifferentalternating
stepT toverifytheeffectivenessofthealternatingoptimizationandtofindthebestvalueT asa
hyperparameterfordifferentmodels. Acrossalltasksandmodels,weobservedthatalternating
optimizationyieldssubstantialimprovementsevenwithaminimalalternatingstep. Thissuggests
that it rapidly narrows the discrepancy between quantized weights and pre-trained weights,
makingourmethodeasytoapply. Forexample,ourmethodachieves88.0%accuracyonMNLI-m
datasetusingonly5alternatingstepsand21.14Rouge-2scoreusingonly1step. Interestingly,we
11

Table 5: Results of LoftQ using NormalFloat for LLAMA-2 series on WikiText-2 and GSM8K.
3/2.5/2.25-bitindicatesmixed-precisionquantization: 4-bitprecisionforthefirst16/8/4layers
and 2-bit precision for the rest of layers. We report the perplexity (the smaller the better) for
WikiText-2andaccuracyforGSM8K.Therankoflow-rankadaptersis64. N.A.indicatesthemodel
doesnotconverge. Wereportthemedianoverfiverandomseeds.
|          |      | LLAMA-2-7b  |        | LLAMA-2-13b |        |
| -------- | ---- | ----------- | ------ | ----------- | ------ |
| Method   | Bit  |             |        |             |        |
|          |      | WikiText-2↓ | GSM8K↑ | WikiText-2↓ | GSM8K↑ |
| LoRA     | 16   | 5.08        | 36.9   | 5.12        | 43.1   |
| LoRA+Reg | 16   | –           | 34.4   | –           | 45.3   |
| QLoRA    | 4    | 5.70        | 35.1   | 5.22        | 39.9   |
| LoftQ    | 4    | 5.24        | 35.0   | 5.16        | 45.0   |
| QLoRA    | 3    | 5.73        | 32.1   | 5.22        | 40.7   |
| LoftQ    | 3    | 5.63        | 32.9   | 5.13        | 44.4   |
| QLoRA    | 2.5  | N.A.        | N.A.   | 19.39       | N.A.   |
| LoftQ    | 2.5  | 5.78        | 31.1   | 5.22        | 41.1   |
| QLoRA    | 2.25 | N.A.        | N.A.   | N.A.        | N.A.   |
| LoftQ    | 2.25 | 6.13        | 26.5   | 5.45        | 38.1   |
| QLoRA    | 2    | N.A         | N.A.   | N.A.        | N.A.   |
| LoftQ    | 2    | 7.85        | 20.9   | 7.69        | 25.4   |
noticedthatincreasingthealternatingstepbeyondacertainpointtendstoresultindiminishing
returns. We suspect this phenomenon occurs because, as the gap becomes smaller, it becomes
more challenging for alternating optimization to consistently minimize the gap at each step.
This challenge emerges because of the inherent errors introduced by the quantization method.
Nevertheless,resultsfromFigure3indicateourmethodisnotsensitivetothealternatingstepT
andisabletoconsistentlyenhancedownstreamfine-tuningperformance.
5 Discussion
Start with quantization or SVD in the alternating optimization? An alternative algorithm
to the alternating optimization is that we first obtain the low-rank approximation A ,B and
t t
then obtain the quantized weight Q by switching Line 3 and Line 4 in Algorithm 1. We note
t
this is a valid alternative method as both still jointly minimize the objective in (6). Table 6
summarizes the performance of this alternative method. It is noteworthy that the alternative
methodstilloutperformsQLoRAsignificantly,eventhoughitisworsethantheprimaryversion.
Thisobservationunderscoresthepotentialforperformanceimprovementbyachievingacloser
12

| 90  |     |      |      |      | 27.0 |     |     |           | 21.5 |       |       |
| --- | --- | ---- | ---- | ---- | ---- | --- | --- | --------- | ---- | ----- | ----- |
|     |     |      | 88.0 |      |      |     |     | 25.2 25.5 |      | 21.14 |       |
| 88  |     |      |      | 87.7 | 25.0 |     |     |           |      |       | 21.09 |
|     |     | 86.6 |      |      |      |     |     |           | 21.0 |       |       |
20.83
|             |      |     |     |     | 22.5     |     | 22.5 |     |         |       |     |
| ----------- | ---- | --- | --- | --- | -------- | --- | ---- | --- | ------- | ----- | --- |
| ycaruccA 85 |      |     |     |     | ycaruccA |     |      |     | 2-EGUOR |       |     |
|             |      |     |     |     | 20.0     |     |      |     |         | 20.05 |     |
|             | 79.9 |     |     |     |          |     |      |     | 20.0    |       |     |
| 80          |      |     |     |     |          | 1.2 |      |     |         |       |     |
1
| 75  |     |                    |     |     |     | 0   |                    |      | 19.0 |                    |      |
| --- | --- | ------------------ | --- | --- | --- | --- | ------------------ | ---- | ---- | ------------------ | ---- |
|     | 0   | 1                  | 5   | 10  |     | 0   | 1                  | 5 10 |      | 0 1                | 5 10 |
|     |     | Alternating Step T |     |     |     |     | Alternating Step T |      |      | Alternating Step T |      |
|     |     | (a)MNLI            |     |     |     |     | (b)GSM8k           |      |      | (c)XSum            |      |
ComparisonofdifferentalternatingstepT
| Figure3: |     |     |     |     |     |     |     | usedinLoftQ.T |     | =0indicatesweuseQLoRA |     |
| -------- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- | --------------------- | --- |
methodthatinitializeslow-rankadaptersby(5). T =1,5,10indicatesweusedifferentT forLoftQ
describedinAlgorithm1. Left: Uniform2-bitDeBERTaV3-base. Middle: NF42-bitLLAMA-2-13b.
Right: NF4BART-large.
approximationofpre-trainedweightswithinthelow-precisionregime.
Table6: Resultsof2-bituniformlyquantizedDeBERTaV3-baseonpartofGLUE.LoftQ(SVDFirst)
indicates the alternative LoftQ that swiches Line 3 and Line 4 in Algorithm 1. We report the
medianoverfourrandomseeds. Thebestresultsoneachtaskareshowninbold.
|     |         |      |                         |                 |     |     |      | MNLI      | QNLI | SST2 |     |
| --- | ------- | ---- | ----------------------- | --------------- | --- | --- | ---- | --------- | ---- | ---- | --- |
|     |         |      |                         | Method          |     |     | Rank |           |      |      |     |
|     |         |      |                         |                 |     |     |      | m/mm      | Acc  | Acc  |     |
|     |         |      |                         | FullFT          |     |     | -    | 90.5/90.6 | 94.0 | 95.3 |     |
|     |         |      |                         | QLoRA           |     |     | 32   | 79.9/79.5 | 83.8 | 86.6 |     |
|     |         |      |                         | LoftQ(SVDFirst) |     |     | 32   | 87.8/87.7 | 84.9 | 89.7 |     |
|     |         |      | LoftQ(QuantiztionFirst) |                 |     |     | 32   | 88.0/88.1 | 92.2 | 94.7 |     |
| 6   | Related | Work |                         |                 |     |     |      |           |      |      |     |
Quantization-AwareTraining(QAT)isoftenusedtoobtainquantizedmodelsthatareadapted
indownstreamtasks(Perietal.,2020;Liuetal.,2023). Itinvolvesquantizationandfullmodel
fine-tuningatthesametime. However,QATrequiresmassivetrainingcost,suchasthegradient
andoptimizationstate. Moreover,itisdifficulttocomputethegradientofquantizedweights. Our
method,withthehelpofLoRA,sidestepstheaforementionedissues,providingalightapproach
fordownstreamtaskadaptation.
Post-TrainingQuantization(PTQ)isacategoryofpopularquantizationframeworks(Frantaretal.,
2022;Xiaoetal.,2023),whichcanalsobeusedfortaskadaptation. Itcalibratesthehigh-precision
13

modelwithasmallsubsetofthetrainingdataset. Therefore,thesubsequentquantizationisguided
by the training dataset, providing task-specific quantized models. Besides, it does not involve
anygradientbackpropagation,soitiscost-efficient.
However,itusuallyresultsinloweraccuracy
comparedtoQAT.
7 Conclusion
We propose LoftQ, a quantization framework for LLMs, which alternatively applies quantiza-
tion and low-rank approximation to the original high-precision pre-trained weights, to obtain
aninitializationforthesubsequentLoRAfine-tuning. Experimentsonnaturallanguageunder-
standing, question answering, summarization, and natural language generation show that our
framework remarkably surpasses existing methods, e.g., QLoRA, for quantizing encoder-only,
encoder-decoder,anddecoder-onlymodels. Wehavenotobservedourmethodexhibitingworse
performanceoverQLoRA.Moreover,ourquantizationframeworkdemonstrateseffectivenessand
robustnessparticularlyinlow-bitquantizationregimes,e.g.,the2-bitlevel.
References
Bai,H.,Hou,L.,Shang,L.,Jiang,X.,King,I.andLyu,M.R.(2022). Towardsefficientpost-training
quantizationofpre-trainedlanguagemodels. AdvancesinNeuralInformationProcessingSystems,
351405–1418.
Bai, H., Zhang, W., Hou, L., Shang, L., Jin, J., Jiang, X., Liu, Q., Lyu, M. and King, I. (2020).
Binarybert: Pushingthelimitofbertquantization. arXivpreprintarXiv:2012.15701.
Bar-Haim, R., Dagan, I., Dolan, B., Ferro, L., Giampiccolo, D., Magnini, B. Szpektor, I.
and
| (2006).                     | Thesecondpascalrecognisingtextualentailmentchallenge. |            |        |              |     |             |              |             |     |
| --------------------------- | ----------------------------------------------------- | ---------- | ------ | ------------ | --- | ----------- | ------------ | ----------- | --- |
| Bentivogli,                 | L., Clark,                                            | P., Dagan, | I.     | Giampiccolo, |     | D.          |              |             |     |
|                             |                                                       |            | and    |              |     | (2009). The | fifth pascal | recognizing |     |
| textualentailmentchallenge. |                                                       |            | InTAC. |              |     |             |              |             |     |
Cer,D.,Diab,M.,Agirre,E.,Lopez-Gazpio,I.andSpecia,L.(2017).SemEval-2017task1: Semantic
textualsimilaritymultilingualandcrosslingualfocusedevaluation. InProceedingsofthe11th
InternationalWorkshoponSemanticEvaluation(SemEval-2017).AssociationforComputational
Linguistics,Vancouver,Canada.
Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J.,
| Hilton, | J., Nakano, | R. et | al.     |          |           |               |                |     |       |
| ------- | ----------- | ----- | ------- | -------- | --------- | ------------- | -------------- | --- | ----- |
|         |             |       | (2021). | Training | verifiers | to solve math | word problems. |     | arXiv |
preprintarXiv:2110.14168.
| Dagan,     | I., Glickman,                        | O.  | Magnini, | B.      |     |                    |         |            |     |
| ---------- | ------------------------------------ | --- | -------- | ------- | --- | ------------------ | ------- | ---------- | --- |
|            |                                      | and |          | (2007). | The | pascal recognising | textual | entailment |     |
| challenge. | InMachineLearningChallengesWorkshop. |     |          |         |     |                    |         |            |     |
14

Dettmers, T., Lewis, M., Belkada, Y. and Zettlemoyer, L. (2022). Llm. int8 (): 8-bit matrix
multiplicationfortransformersatscale. arXivpreprintarXiv:2208.07339.
Dettmers,T.,Pagnoni,A.,Holtzman,A.andZettlemoyer,L.(2023). Qlora: Efficientfinetuning
ofquantizedllms. arXivpreprintarXiv:2305.14314.
Diao, S., Pan, R., Dong, H., Shum, K. S., Zhang, J., Xiong, W. and Zhang, T. (2023). Lmflow:
An extensible toolkit for finetuning and inference of large foundation models. arXiv preprint
arXiv:2306.12420.
Dolan, W. B. and Brockett, C. (2005). Automatically constructing a corpus of sentential para-
phrases. InProceedingsoftheThirdInternationalWorkshoponParaphrasing(IWP2005).
Frantar, E., Ashkboos, S., Hoefler, T. and Alistarh, D. (2022). Gptq: Accurate post-training
quantizationforgenerativepre-trainedtransformers. arXivpreprintarXiv:2210.17323.
Giampiccolo, D., Magnini, B., Dagan, I. and Dolan, B. (2007). The third PASCAL recognizing
textualentailmentchallenge. InProceedingsoftheACL-PASCALWorkshoponTextualEntailment
andParaphrasing.AssociationforComputationalLinguistics,Prague.
He,J.,Zhou,C.,Ma,X.,Berg-Kirkpatrick,T.andNeubig,G.(2021a). Towardsaunifiedviewof
parameter-efficienttransferlearning. arXivpreprintarXiv:2110.04366.
He,P.,Gao,J.andChen,W.(2021b). Debertav3: Improvingdebertausingelectra-stylepre-training
withgradient-disentangledembeddingsharing. arXivpreprintarXiv:2111.09543.
Hermann,K.M.,Kocisky,T.,Grefenstette,E.,Espeholt,L.,Kay,W.,Suleyman,M.andBlunsom,
P.(2015). Teachingmachinestoreadandcomprehend. Advancesinneuralinformationprocessing
systems,28.
Hu,E.J.,Shen,Y.,Wallis,P.,Allen-Zhu,Z.,Li,Y.,Wang,S.,Wang,L.andChen,W.(2021). Lora:
Low-rankadaptationoflargelanguagemodels. arXivpreprintarXiv:2106.09685.
Levesque,H.,Davis,E.andMorgenstern,L.(2012). Thewinogradschemachallenge. InThirteenth
internationalconferenceontheprinciplesofknowledgerepresentationandreasoning.
Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V. and
Zettlemoyer,L.(2019). Bart: Denoisingsequence-to-sequencepre-trainingfornaturallanguage
generation,translation,andcomprehension. arXivpreprintarXiv:1910.13461.
Lewis,M.,Liu,Y.,Goyal,N.,Ghazvininejad,M.,Mohamed,A.,Levy,O.,Stoyanov,V.andZettle-
moyer, L. (2020). BART: Denoising sequence-to-sequence pre-training for natural language
generation, translation, and comprehension. In Proceedings of the 58th Annual Meeting of the
AssociationforComputationalLinguistics.AssociationforComputationalLinguistics,Online.
15

Li, Y., Yu, Y., Zhang, Q., Liang, C., He, P., Chen, W. and Zhao, T. (2023). Losparse: Structured
compression of large language models based on low-rank and sparse approximation. arXiv
preprintarXiv:2306.11222.
Lin,C.-Y.(2004). ROUGE:Apackageforautomaticevaluationofsummaries. InTextSummarization
BranchesOut.AssociationforComputationalLinguistics,Barcelona,Spain.
Liu, Z., Oguz, B., Zhao, C., Chang, E., Stock, P., Mehdad, Y., Shi, Y., Krishnamoorthi, R. and
Chandra,V.(2023). Llm-qat: Data-freequantizationawaretrainingforlargelanguagemodels.
arXivpreprintarXiv:2305.17888.
Loshchilov, I. and Hutter, F. (2017). Decoupled weight decay regularization. arXiv preprint
arXiv:1711.05101.
Merity,S.,Xiong,C.,Bradbury,J.andSocher,R.(2016). Pointersentinelmixturemodels.
Narayan, S., Cohen, S. B. and Lapata, M. (2018). Don’t give me the details, just the summary!
topic-awareconvolutionalneuralnetworksforextremesummarization. ArXiv,abs/1808.08745.
Nie,Y.,Williams,A.,Dinan,E.,Bansal,M.,Weston,J.andKiela,D.(2019). Adversarialnli: A
newbenchmarkfornaturallanguageunderstanding. ArXiv,abs/1910.14599.
https://api.semanticscholar.org/CorpusID:207756753
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z.,
Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Te-
jani,A.,Chilamkurthy,S.,Steiner,B.,Fang,L.,Bai,J.andChintala,S.(2019). Pytorch: An
imperative style, high-performance deep learning library. In Advances in Neural Information
ProcessingSystems32.CurranAssociates,Inc.,8024–8035.
Peri, D., Patel, J. and Park, J. (2020). Deploying quantization-aware trained networks using
tensorrt. InGPUTechnologyConference.
Rajpurkar,P.,Zhang,J.,Lopyrev,K.andLiang,P.(2016). SQuAD:100,000+questionsformachine
comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural
LanguageProcessing.AssociationforComputationalLinguistics,Austin,Texas.
Shen,S.,Dong,Z.,Ye,J.,Ma,L.,Yao,Z.,Gholami,A.,Mahoney,M.W.andKeutzer,K.(2020).
Q-bert: Hessian based ultra low precision quantization of bert. In Proceedings of the AAAI
ConferenceonArtificialIntelligence,vol.34.
Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A. and Potts, C. (2013).
Recursivedeepmodelsforsemanticcompositionalityoverasentimenttreebank. InProceedings
of the 2013 Conference on Empirical Methods in Natural Language Processing. Association for
ComputationalLinguistics,Seattle,Washington,USA.
16

Touvron,H.,Martin,L.,Stone,K.,Albert,P.,Almahairi,A.,Babaei,Y.,Bashlykov,N.,Batra,S.,
Bhargava,P.,Bhosale,S.etal.(2023). Llama2: Openfoundationandfine-tunedchatmodels.
arXivpreprintarXiv:2307.09288.
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł.
and
Polosukhin,I.(2017). Attentionisallyouneed. Advancesinneuralinformationprocessingsystems,
30.
Wang,A.,Singh,A.,Michael,J.,Hill,F.,Levy,O.andBowman,S.R.(2019).
GLUE:Amulti-task
benchmarkandanalysisplatformfornaturallanguageunderstanding.InInternationalConference
onLearningRepresentations.
| Warstadt, | A., Singh, | A. Bowman, | S.  | R.      |        |                       |     |            |
| --------- | ---------- | ---------- | --- | ------- | ------ | --------------------- | --- | ---------- |
|           |            | and        |     | (2019). | Neural | network acceptability |     | judgments. |
TransactionsoftheAssociationforComputationalLinguistics,7625–641.
Williams,A.,Nangia,N.andBowman,S.(2018). Abroad-coveragechallengecorpusforsentence
understandingthroughinference. InProceedingsofthe2018ConferenceoftheNorthAmerican
ChapteroftheAssociationforComputationalLinguistics: HumanLanguageTechnologies,Volume1
(LongPapers).AssociationforComputationalLinguistics,NewOrleans,Louisiana.
Xiao,G.,Lin,J.,Seznec,M.,Wu,H.,Demouth,J.andHan,S.(2023).
|     |     |     |     |     |     | Smoothquant: |     | Accurateand |
| --- | --- | --- | --- | --- | --- | ------------ | --- | ----------- |
efficient post-training quantization for large language models. In International Conference on
MachineLearning.PMLR.
| Zafrir, | O., Boudoukh, | G., Izsak, | P. Wasserblat, |     | M.  |                 |           |            |
| ------- | ------------- | ---------- | -------------- | --- | --- | --------------- | --------- | ---------- |
|         |               |            | and            |     |     | (2019). Q8bert: | Quantized | 8bit bert. |
In2019FifthWorkshoponEnergyEfficientMachineLearningandCognitiveComputing-NeurIPS
Edition(EMC2-NIPS).IEEE.
| Zhang,         | H., Dauphin,                   | Y. N. | Ma, T.  |       |                 |          |          |         |
| -------------- | ------------------------------ | ----- | ------- | ----- | --------------- | -------- | -------- | ------- |
|                |                                | and   | (2019). | Fixup | initialization: | Residual | learning | without |
| normalization. | arXivpreprintarXiv:1901.09321. |       |         |       |                 |          |          |         |
Zhang,Q.,Chen,M.,Bukharin,A.,He,P.,Cheng,Y.,Chen,W.andZhao,T.(2023). Adaptive
budgetallocationforparameter-efficientfine-tuning.
arXivpreprintarXiv:2303.10512.
17

| A Model |     | Compression | Ratio | and | Memory | Footprint |     |     |
| ------- | --- | ----------- | ----- | --- | ------ | --------- | --- | --- |
WereportthecompressionratioafterapplyingLoftQinTable7. Itisdefinedas
backbonesize+LoRAadaptersize
|     |     | compressionration= |     |     |     |     |     | .   |
| --- | --- | ------------------ | --- | --- | --- | --- | --- | --- |
pre-trainedsize
We also measure the GPU memory cost during training. Given that GPU memory varies by
models,tasks,sequencelengths,batchsizes,etc. WereportLLAMA-2onGSM8Kasanexamplein
Table8.
Table7: Compressionratiosofbackbones.
|                |                |             | Compression |                    | Trainable |           |        | Quantization |
| -------------- | -------------- | ----------- | ----------- | ------------------ | --------- | --------- | ------ | ------------ |
|                |                | Model       |             |                    |           | Rank      | Bits   |              |
|                |                |             | ratio(%)    |                    | ratio(%)  |           |        | method       |
|                | DeBERTaV3-base |             | 15.6        |                    | 3.1       | 16        | 2      | Uniform      |
|                | DeBERTaV3-base |             | 18.8        |                    | 6.3       | 32        | 2      | Uniform      |
|                | DeBERTaV3-base |             | 17.2        |                    | 3.1       | 16        | 2      | NF2          |
|                | DeBERTaV3-base |             | 20.4        |                    | 6.3       | 32        | 2      | NF2          |
|                |                | BART-large  | 15.3        |                    | 1.2       | 8         | 4      | NF2          |
|                |                | BART-large  | 16.7        |                    | 2.5       | 16        | 4      | NF2          |
|                |                | BART-large  | 27.8        |                    | 1.2       | 8         | 4      | NF4          |
|                |                | BART-large  | 29.0        |                    | 2.5       | 16        | 4      | NF4          |
|                |                | BART-large  | 26.2        |                    | 1.2       | 8         | 4      | Uniform      |
|                |                | BART-large  | 27.5        |                    | 2.5       | 16        | 4      | Uniform      |
|                |                | LLAMA-2-7b  | 16.6        |                    | 2.4       | 64        | 2      | Nf2          |
|                |                | LLAMA-2-7b  | 29.0        |                    | 2.4       | 64        | 4      | Nf4          |
|                |                | LLAMA-2-13b | 16.0        |                    | 1.9       | 64        | 2      | Nf2          |
|                |                | LLAMA-2-13b | 28.5        |                    | 1.9       | 64        | 4      | Nf4          |
|                |                |             | Table8:     | GPUmemoryfootprint |           |           |        |              |
|                |                | Model       | Dataset     | Seqlength          |           | Batchsize | GPUMem |              |
|                |                | LLAMA-2-7b  | GSM8K       |                    | 384       | 1         |        | 15GB         |
|                |                | LLAMA-2-13b | GSM8K       |                    | 384       | 1         |        | 24GB         |
| B Quantization |                | Time        |             |                    |           |           |        |              |
WereporttheexecutiontimeofLoftQapplyingtoasingleweightmatrixinTable9. Thetimeis
testedonIntel(R)Xeon(R)CPUE5-2650v4@2.20GHz.
18

Table9: ExecutiontimeofLoftQapplyingtodifferentweightmatrices.
|     |     | Model |     | Size | StepT | Quantizationmethod |     |     |     | Time |
| --- | --- | ----- | --- | ---- | ----- | ------------------ | --- | --- | --- | ---- |
768×768
|     |     | DeBERTaV3-base |     |           |     | 5   | Uniform |     |     | 1s  |
| --- | --- | -------------- | --- | --------- | --- | --- | ------- | --- | --- | --- |
|     |     | BART-large     |     | 1024×1024 |     | 5   |         | NF4 |     | 1s  |
4096×4096
|     |     | LLAMA-2-7b |     |     |     | 5   |     | NF4 |     | 21s |
| --- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
5120×5120
|     |      | LLAMA-2-13b |            |     |     | 5   |     | NF4 |     | 43s |
| --- | ---- | ----------- | ---------- | --- | --- | --- | --- | --- | --- | --- |
| C   | GLUE | Dataset     | Statistics |     |     |     |     |     |     |     |
WepresentthedatasetstatisticsofGLUEWangetal.(2019)inthefollowingtable.
|     | Corpus | Task |     | #Train | #Dev | #Test | #Label |     | Metrics |     |
| --- | ------ | ---- | --- | ------ | ---- | ----- | ------ | --- | ------- | --- |
Single-SentenceClassification(GLUE)
|     | CoLA | Acceptability |     | 8.5k | 1k  | 1k   | 2   |     | Matthewscorr |     |
| --- | ---- | ------------- | --- | ---- | --- | ---- | --- | --- | ------------ | --- |
|     | SST  | Sentiment     |     | 67k  | 872 | 1.8k | 2   |     | Accuracy     |     |
PairwiseTextClassification(GLUE)
|     | MNLI | NLI        |     | 393k | 20k  | 20k  | 3   |     | Accuracy    |     |
| --- | ---- | ---------- | --- | ---- | ---- | ---- | --- | --- | ----------- | --- |
|     | RTE  | NLI        |     | 2.5k | 276  | 3k   | 2   |     | Accuracy    |     |
|     | QQP  | Paraphrase |     | 364k | 40k  | 391k | 2   |     | Accuracy/F1 |     |
|     | MRPC | Paraphrase |     | 3.7k | 408  | 1.7k | 2   |     | Accuracy/F1 |     |
|     | QNLI | QA/NLI     |     | 108k | 5.7k | 5.7k | 2   |     | Accuracy    |     |
TextSimilarity(GLUE)
|     | STS-B | Similarity |          | 7k                         | 1.5k | 1.4k | 1   | Pearson/Spearmancorr |     |     |
| --- | ----- | ---------- | -------- | -------------------------- | ---- | ---- | --- | -------------------- | --- | --- |
|     |       |            | Table10: | SummaryoftheGLUEbenchmark. |      |      |     |                      |     |     |
GLUEincludestwosingle-sentenceclassificationtasks: SST-2(Socheretal.,2013)andCoLA
(Warstadt et al., 2019), and three similarity and paraphrase tasks: MRPC (Dolan and Brockett,
2005),STS-B(Ceretal.,2017),andQQP.GLUEalsoincludesfournaturallanguageinferencetasks
in GLUE: MNLI (Williams et al., 2018), QNLI (Rajpurkar et al., 2016), RTE (Dagan et al., 2007;
Bar-Haimetal.,2006;Giampiccoloetal.,2007;Bentivoglietal.,2009),andWNLI(Levesqueetal.,
2012).
19

| D Natural | Language |     | Understanding |     |     |     |     |     |     |
| --------- | -------- | --- | ------------- | --- | --- | --- | --- | --- | --- |
D.1 GLUEwith4-bit
We show the 4-bits results in the Table 11. Both methods can achieve performance close to
full-finetuning.
Table11: Resultswith4-bitLoftQofDeBERTaV3-basemodelsonGLUEdevelopmentsetusing
NF4quantization. Wereportthemedianoverfourseeds. ResultswithN.A.indicatethemodel
| doesnotconverge. |     | Thebestresultsoneachdatasetareshowninbold |     |      |           |       |      |      |     |
| ---------------- | --- | ----------------------------------------- | --- | ---- | --------- | ----- | ---- | ---- | --- |
|                  |     |                                           |     |      | MNLI      | SST-2 | QNLI | ANLI |     |
|                  |     | Method                                    |     | Rank |           |       |      |      |     |
|                  |     |                                           |     |      | m/mm      | Acc   | Acc  | Acc  |     |
|                  |     | FullFT                                    |     | -    | 90.5/90.6 | 95.3  | 94.0 | 59.8 |     |
|                  |     | QLoRA                                     |     | 32   | 89.9/89.9 | 95.3  | 94.2 | 59.4 |     |
|                  |     | LoftQ                                     |     | 32   | 89.9/90.0 | 95.3  | 94.1 | 59.9 |     |
D.2 TrainingDetails
ImplementationDetails. TheimplementationofLoftQisbasedonpubliclyavailableHuggingface
(Paszkeetal.,2019)code-base**.
|                         |     |     |                                |     |     |     | −5,5×10 |     | −5,1×10 −4,5×10 −4},and |
| ----------------------- | --- | --- | ------------------------------ | --- | --- | --- | ------- | --- | ----------------------- |
| Hyper-parameterDetails. |     |     | Weselectthelearningrateof{1×10 |     |     |     |         |     |                         |
use the selected learning rate for both uniform quantization experiments and nf2 quantization
experiments. We use batch size of 32 for all GLUE tasks and ANLI. We use batch size of 16 for
| SQuADv1.1. | WeuseLoftQof5iterationsforallGLUEtasks. |     |     |     |     |     |     |     |     |
| ---------- | --------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
Table12summarizesthedetailedhyperparametersforeachtaskusedintrainingDeBERTaV3-
baseusinguniformquantization. Table13summarizesthedetailedhyperparametersforeachtask
usedintrainingDeBERTaV3-baseusingnf2quantization.
Table 12: Hyper-parameter setup of LoftQ for GLUE benchmark for training DeBERTaV3-base
usingUniformquantization.
Hyper-parameter MNLI RTE QNLI MRPC QQP SST-2 CoLA STS-B SQuADv1.1 ANLI
|     | #epochs | 5      | 20     | 10     | 60     | 10 10         | 60     | 60     | 10 12         |
| --- | ------- | ------ | ------ | ------ | ------ | ------------- | ------ | ------ | ------------- |
|     |         | 1×10−4 | 5×10−4 | 5×10−5 | 1×10−4 | 5×10−5 5×10−5 | 5×10−5 | 5×10−5 | 5×10−5 5×10−5 |
Learningrate
**https://github.com/huggingface/transformers/tree/main/examples/pytorch
20

Table 13: Hyper-parameter setup of LoftQ for GLUE benchmark for training DeBERTaV3-base
usingNF2quantization.
Hyper-parameter MNLI RTE QNLI MRPC QQP SST-2 CoLA STS-B SQuADv1.1 ANLI
| #epochs |     | 5             | 20  | 10 60         | 10 10         | 60     | 60     | 10 12         |
| ------- | --- | ------------- | --- | ------------- | ------------- | ------ | ------ | ------------- |
|         |     | 1×10−4 5×10−5 |     | 5×10−5 1×10−4 | 5×10−5 5×10−5 | 5×10−5 | 1×10−4 | 5×10−5 5×10−5 |
Learningrate
E Summarization
E.1 TrainingDetails
|     |     |     |     |     |     |     | −5,5×10 | −5,7×10 −5,2×10 −4,3× |
| --- | --- | --- | --- | --- | --- | --- | ------- | --------------------- |
WechooseAdamastheoptimizerandtrylearningratefrom{1×10
−4,4×10 −4}. WeshowtheoptimallearningratefordifferentsettingsinTable14.
10 WeuseLoftQ
of1iterationforallBART-largeexperiments. Table14andTable15summarizethelearningrate
andotherhyper-parametersforCNN/DailyMailandXSum.
Table14: Hyper-parametersetupofLoftQBART-largeonCNN/DailyMail
|     |     |     |     | NF4 | 4-bitUniform |     |     | NF2 |
| --- | --- | --- | --- | --- | ------------ | --- | --- | --- |
Hyperparameter
|     |              |                                             | rank8 | rank16 | rank8        | rank16 | rank8 | rank16 |
| --- | ------------ | ------------------------------------------- | ----- | ------ | ------------ | ------ | ----- | ------ |
|     | Learningrate |                                             | 2e-4  | 2e-4   | 2e-4         | 3e-4   | 2e-4  | 2e-4   |
|     | Epoch        |                                             |       | 15 15  | 15           | 15     |       | 15 15  |
|     | Batchsize    |                                             |       | 64 64  | 64           | 64     |       | 64 64  |
|     | Table15:     | Hyper-parametersetupofLoftQBART-largeonXSum |       |        |              |        |       |        |
|     |              |                                             |       | NF4    | 4-bitUniform |        |       | NF2    |
Hyperparameter
|     |              |     | rank8 | rank16 | rank8 | rank16 | rank8 | rank16 |
| --- | ------------ | --- | ----- | ------ | ----- | ------ | ----- | ------ |
|     | Learningrate |     | 2e-4  | 2e-4   | 2e-4  | 2e-4   | 2e-4  | 2e-4   |
|     | Epoch        |     |       | 25 25  | 25    | 25     |       | 25 25  |
|     | Batchsize    |     |       | 32 32  | 32    | 32     |       | 32 32  |
21

| F Natural |     | Language | Generation |     |     |     |     |     |     |
| --------- | --- | -------- | ---------- | --- | --- | --- | --- | --- | --- |
Wesetthebatchsizeas32forWikiText-2and16forGSM8K.Wetrain2epochsonWikiText-2and6
epochsonGSM8K.Weselectlearningratefrom{1×10 −5,5×10 −5,7×10 −5,1×10 −4,,3×10 −4,4×10 −4}.
SpecificsettingsaresummarizedinTable16andTable17.
|     | Table16:    | Hyper-parametersetupofLoftQLLAMA-2-seriesonGSM8K |                |              |     |         |                     |         |     |
| --- | ----------- | ------------------------------------------------ | -------------- | ------------ | --- | ------- | ------------------- | ------- | --- |
|     | Model       |                                                  | Hyperparameter |              |     | NF4     | NF2 Mixed-precision |         |     |
|     |             |                                                  |                |              |     | 3×10 −4 | 3×10 −4             | 3×10 −4 |     |
|     | LLAMA-2-7b  |                                                  |                | learningrate |     |         |                     |         |     |
|     | LLAMA-2-13b |                                                  |                | learningrate |     | 1×10 −4 | 1×10 −4             | 3×10 −4 |     |
Table17: Hyper-parametersetupofLoftQLLAMA-2-seriesonWikiText-2
|              | Model       |     | Hyperparameter |              |     | NF4     | NF2 Mixed-precision |         |     |
| ------------ | ----------- | --- | -------------- | ------------ | --- | ------- | ------------------- | ------- | --- |
|              |             |     |                |              |     | 3×10 −4 | 3×10 −4             | 3×10 −4 |     |
|              | LLAMA-2-7b  |     |                | learningrate |     |         |                     |         |     |
|              |             |     |                |              |     | −4      | −4                  | −4      |     |
|              | LLAMA-2-13b |     |                | learningrate |     | 1×10    | 1×10                | 3×10    |     |
| G Comparison |             | to  | Pruning        |              |     |         |                     |         |     |
Pruningisalsoawidelyusedcompressionmethod. HerewecompareLoftQwiththestate-of-the-
artpruningmethodLietal.(2023). WeshowthecomparisoninTable18. Wecanseeourmethod
significantlyoutperformsthepruningmethodsonDeBERTaV3-basemodel. Wealsoremarkthat
LoftQ can consistently reduce the memory of both training and storage. In contrast, pruning
requirestrainingtheentirefull-precisionmatrix,whichimpliesthatitcannotachieveanymemory
savingsduringthetrainingstage.
| H Extension |     | to Convolutional |     |     | Layers |     |     |     |     |
| ----------- | --- | ---------------- | --- | --- | ------ | --- | --- | --- | --- |
∈
Low-rankadapterscanalsobeappliedtoconvolutionallayers. GivenaninputfeaturemapX
| Rh×w×c |     |     |     |     |     | ×d×d,i |     |     |     |
| ------ | --- | --- | --- | --- | --- | ------ | --- | --- | --- |
1 andc 2DconvolutionalkernelsK ∈Rc 1 =1,2,...,c ,theoutputoftheconvolutional
|     | 2   |     |     |     | i   |     | 2   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
layeris
|     |     |     |     | Y =stack(X⊗K |     | ,...,X⊗K | ),  |     | (10) |
| --- | --- | --- | --- | ------------ | --- | -------- | --- | --- | ---- |
|     |     |     |     |              |     | 1        | c   |     |      |
2
∈Rh×w×c and⊗denotesthe2Dconvolutionoperation.
| whereY |     | 2   |     |     |     |     |     |     |     |
| ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
22

Table 18: Results of LoftQ using 2-bits uniform quantization compared with LoSparse with
DeBERTaV3-basemodelsonsomeofGLUEdevelopmentsets. HereRatioistheproportionoftotal
remainingweights. ResultswithN.A.indicatethemodeldoesnotconverge.
|     |     |     |        |     |       | MNLI      | SST-2 | QNLI      |
| --- | --- | --- | ------ | --- | ----- | --------- | ----- | --------- |
|     |     |     | Method |     | Ratio |           |       |           |
|     |     |     |        |     |       | m/mm      |       | Acc Acc   |
|     |     |     | FullFT |     | 100%  | 90.5/90.6 |       | 95.3 94.0 |
|     |     |     |        |     | 15%   | 83.3/82.9 |       | 87.6 90.4 |
LoSparse
|     |     |     |     |     | 20%   | 84.5/83.8 |     | 91.7 88.6 |
| --- | --- | --- | --- | --- | ----- | --------- | --- | --------- |
|     |     |     |     |     | 15.6% | 87.3/87.1 |     | 94.0 90.6 |
LoftQ
|     |     |     |     |     | 18.8% | 88.0/88.1 |     | 94.7 92.4 |
| --- | --- | --- | --- | --- | ----- | --------- | --- | --------- |
WecanreformulateEquation(10)intomatrixmultiplicationas
|     |     |     |     |     |     | =Z×H | ⊤   |     |
| --- | --- | --- | --- | --- | --- | ---- | --- | --- |
|     |     |     |     |     |     | Y    | ,   |     |
whereZ ∈Rhw×c d2 ,H ∈Rc ×c d2 , byextendingandflatteningtheinputX togetherwithconcate-
|     |     | 1   | 2   | 1   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
∈Rc
natingandflatteningkernels. Wefirstextendavectorx 1 byitsneighborvectorswithinthe
i,j
kernelwindow:
′
|     |     |     |            | x   | =Concat(x | i−d,j−d | ,...,x  | ).         |
| --- | --- | --- | ---------- | --- | --------- | ------- | ------- | ---------- |
|     |     |     |            |     | i,j       |         | i+d,j+d |            |
|     |     |     |            |     |           | 2       | 2       | 2 2        |
|     |     |     | ′ ∈ Rh×w×c | d2  |           |         | ′       | ∈ Rhw×c d2 |
Now, X becomes X 1 . We then flatten X into Z 1 . For kernels, we first
| concatenate{K |     |        | }intoH | ′ ∈Rc | ×c  | ×d×d.          |     | ′      |
| ------------- | --- | ------ | ------ | ----- | --- | -------------- | --- | ------ |
|               |     | ,...,K |        |       | 2 1 | WethenflattenH |     | intoH. |
|               |     | 1      | c      |       |     |                |     |        |
2
| NotethatH |     | canbeapproximatedbyalow-rankmatrix |     |     |     |     |     |     |
| --------- | --- | ---------------------------------- | --- | --- | --- | --- | --- | --- |
⊤
|        |     |      |            |        |     | R=UV                | ,   |                             |
| ------ | --- | ---- | ---------- | ------ | --- | ------------------- | --- | --------------------------- |
|        | ∈Rc | ×r,V | ∈Rc d2×r,r | ≪min{c |     | d2}bySVD.Therefore, |     |                             |
| whereU |     | 2    | 1          |        |     | 2 ,c 1              |     | theoriginalconvolutionlayer |
canbeapproximatedas
|     |     |     |     |     |     | Y(cid:98)=Z×(UV | ⊤ ⊤ |     |
| --- | --- | --- | --- | --- | --- | --------------- | --- | --- |
) (11)
|     |     |     |     |     |     | =(Z×V)×U | ⊤   |     |
| --- | --- | --- | --- | --- | --- | -------- | --- | --- |
(12)
|     |     |     |     |     |     | =M×U | ⊤   |     |
| --- | --- | --- | --- | --- | --- | ---- | --- | --- |
. (13)
Note that Z×V can be restored into a convolution operation where we have r kernels D ∈
i
Rc ×d×d,i =1,2,,...,r andM×U ⊤ canalsoberestoredintoaconvolutionoperationwherewehave
1
∈Rr×1×1,i
| c kernelsU |     |     | =1,2,,...,c |     | .   |     |     |     |
| ---------- | --- | --- | ----------- | --- | --- | --- | --- | --- |
| 2          |     | i   |             |     | 2   |     |     |     |
23

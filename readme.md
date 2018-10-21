
# Programa Nanodegree Data Science for Business

## by Rafael Tomaz, Identify Fraud from Enron Email

#### Visão Geral do Projeto

Em 2000, Enron era uma das maiores empresas dos Estados Unidos. Já em 2002, ela colapsou e quebrou devido a uma fraude que envolveu grande parte da corporação. Resultando em uma investigação federal, muitos dados que são normalmente confidenciais, se tornaram públicos, incluindo dezenas de milhares de e-mails e detalhes financeiros para os executivos dos mais altos níveis da empresa. Neste projeto, você irá bancar o detetive, e colocar suas habilidades na construção de um modelo preditivo que visará determinar se um funcionário é ou não um funcionário de interesse (POI). Um funcionário de interesse é um funcionário que participou do escândalo da empresa Enron. Para te auxiliar neste trabalho de detetive, nós combinamos os dados financeiros e sobre e-mails dos funcionários investigados neste caso de fraude, o que significa que eles foram indiciados, fecharam acordos com o governo, ou testemunharam em troca de imunidade no processo.
 

### Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  
    [relevant rubric items: “data exploration”, “outlier investigation”]

 O objetivo deste projeto era utilizar os dados financeiros e e-mail da Enron para construir um modelo preditivo para determinar se um indivíduo poderia ser considerado uma "pessoa de interesse" (POI) ou não, pessoa causadora de fraude na Enron., como o conjunto de dados continha dados rotulados as pessoas culpáveis ​​já estavam listadas como POIs.

O Conjunto de dados possui 146 registros, com 21 características disponíveis, sendo 14 financeiras (pagamentos e investimentos), 6 de e-mail e 1 rôtulos (se é um POI), dos 146 registros possuem 18 (12% dos dados) POIs e 128(88%) non-POIs, durante a análise identifiquei que 44.29% dos dados estavam faltantes com 'NaN'.

__Analise dos Dados__:

- Ajustes de dados:
    - BELFER ROBERT estavam com dados deslocados a direita, atualizei os dados conforme o PDF
    - BHATNAGAR SANJAY, estavam com os dados deslocados a esquerda, realizei o mesmo procedimento.
    
    
- Removidos:
    - 'THE TRAVEL AGENCY IN THE PARK' : agencia de viagem, empresa co-irmã da Enron e o valor não é tão alto e não tem muita correlação com outros dados.
    - 'TOTAL' : o dado é um outlier avaliado pelo grafico
    - 'LOCKHART EUGENE E' : todos os valores estão como NaN
    
    
- POIs que não possuem email de / Para:
    - FASTOW ANDREW S
    - HIRKO JOSEPH
    - KOPPER MICHAEL J
    - YEAGER F SCOTT

Após ajustes e remover os outliers, o dataset ficou com 143 registros,  com 18 (__13%__) POIs e 125 (__87%__)non-POIs

### What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? 
    As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

Para otimizar e selecionar as características mais influentes do dataset utilizei a função de seleção de recursos automatizado __SelectKBest__ do scikit-learn, realizei testes utilizando o SelectKBest juntamente com o [Pipeline](http://scikit-learn.org/stable/modules/pipeline.html) e o GridSearchCV para selecionar o algorítico com a resposta com as melhores características.

A abordagem do __SelectKBest__ é um algoritmo de seleção de recurso uni-variado automatizado, e sendo assim as melhores variáveis enumeradas não levaram em conta as variáveis relacionadas com e-mail, e utilizei todas as features do dominio, menos a de email, devido a isso criei os recursos a seguir.

Na fase de feature engeneering, criei 5 novos recursos para auxiliar na análise e enriquecer os dados:

'sum_total_poi_messages',
                'bonus_over_total_salary',
                'salary_over_total_salary',
                'shared_receipt_over_from_messages',
                'shared_receipt_over_to_messages']

- bonus_over_total_salary: 
    - com o quociente da divisão do bonus pelo total do salário para saber quanto o bonus representa pelo total pago ao funcionário
    
- salary_over_total_salary:
    - com o quociente da divisão do salário pelo total do salário para saber quanto o bonus representa pelo total pago ao funcionário

- sum_total_poi_messages :
    - Com a somatória de todos as correspondências com POIs, com as variáveis "from_poi_to_this_person" e "agg_total_poi_correspondence"
    
- shared_receipt_over_from_messages:
    - com o quociente dos emails compartilhados entre o usuário e um POI e as mensagens enviadas deste usuário

- shared_receipt_over_to_messages:
    - com o quociente dos emails compartilhados entre o usuário e um POI e as mensagens recebidas para este usuário

Para selecionar as características com maior relevância e verificar se utilizar em 2 análises Features sem Feature Engeneering e com Feature Engeneering.

> __Sem Feature Engeneering__

Característica | Pontuação          
-|-
total_stock_value | 22,510
exercised_stock_options | 22,349
bonus | 20,792
salary | 18,289
deferred_income | 11,425
long_term_incentive | 9,922
total_payments | 9,284
restricted_stock | 8,825
shared_receipt_with_poi | 8,589
loan_advances | 7,184
expenses | 5,419
from_poi_to_this_person | 5,243
other | 4,202
from_this_person_to_poi | 2,383
director_fees | 2,131 
to_messages | 1,646
restricted_stock_deferred | 0,768
deferral_payments | 0,229
from_messages | 0,169


> __Com  Feature Engeneering__

Característica | Pontuação          
-|-
total_stock_value | 22,510
exercised_stock_options | 22,349
bonus | 20,792
bonus_over_total_salary | 20,715
salary | 18,289
deferred_income | 11,425
long_term_incentive | 9,922
total_payments | 9,284
shared_receipt_over_to_messages | 9,101
restricted_stock | 8,825
shared_receipt_with_poi | 8,589
loan_advances | 7,184
shared_receipt_over_from_messages | 5,689
expenses | 5,419
from_poi_to_this_person | 5,243
sum_total_poi_messages | 4,863
other | 4,202
salary_over_total_salary | 2,687
from_this_person_to_poi | 2,382
director_fees | 2,131
to_messages | 1,646
restricted_stock_deferred | 0,768
deferral_payments | 0,229
from_messages | 0,169

Devido os tipos de algoritmos escolhidos foram Naive Bayes e Decision Tree são algoritmos que não precisam de escalonamento e normalização não foi realizado processo de normaliazação neste caso, não utilizei o algoritomos para processamento de escalonamento e normalização como MinMaxScaler ou StandardScaler por exemplo.

### What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  
    [relevant rubric item: “pick an algorithm”]

Eu avaliei os algoritmos _Naive Bayes Gaussian, Decision Tree Classifier e Decision Tree Regressor_ porém acabei escolhendo o modelo de classificação __Naive Bayes Gaussian__  pois possui os melhores resultados de validação como (acuracia "**_0,847_**"", precisão "**_0,405_**"" e recall "**_0,304_**"") e que acabou sendo a opção final.

### What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune?
    _(Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]_


Para ajustar os parâmetros de um algoritmo, signica conseguir uma combinação que possa trazer um melhor resultado para o modelo, eu fiz isso fornecendo uma matriz com todos os parametros possíveis que eu queria parametrizar para escolha para o GridSearchCV, se não fizer isso bem corre risco de ter overfitting, ocorre com frequencia dos parametros fornecidos para as previsões sobre os dados de treinamento tenham ótimo desempenho, com desempenhos muito altos devem ser observados, e os dados de teste não sejam tão bons assim, dessa forma estou utilizando a precisão e recall para medir o desempenho, as respectivas formulas são:

    Precisão = Verdadeiros Positivos / (Verdadeiros Positivos + Falsos Positivos)

    Recall = Verdadeiros Positivos / (Verdadeiros Positivos + Falsos Negativos)


A precisão é a relação da frequência que identificamos corretamente os POIs que identificamos como POI, se a precisão é baixa quer dizer que temos falsos positivos demais, portanto as pessoas que estão marcadas como POIs quando na verdade não são.

O recall ou abrangência é uma relação entre a frequência para identificarmos corretamente os POIs e o número total de POIs no conjunto de dados, incluindo os que perdemos, que não conseguimos identificar (foram marcados como não POIs). Se o recall for alto, podemos ter certeza de que, se uma pessoa for um POI, vamos marcá-lo como tal. Se o recall for baixo, significa que estamos perdendo muitos POIs, ou seja, marcando-os como não-POI.

Para avaliar a performance do melhor algoritmo, eu avaliei os dados de acurácia, precisão e recall em todos os modelos e algorítmos utilizando o função "__test_classifier__" disponível no _tester.py_. Para entender a evolução dos cenários avaliei em 3 pontos o código:

> Com o Tuning ajustando os parâmetros das variâveis dos modelos com o GridSearchCV
    Cada Algoritmo foi ajustados os parâmetros conforme suas definições em suas documentações

- Naive Bayes Gaussian
    Não Possui Tunning para configuração apenas se posso realizar alteração alteração de parametros
    
- Decision Tree Classifier
    - criterion : gini e entropy
    - splitter : best
    - max_depth: 2,4,6

> __Sem Feature Engeneering__

Algoritmo | Acuracia | Precisão| Recall | F1 | F2 | Total Pred. | True Posit. | False Posit. | False Negat. | True Negat.
- |- | - | - | - | - | - | - | - | - | -
Naive Bayes Gaussian | 0.84767 | 0,356 | 0,330, | 0,342 | 0,335 | 15000 | 661 | 1194 | 1339 | 11806
Decision Tree Classifier | 0,800 | 0,2443 | 0,2380 | 0,241 | 0,239 | 15000 | 476 | 1472 | 1524 | 11528
Decision Tree Classifier ( Tunning)| 0,821 | 0,248 | 0,166 | 0,199 | 0,178 | 15000 | 332 | 1008 | 1668 | 11992


> __Sem Feature Engeneering__

Algoritmo | Acuracia | Precisão| Recall | F1 | F2 | Total Pred. | True Posit. | False Posit. | False Negat. | True Negat.
- |- | - | - | - | - | - | - | - | - | -
Naive Bayes Gaussian | 0,831 | 0,340 | 0,280 | 0,307 | 0,290 | 15000 | 561 | 1088 | 1439 | 11912
Decision Tree Classifier | 0,794 | 0,234 | 0,239 | 0,237 | 0,238 | 15000 | 478 | 1558 | 1522 | 11442
Decision Tree Classifier ( Tunning)| 0,820 | 0,258 | 0,185 | 0,216 | 0,196 | 15000 | 370 | 1060 | 1630 | 11940


A melhor estimativa de algoritmo e foi utilizado como algoritmo final e foi utilizado pelo __GridSearchCV__ foi:

__*Conforme avaliação de acuracia, precisão e recall o melhor algoritmo foi o Naive Bayes Gaussian, e a estimativa solicitada como precisão e recall seja ao menos 0.3, com base na analise dos dados apresentados acima e dos testes realizados, Naive Bayes Gaussian, apresentou o melhor resultado com features padrões sem utilizar as features que foram realizado feature engeneering e utilizando SelectKBest com K=10.*__

### Logs Execução algoritmos
######################################
####                               ###
####   Print K best features_list  ###
####                               ###
######################################


[('total_stock_value', 22.510549090242055), ('exercised_stock_options', 22.348975407306217), ('bonus', 20.792252047181535), ('salary', 18.289684043404513), ('deferred_income', 11.424891485418364), ('long_term_incentive', 9.922186013189823), ('total_payments', 9.283873618427371), ('restricted_stock', 8.825442219916463), ('shared_receipt_with_poi', 8.589420731682381), ('loan_advances', 7.184055658288725), ('expenses', 5.418900189407036), ('from_poi_to_this_person', 5.243449713374958), ('other', 4.202436300271228), ('from_this_person_to_poi', 2.382612108227674), ('director_fees', 2.1314839924612046), ('to_messages', 1.6463411294420076), ('restricted_stock_deferred', 0.7681463447871311), ('deferral_payments', 0.22885961902145746), ('from_messages', 0.16970094762175533)]


######################################
####                               ###
####  Print K best my_feature_list ###
####                               ###
######################################


[('total_stock_value', 22.510549090242055), ('exercised_stock_options', 22.348975407306217), ('bonus', 20.792252047181535), ('bonus_over_total_salary', 20.715596247559954), ('salary', 18.289684043404513), ('deferred_income', 11.424891485418364), ('long_term_incentive', 9.922186013189823), ('total_payments', 9.283873618427371), ('shared_receipt_over_to_messages', 9.101268739193504), ('restricted_stock', 8.825442219916463), ('shared_receipt_with_poi', 8.589420731682381), ('loan_advances', 7.184055658288725), ('shared_receipt_over_from_messages', 5.689969434164784), ('expenses', 5.418900189407036), ('from_poi_to_this_person', 5.243449713374958), ('sum_total_poi_messages', 4.863681839412244), ('other', 4.202436300271228), ('salary_over_total_salary', 2.687417590844055), ('from_this_person_to_poi', 2.382612108227674), ('director_fees', 2.1314839924612046), ('to_messages', 1.6463411294420076), ('restricted_stock_deferred', 0.7681463447871311), ('deferral_payments', 0.22885961902145746), ('from_messages', 0.16970094762175533)]


####################################################



###################################
####                            ###
####   Classificadores          ###
####                            ###
#### Sem Featuring Engeneerinng ###
####                            ###
###################################



###################################
####                            ###
####   GaussianNB               ###
####                            ###
###################################


/Users/rafaelstomaz/anaconda3/envs/py27/lib/python2.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples.
  'precision', 'predicted', average, warn_for)
/Users/rafaelstomaz/anaconda3/envs/py27/lib/python2.7/site-packages/sklearn/feature_selection/univariate_selection.py:113: UserWarning: Features [5] are constant.
  UserWarning)
/Users/rafaelstomaz/anaconda3/envs/py27/lib/python2.7/site-packages/sklearn/feature_selection/univariate_selection.py:114: RuntimeWarning: invalid value encountered in divide
  f = msb / msw
GridSearchCV(cv=5, error_score='raise',
       estimator=Pipeline(memory=None,
     steps=[('feature_selection', SelectKBest(k=10, score_func=<function f_classif at 0x1a22f15aa0>)), ('classify', GaussianNB(priors=None))]),
       fit_params={}, iid=True, n_jobs=1,
       param_grid=[{'feature_selection__k': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]}],
       pre_dispatch='2*n_jobs', refit=True,
       scoring=make_scorer(precision_and_recall_score), verbose=0)
  Accuracy: 0.84767 Precision: 0.40519  Recall: 0.30450 F1: 0.34770 F2: 0.32043
  Total predictions: 15000  True positives:  609  False positives:  894 False negatives: 1391 True negatives: 12106

Estimator

Pipeline(memory=None,
     steps=[('feature_selection', SelectKBest(k=10, score_func=<function f_classif at 0x1a22f15aa0>)), ('classify', GaussianNB(priors=None))])

Features importances
{'feature_selection__k': 10}
0.254912405303

###################################
####                            ###
####  DecisionTreeClassifier    ###
####                            ###
###################################

GridSearchCV(cv=5, error_score='raise',
       estimator=Pipeline(memory=None,
     steps=[('feature_selection', SelectKBest(k=10, score_func=<function f_classif at 0x1a1add7aa0>)), ('classify', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='best'))]),
       fit_params={}, iid=True, n_jobs=1,
       param_grid=[{'feature_selection__k': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]}],
       pre_dispatch='2*n_jobs', refit=True,
       scoring=make_scorer(precision_and_recall_score), verbose=0)
  Accuracy: 0.80027 Precision: 0.24435  Recall: 0.23800 F1: 0.24113 F2: 0.23924
  Total predictions: 15000  True positives:  476  False positives: 1472 False negatives: 1524 True negatives: 11528

Estimator

Pipeline(memory=None,
     steps=[('feature_selection', SelectKBest(k=11, score_func=<function f_classif at 0x1a1add7aa0>)), ('classify', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='best'))])

Features importances
{'feature_selection__k': 11}
0.350911458333


###################################
####                            ###
####  DecisionTreeClassifier    ###
####                            ###
####        Tunning             ###
####                            ###
###################################



GridSearchCV(cv=5, error_score='raise',
       estimator=Pipeline(memory=None,
     steps=[('feature_selection', SelectKBest(k=10, score_func=<function f_classif at 0x1a1add7aa0>)), ('classify', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='best'))]),
       fit_params={}, iid=True, n_jobs=1,
       param_grid=[{'feature_selection__k': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 'classify__splitter': ['best'], 'classify__criterion': ['gini', 'entropy'], 'classify__max_depth': [2, 4, 6]}],
       pre_dispatch='2*n_jobs', refit=True,
       scoring=make_scorer(precision_and_recall_score), verbose=0)
  Accuracy: 0.82160 Precision: 0.24776  Recall: 0.16600 F1: 0.19880 F2: 0.17773
  Total predictions: 15000  True positives:  332  False positives: 1008 False negatives: 1668 True negatives: 11992

Estimator

Pipeline(memory=None,
     steps=[('feature_selection', SelectKBest(k=11, score_func=<function f_classif at 0x1a1add7aa0>)), ('classify', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=6,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='best'))])

Features importances
{'feature_selection__k': 11, 'classify__splitter': 'best', 'classify__criterion': 'gini', 'classify__max_depth': 6}
0.285807291667



####################################################



###################################
####                            ###
####   Classificadores          ###
####                            ###
#### Com Featuring Engeneerinng ###
####                            ###
###################################



###################################
####                            ###
####   GaussianNB               ###
####                            ###
###################################


GridSearchCV(cv=5, error_score='raise',
       estimator=Pipeline(memory=None,
     steps=[('feature_selection', SelectKBest(k=10, score_func=<function f_classif at 0x1a1add7aa0>)), ('classify', GaussianNB(priors=None))]),
       fit_params={}, iid=True, n_jobs=1,
       param_grid=[{'feature_selection__k': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}],
       pre_dispatch='2*n_jobs', refit=True,
       scoring=make_scorer(precision_and_recall_score), verbose=0)
  Accuracy: 0.83153 Precision: 0.34021  Recall: 0.28050 F1: 0.30748 F2: 0.29070
  Total predictions: 15000  True positives:  561  False positives: 1088 False negatives: 1439 True negatives: 11912

Estimator

Pipeline(memory=None,
     steps=[('feature_selection', SelectKBest(k=16, score_func=<function f_classif at 0x1a1add7aa0>)), ('classify', GaussianNB(priors=None))])

Features importances
{'feature_selection__k': 16}
0.405598958333
###################################
####                            ###
####  DecisionTreeClassifier    ###
####                            ###
###################################

GridSearchCV(cv=5, error_score='raise',
       estimator=Pipeline(memory=None,
     steps=[('feature_selection', SelectKBest(k=10, score_func=<function f_classif at 0x1a1add7aa0>)), ('classify', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='best'))]),
       fit_params={}, iid=True, n_jobs=1,
       param_grid=[{'feature_selection__k': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}],
       pre_dispatch='2*n_jobs', refit=True,
       scoring=make_scorer(precision_and_recall_score), verbose=0)
  Accuracy: 0.79467 Precision: 0.23477  Recall: 0.23900 F1: 0.23687 F2: 0.23814
  Total predictions: 15000  True positives:  478  False positives: 1558 False negatives: 1522 True negatives: 11442

Estimator

Pipeline(memory=None,
     steps=[('feature_selection', SelectKBest(k=14, score_func=<function f_classif at 0x1a1add7aa0>)), ('classify', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='best'))])

Features importances
{'feature_selection__k': 14}
0.276041666667


###################################
####                            ###
####  DecisionTreeClassifier    ###
####                            ###
####        Tunning             ###
####                            ###
###################################



GridSearchCV(cv=5, error_score='raise',
       estimator=Pipeline(memory=None,
     steps=[('feature_selection', SelectKBest(k=10, score_func=<function f_classif at 0x1a1add7aa0>)), ('classify', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='best'))]),
       fit_params={}, iid=True, n_jobs=1,
       param_grid=[{'feature_selection__k': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], 'classify__splitter': ['best'], 'classify__criterion': ['gini', 'entropy'], 'classify__max_depth': [2, 4, 6]}],
       pre_dispatch='2*n_jobs', refit=True,
       scoring=make_scorer(precision_and_recall_score), verbose=0)
  Accuracy: 0.82067 Precision: 0.25874  Recall: 0.18500 F1: 0.21574 F2: 0.19618
  Total predictions: 15000  True positives:  370  False positives: 1060 False negatives: 1630 True negatives: 11940

Estimator

Pipeline(memory=None,
     steps=[('feature_selection', SelectKBest(k=14, score_func=<function f_classif at 0x1a1add7aa0>)), ('classify', DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=6,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='best'))])

Features importances
{'feature_selection__k': 14, 'classify__splitter': 'best', 'classify__criterion': 'entropy', 'classify__max_depth': 6}
0.436197916667



####################################################

### What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  
    [relevant rubric items: “discuss validation”, “validation strategy”]


Validação é o processo de verificar seu modelo em um dado de treinamennto e depois testa a previsão do seu modelo em relação aos em um dado diferente, um erro classico que ocorre é o __"overfitting"__ que ocorre quando se treina e testa um algoritmo com todos os mesmo dados disponíveis em vez de dividi-los, tendo assim um desempenho muito bom. O overfitting acaba fazendo com que o modelo memorize a classificação e não _aprenda_ a generalizar e aplique essa informação a novos conjuntos de dados.

No pacote do scikit-learn há um método auxiliar chamado cross_validation.train_test_split que faz exatamente isso.

Para validar minha analise eu utilizei a função test_classifier dentro do tester.py pois utiliza a função StratifiedShuffleSplit que utiliza a estratégia k-fold, que foi utilizado para dividir os dados pelos parametros de treinamento e teste e selecionar os parametros que melhor desempenharam segregando em 1000 divisões, essa estratificação é nnecessária pois temos muito pouco POIs em comparação a não POIs, e não queremos treinar o conjunto de dados que innclua  apenas membros que estão classificados como não POIs.

### Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. 
    [relevant rubric item: “usage of evaluation metrics”]

A precisão do meu algoritmo foi de 0,40, isso quer dizer que quando ele informou que alguém era um POI, era verdade 40% do tempo.

O recall era 0,30. Isso indica que onde todos os POIs que o algoritmo viu ele identificou corretamente 30% deles como POI (e perdeu 70%).

### Observação sobre regressor em modelo de classificação

Na submissão anterior realizei uma análise com "Decision Tree Regressor", inseri esta análise para comprovar que realizando uma verificação de dados de um modelo que deveria ser analisado por um classificador por um modelo de regressão até seria possível porém não seria a melhor abordagem pois teria perda na análise e até em boas partes das predições sendo assim o modelo não seria o mais efetivo e nem mesmo a melhor escolha, sendo assim, cada modelo deve ter a sua escolha correta, dessa forma foi um bom caso de estudo para entendimento de onde deve ser utilizado o algoritmo de classificação corretamente.

### Referências:

Durante o trabalho eu consultei os seguintes materiais:

- Documentação Scikit learn
- Documentação do Python e Pandas

- Exemplos e arquivos do curso

- Apostilas do curso de MBA em Machine Learning e Inteligencia artifical da Universidade Fiap:
    - Estatistica para Inteligência artifical
    - Python para Inteligência artifical
    - Modelos Machine Learning
    
- Stack Overflow para mensagens de erro durante o projeto.

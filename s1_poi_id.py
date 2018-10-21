#!/usr/bin/python

import sys
import pickle

from sklearn.cross_validation import StratifiedShuffleSplit

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier,main

# parameter optimization (not currently used)
from sklearn.grid_search import GridSearchCV

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# todos em dolares americanos (USD) de valores de pagamentos recebidos
all_features_finacial_payments = ['salary',
'bonus',
'long_term_incentive',
'deferred_income',
'deferral_payments',
'loan_advances',
'other',
'expenses',
'director_fees',
'total_payments']

# variaveis em dolares americanos (USD) de valores de investimentos
all_features_finacial_stocks = ['exercised_stock_options',
 'restricted_stock',
 'restricted_stock_deferred',
 'total_stock_value']

# as unidades aqui sao geralmente em numero de emails; a excecao notavel aqui e o atributo email_address, que e uma string
all_features_email = ['to_messages',
'from_poi_to_this_person',
'from_messages',
'from_this_person_to_poi',
'shared_receipt_with_poi']

# atributo string de email
email_address = 'email_address'

# atributo objetivo logico (booleano), representado como um inteiro
target_label = 'poi'

# lista de chaves para serem removidas durante a analise
pessoas_para_serem_removidas = set()

# Candidatos a POIS
pois_candidates = set()


## Main Functions

def normalize_value_to_numpy_nan(value):
    if value is None or value == 'NaN':
        value = np.nan
    return value

def print_index_df(df):
    for x in df.index:
        print x

def get_array_string_df(df):
    str_indext_list = []
    for x in df.index:
        str_indext_list.append(str(x))
    return str_indext_list

def remove_outliers_df(df, set_key):
    return df.drop(axis=0, labels=set_key)

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print "####################################################"
print "###    Entendimento dos Dados e das Perguntas    ###"
print "####################################################"
features_list = [target_label] + all_features_finacial_payments + all_features_finacial_stocks + all_features_email # You will need to use more features

print "Convertendo o dicionario dos dados para pandas para iniciar as analises de dados e ajustar os dados"
df = pd.DataFrame.from_dict(data_dict,orient='index')
df = df.applymap(lambda x: normalize_value_to_numpy_nan(x))

print "Analise dos dados"

is_poi = df[target_label] == True
pois_count = df['poi'].value_counts()
total_items = df.shape[0]

print "Existem %d data points (pessoas) no dataset" % total_items

print "No dataset possuem %d (%2.f Porcento) POIs e %d (%2.f Porcento) non-POIs" % (pois_count[1],(pois_count[1] * 1. / total_items) * 100., pois_count[0], (pois_count[0] * 1. / total_items) * 100.)
print "Possui %d caracteristicas disponiveis, 14 atributos financeiros, 6 atributos de email e 1 atributo de rotulo, alem de uma chave que e o nome do POI" % df.shape[1]

print "Existem caracteristicas com muitos valores faltando, Nulos ('NaN')"

missing_values_by_columns = df.shape[0]-df.count()
missing_values_by_columns.sort_values(axis=0, ascending=False, inplace=True)

total_celulas_preenchidas = sum(df.count())
total_celulas_faltantes = sum(missing_values_by_columns)
total_celulas = total_celulas_preenchidas + total_celulas_faltantes
print "Existem %d ceculas no total, porem %d celulas estao preenchidas e %d celulas estao vazias, no total de %.2f (Porcento)" % (total_celulas, total_celulas_preenchidas, total_celulas_faltantes, (total_celulas_faltantes * 1. / total_celulas) * 100)


print "Count para variaveis de POIs"
print pois_count

print df[is_poi].count()

print "Os nomes dos POIS sao:"
print print_index_df(df[is_poi])


print "POIs que nao possuem email de / Para:"
print
pois_var = df[df['to_messages'].isnull() & df['from_messages'].isnull() & is_poi]
print_index_df(pois_var)

user_without_financial_information = df[
    df['salary'].isnull() &
    df['deferral_payments'].isnull() &
    df['total_payments'].isnull() &
    df['loan_advances'].isnull() &
    df['bonus'].isnull() &
    df['restricted_stock_deferred'].isnull() &
    df['deferred_income'].isnull() &
    df['total_stock_value'].isnull() &
    df['expenses'].isnull() &
    df['exercised_stock_options'].isnull() &
    df['other'].isnull() &
    df['long_term_incentive'].isnull() &
    df['restricted_stock'].isnull() &
    df['director_fees'].isnull()
]

print"Usuarios que nao possuem dados financeiros, portanto podem ser removidos"
print_index_df(user_without_financial_information)

print "Com base nos dados acima por se tratar de ser apenas de uma empresa de viagem prestadora de servico e nao ter valor expressivo e nao ser um POI podera ser removido do dataset"
print "Analisando o PDF incluido no trabalho no Footnotes indice (e) 'Payments were made by Enron employees on account of business-related travel to The Travel Agency in the Park (later Alliance Worldwide), which was co- owned by the sister of Enron's former Chairman. Payments made by the Debtor to reimburse employees for these expenses have not been included.'"
print "Portanto pode-se ser removido da analise"
travel_agency = 'THE TRAVEL AGENCY IN THE PARK'
df.loc[travel_agency]

pessoas_para_serem_removidas.update(get_array_string_df(user_without_financial_information))
pessoas_para_serem_removidas.add(travel_agency)
print pessoas_para_serem_removidas

print "Pessoas que nao possuem total de pagamentos e total de stock value"
user_without_financial_total_payments_and_total_stock_values_information = df[
    df['total_payments'].isnull() &
    df['total_stock_value'].isnull()
]
print_index_df(user_without_financial_total_payments_and_total_stock_values_information)

user_without_financial_total_payments_and_total_stock_values_information[all_features_finacial_payments]
user_without_financial_total_payments_and_total_stock_values_information[all_features_finacial_payments]
user_without_financial_total_payments_and_total_stock_values_information[all_features_email + [target_label]]

df.fillna(value=0, inplace=True)

print "Verificar se possui erro de cadastro em dados financeiros de pagamento e stock"
print
print "Pagamento:"
print
erros = (df[df[all_features_finacial_payments[:-1]].sum(axis='columns') != df['total_payments']])
erros[all_features_finacial_payments]

print "Stock:"
print
erros[all_features_finacial_stocks]

print "Para analisar os casos de  BELFER ROBERT e BHATNAGAR SANJAY, irei comparar os dados com os dados oficiais disponiveis no pdf oficial"
print
print "BELFER ROBERT: identifiquei que os dados financeiros foram deslocados a direita, sendo assim o valor deferred_income  esta no lugar incorreto  deferral_payments e assim por diante, irei deslocar os dados para a esquerda e colocar o ultimo valor de total_stock_value conforme do pdf para ficar correto"
print
print "BHATNAGAR SANJAY: identifiquei que os dados financeiros foram deslocados a esquerda, diferente dos dados de Robert, dessa forma irei corrigir deslocando os dados a direita, irei complementar o valor salary, conforme o valor do PDF"

print "Dados ROBERT BELFER antes da mudanca"
robert_belfer = df[:].loc['BELFER ROBERT']
print robert_belfer

print "Dados ROBERT BELFER  apos da mudanca"
robert_belfer['salary'] = 0.0
robert_belfer['bonus'] = 0.0
robert_belfer['long_term_incentive'] = 0.0
robert_belfer['deferred_income'] = -102500.0
robert_belfer['deferral_payments'] = 0.0
robert_belfer['loan_advances'] = 0.0
robert_belfer['other'] = 0.0
robert_belfer['expenses'] = 3285.0
robert_belfer['director_fees'] = 102500.0
robert_belfer['total_payments'] = 3285.0
robert_belfer['exercised_stock_options'] = 0.0
robert_belfer['restricted_stock'] = 44093.0
robert_belfer['restricted_stock_deferred'] = -44093.0
robert_belfer['total_stock_value'] = 0.0
df[:].loc['BELFER ROBERT'] = robert_belfer
print robert_belfer

print "Dados SANJAY BHATNAGAR antes da mudanca"
bhatnagar_sanjay = df[:].loc['BHATNAGAR SANJAY']
print bhatnagar_sanjay

print "Dados SANJAY BHATNAGAR apos da mudanca"
bhatnagar_sanjay['salary'] = 0.0
bhatnagar_sanjay['bonus'] = 0.0
bhatnagar_sanjay['long_term_incentive'] = 0.0
bhatnagar_sanjay['deferred_income'] = 0.0
bhatnagar_sanjay['deferral_payments'] = 0.0
bhatnagar_sanjay['loan_advances'] = 0.0
bhatnagar_sanjay['other'] = 0.0
bhatnagar_sanjay['expenses'] = 137864.0
bhatnagar_sanjay['director_fees'] = 0.0
bhatnagar_sanjay['total_payments'] = 137864.0
bhatnagar_sanjay['exercised_stock_options'] = 15456290.0
bhatnagar_sanjay['restricted_stock'] = 2604490.0
bhatnagar_sanjay['restricted_stock_deferred'] = -2604490.0
bhatnagar_sanjay['total_stock_value'] = 15456290.0
df[:].loc['BHATNAGAR SANJAY'] = bhatnagar_sanjay
print bhatnagar_sanjay

print "Verificar se ainda existe erros em dados financeiros:"
len(df[df[all_features_finacial_payments[:-1]].sum(axis='columns') != df['total_payments']])

print "Analise de salario por bonus"
sns.lmplot("salary", "bonus", df[all_features_finacial_payments],
          scatter_kws={"marker":"x", "color":"blue"},
          line_kws={"linewidth":1, "color": "orange"})

print "Empregados que possuem salarios acima de 1Mi e bonus 5Mi  respectivamente"
print
salay_gte1MI_or_bonus_gte5MI = df[(df['salary'] >= 1000000) | (df['bonus'] >= 5000000)]
print salay_gte1MI_or_bonus_gte5MI

print "A chave TOTAL e um sumarizador, portanto podera ser removido e eh um outlier, pois nao se trata de um POI ou candidato a POI, os demais usuarios desta lista devera ser analisado pois podem ser candidatos a POI"
pois_candidates.update(['FREVERT MARK A','LAVORATO JOHN J'])
print pois_candidates

pessoas_para_serem_removidas.add('TOTAL')
print "Pessoas a serem removidas "
print pessoas_para_serem_removidas

### Task 2: Remove outliers
df = remove_outliers_df(df, pessoas_para_serem_removidas)
print "Analise de salario por bonus apos a remocao"
sns.lmplot("salary", "bonus", df[all_features_finacial_payments],
          scatter_kws={"marker":"x", "color":"blue"},
          line_kws={"linewidth":1, "color": "orange"})

#Tukey's fences
#Other methods flag observations based on measures such as the interquartile range. For example, if {\displaystyle Q_{1}} Q_{1} and {\displaystyle Q_{3}} Q_{3} are the lower and upper quartiles respectively, then one could define an outlier to be any observation outside the range:

#Q1-k(Q3-Q1),Q3+k(Q3-Q1)
#for some nonnegative constant {\displaystyle k} k. John Tukey proposed this test, where {\displaystyle k=1.5} k=1.5 indicates an "outlier", and {\displaystyle k=3} k=3 indicates data that is "far out".[15]

Q1 = df.quantile(q=0.25)
Q3 = df.quantile(q=0.75)
k = 1.5

outliers = df[(df < (Q1-k*(Q3-Q1))) | (df > (Q3+k*(Q3-Q1)))].count(axis=1)
outliers.sort_values(axis=0, ascending=False, inplace=True)
outliers[outliers > 1]

non_poi = df[df['poi'] == False]
salary_bonus_array = ['salary','bonus']

Q1 = non_poi[salary_bonus_array].quantile(q=0.25)
Q3 = non_poi[salary_bonus_array].quantile(q=0.75)
outliers = df[(df[salary_bonus_array] < (Q1-k*(Q3-Q1))) | (df[salary_bonus_array] > (Q3+k*(Q3-Q1)))].count(axis=1)
outliers.sort_values(axis=0, ascending=False, inplace=True)
df.loc[get_array_string_df(outliers[outliers > 0])]

print "Nao possui outlier com dados de salario e bonus"

pois_count = df['poi'].value_counts()
total_items = df.shape[0]
print "Apos ajustes e remover os outliers, o dataset ficou com %d registros"% total_items
print "Destes um total de dataset possuem %d (%2.f Porcento) POIs e %d (%2.f Porcento)non-POIs" % (pois_count[1],(pois_count[1] * 1. / total_items) * 100., pois_count[0], (pois_count[0] * 1. / total_items) * 100.)

### Task 3: Create new feature(s)
### Criacao de novas caracteristicas (relacionado com o mini-projeto: Licao 11)
df['sum_total_poi_messages'] =  df['from_poi_to_this_person'] + df['from_this_person_to_poi']

df['bonus_over_total_salary'] = df['bonus'] / df['total_payments']
df['salary_over_total_salary'] = df['salary'] / df['total_payments']

df['shared_receipt_over_from_messages'] =  df['shared_receipt_with_poi'] / df['from_messages']
df['shared_receipt_over_to_messages'] =  df['shared_receipt_with_poi'] / df['to_messages']

new_features = ['sum_total_poi_messages',
                'bonus_over_total_salary',
                'salary_over_total_salary',
                'shared_receipt_over_from_messages',
                'shared_receipt_over_to_messages']

# ajustes de possiveis dados nulos
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(value=0.0, inplace=True)

#criar my_dataset com dataframe apos a limpeza dos dados e criacao das novas features
my_dataset = pd.DataFrame.to_dict(df,orient='index')
my_feature_list = features_list[:]

### Criar duas variaveis locais labels e features para serem utilizadas para teste local
# Selecao de caracteristicas feita de forma inteligente (relacionado com o mini-projeto: Licao 11)
labels,features = df[my_feature_list[0]],df[my_feature_list[1:]]

from sklearn.feature_selection import SelectKBest

k = 8
k_best = SelectKBest(k=k)
k_best.fit(features, labels)
scores = k_best.scores_
unsorted_pairs = zip(my_feature_list[1:], scores)
sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
k_best_features = dict(sorted_pairs[:k])

print "{0} melhores caracteristicas: {1} a serem utilizadas \n".format(k, k_best_features.keys())
print sorted_pairs

print "Inserir na minha lista de features, as features levantadas pelo k best e as 3 novas features criadas"
my_feature_list_old = my_feature_list[:]
my_feature_list = [target_label] + k_best_features.keys() + new_features
print my_feature_list

# print features
print "{0} caracteristicas selecionadas: {1}\n".format(len(my_feature_list) - 1, my_feature_list[1:])

### Criar duas variaveis locais labels e features para serem utilizadas para teste local
# Ajuste de escala das caracteristicas feito corretamente
df2 = df.copy()
labels,features = df[my_feature_list[0]],df[my_feature_list[1:]]

# escalonamento de caracteristicas via min-max
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

# copiand as features escalonadas para dentro do dataframe
df2[my_feature_list[1:]] = features

#realizando uma copia do my_dataframe para um outro atributo para avaliar os atributos iniciais
my_dataset_old = my_dataset.copy()

# criar o my_dataset com os dados do dataframe com os dados escalonados
my_dataset = pd.DataFrame.to_dict(df2,orient='index')

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

print "####################################################"
print
print
print
print "###################################"
print "####                            ###"
print "####   Classificadores          ###"
print "#### Sem Featuring Engeneerinng ###"
print "###################################"
print
print "###################################"
print "####                            ###"
print "####   GaussianNB               ###"
print "####                            ###"
print "###################################"
print
### Naive Bayes Gaussian
from sklearn.naive_bayes import GaussianNB
nf_clf = GaussianNB()
print "Realizando teste com o dataset sem escalonamento de variaveis"
test_classifier(nf_clf, my_dataset_old, my_feature_list_old)
print
print
print "Realizando teste com o dataset com escalonamento de variaveis e feature engeneering com a criacao das novas 3 variaveis"
test_classifier(nf_clf, my_dataset, my_feature_list)

dump_classifier_and_data(nf_clf, my_dataset, my_feature_list)
main()

print
print
print "Tunning"
print "Nao Possui Tunning"
print
print "###################################"
print "####                            ###"
print "####  DecisionTreeClassifier    ###"
print "####                            ###"
print "###################################"
print
### Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()
print "Realizando teste com o dataset sem escalonamento de variaveis"
test_classifier(dt_clf, my_dataset_old, my_feature_list_old)
print
print
print "Realizando teste com o dataset com escalonamento de variaveis e feature engeneering com a criacao das novas 3 variaveis"
test_classifier(dt_clf, my_dataset, my_feature_list)
print
print
print "Tuning GridSearchCV"
print
print
# parameters = {"criterion": ['gini', 'entropy'],
#               "min_samples_split": [40]
#               }
#
parameters = {"criterion": ['gini', 'entropy'],
              "splitter":['best'],
              "max_depth" : [15,20]
              }
opt_model_dt_clf = GridSearchCV(dt_clf, param_grid=parameters)
test_classifier(opt_model_dt_clf, my_dataset, my_feature_list)
print
print
print "Estimator"
print
print opt_model_dt_clf.best_estimator_
print
print "Features importances"
print
print opt_model_dt_clf.best_estimator_.feature_importances_
print
print "###################################"
print "####                            ###"
print "####  DecisionTreeRegressor     ###"
print "####                            ###"
print "###################################"
print
### Decision Tree Regressor
from sklearn import tree
dtr_model = tree.DecisionTreeRegressor()
print "Realizando teste com o dataset sem escalonamento de variaveis"
test_classifier(dtr_model, my_dataset_old, my_feature_list_old)
print
print
print "Realizando teste com o dataset com escalonamento de variaveis e feature engeneering com a criacao das novas 3 variaveis"
test_classifier(dtr_model, my_dataset, my_feature_list)
print
print
print "Tuning GridSearchCV"
print
print
parameters = {"criterion": ['mse'],
              "splitter": ['best', 'random'],
              "presort": [True, False],
              "min_samples_split": [40,60],
              "random_state": [20, 40]
              }
opt_model_dtr_model = GridSearchCV(dtr_model, param_grid=parameters)

test_classifier(opt_model_dtr_model, my_dataset, my_feature_list)
print
print
print "Estimator"
print
print opt_model_dtr_model.best_estimator_
print
print "Features importances"
print
print opt_model_dtr_model.best_estimator_.feature_importances_
print
print "####################################################"


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

clf = nf_clf
print clf
dump_classifier_and_data(clf, my_dataset, my_feature_list)
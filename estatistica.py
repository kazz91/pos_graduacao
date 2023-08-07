import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

**a- Leia o arquivo para um Data Frame (DF)**


#importar planilha para um dataframe
from google.colab import files


df = pd.read_csv('/content/drive/MyDrive/trabalho/dataset_diabetes/diabetic_data.csv')

**b- Exiba as Estatísticas Descritivas do DF usando o DESCRIBE**



#analise exploratoria todo dataframe
df.describe()

**c- Calcule e armazene o vetor de médias por Município (media_mun) (92) e o
vetor de médias por Ano (media_ano) (22)**

#media ano
media_ano = df.mean((0),(1)) 
print("O valor da média pelos anos é\n\n",media_ano)

#media municipio
media_mun = df.mean ((1),(0))
print("O valor da média pelos municípios é\n\n",media_mun)

**d- Realize uma Análise Exploratória de Dados para os vetores de médias por
município e por ano (media_mun e media_ano)**

#Ano analise exploratoria
media_a=media_ano.mean()
med_a=media_ano.median()
moda_a=media_ano.mode()
min_a=media_ano.min()
max_a=media_ano.max()

q1a = media_ano.quantile(q=0.25)  #1º quartil
q2a = media_ano.quantile(q=0.50)  #2º quartil
q3a = media_ano.quantile(q=0.75)  #3º quartil

var_ano = media_ano.var()      #variancia
dp_ano  = media_ano.std()      #desvio padrao
cv=dp_ano/media_a              #coeficiente de variacao

print ("Média:",(media_a))
print ("Mediana:",(med_a))
print ("Moda", (moda_a))
print ("Minimo",(min_a))
print ("Máximo",(max_a))
print ("1º Quartil:",(q1a))
print ("2º Quartil:",(q2a))
print ("3º Quartil:",(q3a))
print ("Variancia:",(var_ano))
print ("Desvio padrao:",(dp_ano))
print ("Coeficiente de Variação",(cv))

#histograma Ano
plt.hist(media_ano)
plt.title('Ano')
plt.xlabel('Taxa Mortalidade')
plt.ylabel('Frequencia')
plt.show()


#boxplot Ano
plt.boxplot(media_ano)
plt.title('Ano')
plt.xlabel('Taxa Mortalidade')
plt.show()


# Municipios analises
media_m=media_mun.mean()
med_m=media_mun.median()
moda_m=media_mun.mode()
min_m=media_mun.min()
max_m=media_mun.max()

q1m = media_mun.quantile(q=0.25) #1º quartil
q2m = media_mun.quantile(q=0.50) #2º quartil
q3m = media_mun.quantile(q=0.75) #3º quartil

var_mun = media_mun.var()      #variancia
dp_mun  = media_mun.std()      #desvio padrao
cvmun=dp_mun/media_m           #coeficiente de variacao

print ("Média:",(media_m))
print ("Mediana:",(med_m))
print ("Moda", (moda_m))
print ("Minimo",(min_m))
print ("Máximo",(max_m))
print ("1º Quartil:",(q1m))
print ("2º Quartil:",(q2m))
print ("3º Quartil:",(q3m))
print ("Variancia:",(var_mun))
print ("Desvio padrao:",(dp_mun))
print ("Coeficiente de Variação",(cvmun))

#histograma municipios
plt.hist(media_mun)
plt.title('Ano')
plt.xlabel('Taxa Mortalidade')
plt.ylabel('Frequencia')
plt.show()

#boxplot municipio
plt.boxplot(media_mun)
plt.title('Ano')
plt.xlabel('Taxa Mortalidade')
plt.show()

**e- OUTLIERS**
Determine os valores dos outliers para cada variável (media_mun e media_ano)
Identifique e exiba os municípios e os anos que são outliers.

#Outliers Ano

lia = q1a - 1.5*(q3a-q1a)
lsa = q3a + 1.5*(q3a-q1a)

print("Limite Inferior:",'%.3f' %(lia))
print("Limite Superior:",'%.3f' %(lsa))
print()
print("OUTLIERS\n")
for i in media_ano.index:
    if (media_ano.loc[i,] < lia) or (media_ano.loc[i,] > lsa):
     print("ano: {:.4s}".format(i),"media_ano: {:.4f}".format(media_ano.loc[i,]))
     print()


#Outliers municipios
lim = q1m - 1.5*(q3m-q1m)
lsm = q3m + 1.5*(q3m-q1m)

print("Limite Inferior",'%.3f' %(lim))
print("Limite Superior",'%.3f' %(lsm))
print()
print("OUTLIERS\n")
for i in range(92):
    if (media_mun[i] < lim) or (media_mun[i] > lsm):
      print("media_mun: {:.4f}".format(media_mun[i]))
      print("município: {:.10s}".format(df.loc[i,'Município']))
      print()


**f- Crie uma rotina para classificar a taxa média de mortalidade por município
(media_mun) em 4 categorias (1-2-3-4) correspondentes aos quartis.
Adicione esse indicador a última coluna do Data Frame original.**

q1m = media_mun.quantile(q=0.25) #1º quartil
q2m = media_mun.quantile(q=0.50) #2º quartil
q3m = media_mun.quantile(q=0.75) #3º quartil
q4m = media_mun.quantile(q=1) #4º quartil
dif = q4m-q1m

print ("1º Quartil:",(q1m))
print ("2º Quartil:",(q2m))
print ("3º Quartil:",(q3m))
print ("4º Quartil:",(q4m))
print ("Diferença interquartilica", (dif))
print()

q1_vetor = []
q2_vetor = []
q3_vetor = []
q4_vetor = []

nova = [] #nova tabela

for x in media_mun.values:
    if x <= q1m:
      q1_vetor.append(x)
      nova.append('q1')
    elif q1m < x <= q2m:
      q2_vetor.append(x)
      nova.append('q2')
    elif q2m < x <= q3m:
      q3_vetor.append(x)
      nova.append('q3')
    else:
      q4_vetor.append(x)
      nova.append('q4')

df['quartile'] = nova
print(df)


**g- Gere indicadores numéricos e gráficos que permitam avaliar a distribuição
desse indicador. Lembre-se que é uma variável discreta…**

import plotly.express as px
table = df[['Município', 'quartile']]
print(table)

plt.hist(df['quartile'])
plt.title('Distribuição dos Quartis')
plt.xlabel('Quartil')
plt.ylabel('Frequência')
plt.show()

 **AMOSTRAGEM**

#AMOSTRA ALEATÓRIA SIMPLES SEM REPOSIÇÃO (PADRÃO) - TAMANHO n
df_sample = df.sample(n=10)

df_sample.shape
df_sample.info()
print(f'dataframe original {df.shape}')
print(f'dataframe de amostragem {df_sample.shape}')

#dados estatisticos 
media_m=df_sample.mean()

var_mun = df_sample.var()      #variancia
dp_mun  = df_sample.std()      #desvio padrao
cvmun=dp_mun/media_m           #coeficiente de variacao

print ("Média:",(media_m))
print ("Variancia:",(var_mun))
print ("Coeficiente de Variação",(cvmun))

#AMOSTRA ALEATÓRIA SIMPLES SEM REPOSIÇÃO (PADRÃO) - FRAÇÃO AMOSTRAL (frac=0.10)

df_sample = df.sample(frac=0.10)

# Informações do dataframe criado:
df_sample.shape
df_sample.info()
print(f'dataframe original {df.shape}')
print(f'dataframe de amostragem sem reposição {df_sample.shape}')

#dados estatisticos 
media_m=df_sample.mean()

var_mun = df_sample.var()      #variancia
dp_mun  = df_sample.std()      #desvio padrao
cvmun=dp_mun/media_m           #coeficiente de variacao

print ("Média:",(media_m))
print ("Variancia:",(var_mun))
print ("Coeficiente de Variação",(cvmun))

#AMOSTRA ALEATÓRIA SIMPLES COM REPOSIÇÃO - TAMANHO n
df_sample = df.sample(n=10,replace=True)

df_sample.shape
df_sample.info()
print(f'dataframe original {df.shape}')
print(f'dataframe de amostragem {df_sample.shape}')

#dados estatisticos 
media_m=df_sample.mean()

var_mun = df_sample.var()      #variancia
dp_mun  = df_sample.std()      #desvio padrao
cvmun=dp_mun/media_m           #coeficiente de variacao

print ("Média:",(media_m))
print ("Variancia:",(var_mun))
print ("Coeficiente de Variação",(cvmun))


**INTERVALO DE CONFIANÇA**

import scipy.stats
import numpy as np

def confident_interval(Xs, n, nc , sigma , s):
    global IC1,IC2
    z = abs(scipy.stats.norm.ppf((1 - nc)/2.)) # RECEBE PROB DEVOLVE (z) QUANTIL(prob,média,dp)
    print("z: {:.2f}".format(z))                                          
    print("sigma:",sigma) 
    if(sigma != -1):   # SE VARIÂNCIA POP É CONHECIDA DISTRIBUIÇÃO NORMAL(CASO-1)
        IC1 = Xs - z*sigma/np.sqrt(n) # LIMITE INFERIOR
        IC2 = Xs + z*sigma/np.sqrt(n) # LIMITE SUPERIOR
        print("CASO-1 - VARIÂNCIA/DP - POP - CONHECIDA")
    else:              # SE VAR POP É DESCONHECIDA ==> TESTA n (AMOSTRA) 
        if(n >= 30):   # SE n (AMOSTRA) > 30 DISTRIBUIÇÃO NORMAL (CASO-2)
            print("s:",s)
            print("CASO-2 - VARIÂNCIA/DP-POP-DESCONHECIDO - n >= 30")
            IC1 = Xs - z*s/np.sqrt(n) # LIMITE INFERIOR (usa s em vez de sigma)
            IC2 = Xs + z*s/np.sqrt(n) # LIMITE SUPERIOR (usa s em vez de sigma) 
        else: # SE n (AMOSTRA) < 30 ENTÃO T-STUDENT (CASO-3)
            t = scipy.stats.t.ppf((1 + nc) / 2., n-1) 
            print("t:",t)
            print("s:",s)
            print("CASO-3 - VAR/DP POP DECONHECIDO n < 30")
            IC1 = Xs - t*s/np.sqrt(n)
            IC2 = Xs + t*s/np.sqrt(n)
    return [IC1, IC2]

global IC1
Xs = df_sample.mean()          # MÉDIA NA AMOSTRA
n  = 20                       # TAMANHO DA AMOSTRA
nc = 0.95                      # NÍVEL DE CONFIANÇA
sigma = 1                      # DEVIO PADRÃO DA POPULAÇÃO  
s= df_sample.std()               # DEVIO PADRÃO DA AMOSTRA
print("s:",s)
IC = confident_interval(Xs,n,nc,sigma,s)
print()
print('Intervalo Confiança:', IC)
print()
print("Amplitude Intervalo:", (IC2-IC1))
print()
print("nc:",nc)
print()

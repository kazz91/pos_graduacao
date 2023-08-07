#importando bibliotecas
import pandas as pd
import io

#importar csv para um dataframe
from google.colab import files

arquivo = files.upload()

df = pd.read_csv('/content/non-verbal tourist data.csv')

#obter informações sobre os dados para ver se existem valores nulos
df.info()

#visualizando as 10 primeiras linhas do dataframe
df.head(10)

#substituindo o '?' por 'N/A'
df.replace({'?':'N/A'},inplace=True)

#traduzindo as colunas
df = df.rename(columns={'sex':'Gênero',
                        'age':'Idade',
                        'country':'País',
                        'returning':'Retornando',
                        'Hostile - friendly':'Hostil - amigável',
                        'Authoritative -anarchic ':'Autoritário -anárquico',
                        'GImg1':'Aperto de mão',
                        'GImg2':'Abraço', 
                        'GImg3':'Beijo',
                        'PImg1':'Postura permissiva',
                        'PImg2':'Postura interessada', 
                        'PImg3':'Postura neutra',
                        'PImg4':'Postura reflexiva',
                        'PImg5':'Postura negativa',
                        'Tense - relaxed':'Tensão - relaxamento',
                        'TAudio1':'Proxêmica/Proximidade',
                        'TAudio2':'Sarcástica',
                        'TAudio3':'Amigável',
                        'QAudio1':'Cuspindo',
                        'QAudio2':'Entre Dentes',
                        'QAudio3':'Murmurando',
                        'Proxemics':'Autoritária',
                        'Type of Client':'Tipo de Cliente'
                        })

#traduzindo termos para português
df.replace({'likes':'Gosta',
            'dislikes':'Não gosta',
            'indiferent':'Indiferente',
            'no':'Não',
            'yes':'Sim',
            'uruguay':'Uruguai',
            'brasil':'Brasil',
            'england':'Inglaterra',
            'canada':'Canadá',
            'argentina':'Argentina',
            'hungary':'Hungría',
            'polish':'Polônia',
            'colombia':'Colômbia',
            'scotland':'Escócia',
            'germany':'Alemanha',
            'cuba':'Cuba',
            'italy':'Itália',
            'russia':'Rússia',
            'mexico':'México',
            'spain':'Espanha',
            'A':'Íntimo (Entre 15cm-45cm)',
            'B':'Pessoal (Entre 46cm-122cm)',
            'C':'Social (Entre 123cm-360cm)',
            'D':'Público (Maior que 360cm)'},
           inplace=True)


#visualizando o dataframe com as alterações feitas
display(df)

#analise exploratoria
df.describe()

#exportando pra uma planilha excel
df.to_excel('df.xlsx')
files.download('df.xlsx')

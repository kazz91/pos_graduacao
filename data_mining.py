# instalar as dependências
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q https://archive.apache.org/dist/spark/spark-3.3.0/spark-3.3.0-bin-hadoop2.tgz 
!tar xf spark-3.3.0-bin-hadoop2.tgz

# configurar as variáveis de ambiente e o Spark
import os

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.3.0-bin-hadoop2.tgz"

!pip install -q findspark

# tornar o pyspark "importável"
import findspark
findspark.init('spark-3.3.0-bin-hadoop2')

# iniciar uma sessão local
from pyspark.sql import SparkSession,SQLContext
import matplotlib.pyplot as plt

spark = SparkSession.builder.master("local[*]").appName("DadosOlist").getOrCreate()

from google.colab import drive
drive.mount('/content/drive') # Importacão da planilha do trabalho pelo Google Drive

#Importa a panilha para um dataframe
pd = spark.read.csv('/content/drive/MyDrive/planilhas/olist_products_dataset.csv',inferSchema=True, header=True, sep=',') 
oid = spark.read.csv('/content/drive/MyDrive/planilhas/olist_order_items_dataset.csv',inferSchema=True, header=True, sep=',')
pyd = spark.read.csv('/content/drive/MyDrive/planilhas/olist_order_payments_dataset.csv',inferSchema=True, header=True, sep=',')
od = spark.read.csv('/content/drive/MyDrive/planilhas/olist_orders_dataset.csv',inferSchema=True, header=True, sep=',') 
sd = spark.read.csv('/content/drive/MyDrive/planilhas/olist_sellers_dataset.csv',inferSchema=True, header=True, sep=',')
cd = spark.read.csv('/content/drive/MyDrive/planilhas/olist_customers_dataset.csv',inferSchema=True, header=True, sep=',')
gd = spark.read.csv('/content/drive/MyDrive/planilhas/olist_geolocation_dataset.csv',inferSchema=True, header=True, sep=',')

pd.createOrReplaceTempView("pd")
oid.createOrReplaceTempView("oid")
pyd.createOrReplaceTempView("pyd")
od.createOrReplaceTempView("od")
sd.createOrReplaceTempView("sd")
cd.createOrReplaceTempView("cd")
gd.createOrReplaceTempView("gd")

pd.show()

oid.show()

pyd.show()

od.show()

sd.show()

cd.show()

gd.show()

query=spark.sql("select sd.seller_state as Estado, count (DISTINCT od.customer_id) as Clientes, count (DISTINCT sd.seller_id) as Vendedores, round(sum(cast(pyd.payment_value as decimal)),2) as Faturamento from od join oid on oid.order_id=od.order_id join sd on oid.seller_id=sd.seller_id join pyd on od.order_id=pyd.order_id join pd on oid.product_id=pd.product_id group by sd.seller_state ")

query.show(100,False)

import pyspark.pandas as ps
import pandas as pf
analise=query.to_pandas_on_spark()
import os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"


clientes_media=analise['Clientes'].mean()
clientes_mediana=analise['Clientes'].median()
clientes_primeiro_quartil=analise['Clientes'].quantile(.25)
clientes_segundo_quartil=analise['Clientes'].quantile(.50)
clientes_terceiro_quartil=analise['Clientes'].quantile(.75)
clientes_quarto_quartil=analise['Clientes'].quantile(1.0)
clientes_desvio_padrao=analise['Clientes'].std()
clientes_maximo=analise['Clientes'].max()
clientes_minimo=analise['Clientes'].min()

vendidos_media=analise['Vendedores'].mean()
vendidos_mediana=analise['Vendedores'].median()
vendidos_primeiro_quartil=analise['Vendedores'].quantile(.25)
vendidos_segundo_quartil=analise['Vendedores'].quantile(.50)
vendidos_terceiro_quartil=analise['Vendedores'].quantile(.75)
vendidos_quarto_quartil=analise['Vendedores'].quantile(1.0)
vendidos_desvio_padrao=analise['Vendedores'].std()
vendidos_maximo=analise['Vendedores'].max()
vendidos_minimo=analise['Vendedores'].min()

vendas_media=analise['Faturamento'].mean()
vendas_mediana=analise['Faturamento'].median()
vendas_primeiro_quartil=analise['Faturamento'].quantile(.25)
vendas_segundo_quartil=analise['Faturamento'].quantile(.50)
vendas_terceiro_quartil=analise['Faturamento'].quantile(.75)
vendas_quarto_quartil=analise['Faturamento'].quantile(1.0)
vendas_desvio_padrao=analise['Faturamento'].std()
vendas_maximo=analise['Faturamento'].max()
vendas_minimo=analise['Faturamento'].min()

from pyspark.sql.types import StructType,StructField, StringType

data = [(clientes_media,clientes_mediana,clientes_primeiro_quartil,clientes_segundo_quartil,clientes_terceiro_quartil,clientes_quarto_quartil,clientes_desvio_padrao,clientes_maximo,clientes_minimo),
  (vendidos_media,vendidos_mediana,vendidos_primeiro_quartil,vendidos_segundo_quartil,vendidos_terceiro_quartil,vendidos_quarto_quartil,vendidos_desvio_padrao,vendidos_maximo,vendidos_minimo),
  (vendas_media,vendas_mediana,vendas_primeiro_quartil,vendas_segundo_quartil,vendas_terceiro_quartil,vendas_quarto_quartil,vendas_desvio_padrao,vendas_maximo,vendas_minimo)
  ]

columns = StructType([ \
StructField("media",StringType(),True), \
StructField("mediana",StringType(),True), \
StructField("primeiro_quartil",StringType(),True), \
StructField("segundo_quartil", StringType(), True), \
StructField("terceiro_quartil", StringType(), True), \
StructField("quarto_quartil", StringType(), True), \
StructField("desvio_padrao", StringType(), True), \
StructField("maximo", StringType(), True), \
StructField("minimo", StringType(), True) \
  ])



spark2 = SparkSession.builder.master("local[*]").appName("AnalisExploratória").getOrCreate()
sdf = spark2.createDataFrame(data=data,schema = columns)

sdf.createOrReplaceTempView("sdf")


export_analise=spark.sql("select round(media,2) as media,mediana,primeiro_quartil,segundo_quartil,terceiro_quartil,round(cast(quarto_quartil as decimal),2) as quarto_quartil,round(desvio_padrao,2) as desvio_padrao,maximo,minimo from sdf")

export_analise.show(100,False)

export_analise.toPandas().to_csv("OlistAnalise.csv", index=False)

from google.colab import files
files.download("OlistAnalise.csv")
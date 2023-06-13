import pandas as pd
pd.set_option('display.float_format', lambda x: '%.2f' % x)
import numpy as np
import datetime
import pandas_datareader.data as web
import yfinance as yf
yf.pdr_override()
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML 
display(HTML("<style>.container { width:100% !important; }</style>"))

# dados
start = datetime.datetime(2023, 1, 2)
end = datetime.datetime(2023, 6, 12)

petr4 = web.get_data_yahoo('PETR4.SA',start,end) 
usim5 = web.get_data_yahoo('USIM5.SA',start,end) 
vale3 = web.get_data_yahoo('VALE3.SA',start,end) 
bbas3 = web.get_data_yahoo('BBAS3.SA',start,end) 
vale3 = web.get_data_yahoo('VALE3.SA',start,end) 
ibov = web.get_data_yahoo('^BVSP',start,end)

cotas = pd.concat([petr4, usim5, vale3])



# trash
cotas = pd.DataFrame() # criar um dataframe
%matplotlib inline # para Jupyter Notebook
petr4.head(3) # tres primeiros
petr4.tail(3) # tres ultimos
petr4.loc['2019-05-20'] # cotacao em dia especifico
petr4.loc['2019-06-01':'2019-06-30'] # cotacoes em periodo especifico

def plot_stock(df, stock_name):                 # grafico
    plt.rcParams["figure.figsize"] = [14,5]
    plt.plot(df.index, df['Close'])    
    plt.xlabel('Data')
    plt.ylabel('Valor (R$)')
    plt.title(stock_name)

plot_stock(petr4, 'Petrobras')
plt.show()

import pandas as pd
pd.set_option('display.float_format', lambda x: '%.2f' % x)
import numpy as np
import datetime
import pandas_datareader.data as web   #pandas-datareader
import yfinance as yf
yf.pdr_override()
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML   #ipython
display(HTML("<style>.container { width:100% !important; }</style>"))
from pypfopt import EfficientFrontier   #PyPortfolioOpt
from pypfopt import risk_models
from pypfopt import expected_returns

# dados
start = datetime.datetime(2023, 1, 2)
end = datetime.datetime(2023, 6, 12)

petr4 = web.get_data_yahoo('PETR4.SA',start,end) 

petr4.head(3) # tres primeiros
petr4.tail(3) # tres ultimos
petr4.loc['2023-05-22'] # cotacao em dia especifico
petr4.loc['2023-06-01':'2023-06-16'] # cotacoes em periodo especifico

# grafico de preco
petr4['Adj Close'].plot()
plt.xlabel("Data")
plt.ylabel("Ajustado")
plt.title("Preço de PETR4")
plt.show()

petr4_daily_returns = petr4['Adj Close'].pct_change()
petr4_monthly_returns = petr4['Adj Close'].resample('M').ffill().pct_change()

# grafico diario
fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
ax1.plot(petr4_daily_returns)
ax1.set_xlabel("Date")
ax1.set_ylabel("Percent")
ax1.set_title("PETR4 daily returns data")
plt.show()

# grafico mensal
fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
ax1.plot(petr4_monthly_returns)
ax1.set_xlabel("Date")
ax1.set_ylabel("Percent")
ax1.set_title("PETR4 monthly returns data")
plt.show()

# histograma
fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
petr4_daily_returns.plot.hist(bins = 60)
ax1.set_xlabel("Daily returns %")
ax1.set_ylabel("Percent")
ax1.set_title("PETR4 daily returns data")
ax1.text(-0.35,200,"Extreme Low\nreturns")
ax1.text(0.25,200,"Extreme High\nreturns")
plt.show()

# para multiplos ativos
tickers = ["ABEV3.SA", "ALPA4.SA", "ALSO3.SA", "ARZZ3.SA", "ASAI3.SA", "AZUL4.SA", "B3SA3.SA", "BBAS3.SA", "BBDC3.SA", "BBDC4.SA", "BBSE3.SA", "BEEF3.SA", "BPAC11.SA", "BRAP4.SA", "BRFS3.SA", "BRKM5.SA", "CASH3.SA", "CCRO3.SA", "CIEL3.SA", "CMIG4.SA", "CMIN3.SA", "COGN3.SA", "CPFE3.SA", "CPLE6.SA", "CRFB3.SA", "CSAN3.SA", "CSNA3.SA", "CVCB3.SA", "CYRE3.SA", "DXCO3.SA", "EGIE3.SA", "ELET3.SA", "ELET6.SA", "EMBR3.SA", "ENBR3.SA", "ENEV3.SA", "ENGI11.SA", "EQTL3.SA", "EZTC3.SA", "FLRY3.SA", "GGBR4.SA", "GOAU4.SA", "GOLL4.SA", "HAPV3.SA", "HYPE3.SA", "IGTI11.SA", "IRBR3.SA", "ITSA4.SA", "ITUB4.SA", "JBSS3.SA", "KLBN11.SA", "LREN3.SA", "LWSA3.SA", "MGLU3.SA", "MRFG3.SA", "MRVE3.SA", "MULT3.SA", "NTCO3.SA", "PCAR3.SA", "PETR3.SA", "PETR4.SA", "PETZ3.SA", "PRIO3.SA", "RADL3.SA", "RAIL3.SA", "RAIZ4.SA", "RDOR3.SA", "RENT3.SA", "RRRP3.SA", "SANB11.SA", "SBSP3.SA", "SLCE3.SA", "SMTO3.SA", "SOMA3.SA", "SUZB3.SA", "TAEE11.SA", "TIMS3.SA", "TOTS3.SA", "UGPA3.SA", "USIM5.SA", "VALE3.SA", "VBBR3.SA", "VIIA3.SA", "VIVT3.SA", "WEGE3.SA", "YDUQ3.SA"]
multpl_stocks = web.get_data_yahoo(tickers, start = start, end = end)

multpl_stock_daily_returns = multpl_stocks['Adj Close'].pct_change()
multpl_stock_monthly_returns = multpl_stocks['Adj Close'].resample('M').ffill().pct_change()

fig = plt.figure()
(multpl_stock_monthly_returns + 1).cumprod().plot()
plt.show()

# media
print(multpl_stock_daily_returns.mean())

# covariancia e correlacao
print(multpl_stock_daily_returns.cov())
print(multpl_stock_daily_returns.corr())

# definindo parametros para a otimizacao
mu = multpl_stock_daily_returns.mean()
S = multpl_stock_daily_returns.cov()

type(mu)
type(S)


# Optimize for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
ef.save_weights_to_file("weights.csv")  # saves to file
print(cleaned_weights)
ef.portfolio_performance(verbose=True)
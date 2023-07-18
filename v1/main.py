import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import data
from datetime import datetime
import yfinance as yf

# today and a year ago
today = datetime.now()
yearAgo = datetime(today.year-1, today.month, today.day)

# three basic companies
companies = ["TSLA", "MSFT", "AMZN"]

# download the data of every company listed in companies
for company in companies:
    globals()[company] = yf.download(company, start="2022-06-06", end="2023-06-06")

companydata = [TSLA, MSFT, AMZN]

# one stock plotted over time

# plt.figure(figsize=(15, 7))
# plt.plot(TSLA["Adj Close"])
# plt.xlabel("Date")
# plt.ylabel("Price")
# plt.grid()
# plt.show()

# moving average calculation: sum = sum of prices for the past few days; ma = sum/x

madays = [10, 20, 30]
for ma in madays:
    mastring = "MA: {}".format(ma)
    TSLA[mastring] = TSLA["Adj Close"].rolling(ma).mean()
    MSFT[mastring] = MSFT["Adj Close"].rolling(ma).mean()
    AMZN[mastring] = AMZN["Adj Close"].rolling(ma).mean()

# plot a company"s moving average, explained above

def plotGraphic(company, companystring):
    plt.figure(figsize=(15, 6))
    plt.plot(company["Adj Close"])
    plt.plot(company["MA: 10"])
    plt.plot(company["MA: 20"])
    plt.plot(company["MA: 30"])
    
    plt.title(companystring)
    plt.xlabel("Date")
    plt.ylabel("Price")

    plt.legend(("Adj Close","MA: 10", "MA: 20", "MA:30"))
    plt.grid()
    plt.show()

for i in range(len(companydata)):
    plotGraphic(companydata[i], companies[i])

for i in range(len(companydata)):
    companydata[i]['Daily Returns'] = companydata[i]['Adj Close'].pct_change()
    sns.displot(companydata[i]['Daily Returns'].dropna(), bins=50, color='blue', kde=True)
    plt.title(companies[i])
    plt.show()

# create dataframe with stock returns
stockreturns = pd.DataFrame(data=np.array([data['Daily Returns'] for data in companydata]).T, columns=companies)
stockreturns.head()

sns.pairplot(stockreturns.dropna())

# build correlation matrix
corr = stockreturns.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(10, 10))
sns.heatmap(corr, mask=mask,  square=True, linewidths=.5, annot=True)
plt.show()

def drawjointplot(data):
    grid = sns.PairGrid(data.dropna())
    grid.map_diag(sns.histplot, bins=40, kde=True)
    grid.map_lower(sns.regplot)
    grid.map_upper(sns.kdeplot)

drawjointplot(stockreturns)

meanincome = stockreturns.mean() # mean income for each stock
covreturns = stockreturns.cov() # covariation 
count = len(stockreturns.columns)
print(meanincome, covreturns, sep='\n')

# generate random shares
def randomportfolio():
    share = np.exp(np.random.randn(count))
    share = share / share.sum()
    return share

def incomeportfolio(Rand):
    return np.matmul(meanincome.values, Rand)

def riskportfolio(Rand):
    return np.sqrt(np.matmul(np.matmul(Rand, covreturns.values), Rand))

combinations = 4000
risk = np.zeros(combinations)
income = np.zeros(combinations)
portfolio = np.zeros((combinations, count))

# create new combinations of shares
for i in range(combinations):
    rand = randomportfolio()

    portfolio[i, :] = rand
    risk[i] = riskportfolio(rand)
    income[i] = incomeportfolio(rand)

plt.figure(figsize=(15, 8))

plt.scatter(risk * 100, income * 100, c="b", marker=".")
plt.xlabel("Risk")
plt.ylabel("Income")
plt.title("Portfolios")
MaxSharpRatio = np.argmax(income / risk)
plt.scatter([risk[MaxSharpRatio] * 100], [income[MaxSharpRatio] * 100], c="r", marker="o", label="Max Sharp ratio")

plt.legend()
plt.show()

best_port = portfolio[MaxSharpRatio]
for i in range(len(companies)):
    print("{} : {}".format(companies[i], best_port[i]))

days = 365
dt = 1 / days
stockreturns.dropna(inplace=True)

mu = stockreturns.mean()
sigma = stockreturns.std()

def monte_carlo(start_price, days, mu, sigma):
    price = np.zeros(days)
    price[0] = start_price
    
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    for x in range(1, days):
        shock[x] = np.random.normal(loc=mu * dt, scale=sigma*np.sqrt(dt))
        drift[x] = mu * dt
        
        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))
        
    return price

# estimate price of tesla stocks
startprice = 199.24
sim = np.zeros(1000)

plt.figure(figsize=(15, 8))
for i in range(1000):
    result = monte_carlo(startprice, days, mu['TSLA'], sigma['TSLA'])
    sim[i] = result[days - 1]
    plt.plot(result)
    
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo analysis for Tesla')

plt.figure(figsize=(10, 7))
plt.hist(sim, bins=100)
plt.figtext(0.6, 0.7, "Mean: {} \nStd: {} \nStart Price: {}".format(sim.mean(), sim.std(), startprice))
plt.show()
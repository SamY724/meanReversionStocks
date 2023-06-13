#This code contains the strategy of comparison, in terms of mean reversion
#This strategy was selected for comparison due to its presence and relative success (warren buffet)
#This strategy is very comparable in context also to the MSc project strategy 

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import random

plt.style.use('fivethirtyeight')

#Setting seed for reproducibility of code
random.seed(42)


amp = yf.download("AMP", start="2010-01-01", end="2021-01-01", interval='1wk')
ivz = yf.download("IVZ", start="2010-01-01", end="2021-01-01", interval='1wk')
trow = yf.download("TROW", start="2010-01-01", end="2021-01-01", interval='1wk')
amg = yf.download("AMG", start="2010-01-01", end="2021-01-01", interval='1wk')
ben = yf.download("BEN", start="2010-01-01", end="2021-01-01", interval='1wk')
blk = yf.download("BLK", start="2010-01-01", end="2021-01-01", interval='1wk')


#FTSE100 Stocks

adm = yf.download("ADM",start="2010-01-01", end="2021-01-01", interval='1wk')
azn = yf.download("AZN",start="2010-01-01", end="2021-01-01", interval='1wk')
tsco = yf.download("TSCO",start="2010-01-01", end="2021-01-01", interval='1wk')
shel = yf.download("SHEL",start="2010-01-01", end="2021-01-01", interval='1wk')
bp = yf.download("BP",start="2010-01-01", end="2021-01-01", interval='1wk')
ba = yf.download("BA",start="2010-01-01", end="2021-01-01", interval='1wk')



#Russell2000 Stocks
bset = yf.download("BSET",start="2010-01-01", end="2021-01-01", interval='1wk')
kmt = yf.download("KMT",start="2010-01-01", end="2021-01-01", interval='1wk')
abm = yf.download("ABM",start="2010-01-01", end="2021-01-01", interval='1wk')
lnn = yf.download("LNN",start="2010-01-01", end="2021-01-01", interval='1wk')
anik = yf.download("ANIK",start="2010-01-01", end="2021-01-01", interval='1wk')
sasr = yf.download("SASR",start="2010-01-01", end="2021-01-01", interval='1wk')

#NaN value purging

#S&P 500 Stocks

amp = amp.dropna(axis=0)
ivz = ivz.dropna(axis=0)
trow = trow.dropna(axis=0)
amg = amg.dropna(axis=0)
ben = ben.dropna(axis=0)
blk = blk.dropna(axis=0)

#FTSE100 Stocks

adm = adm.dropna(axis=0)
azn = azn.dropna(axis=0)
tsco = tsco.dropna(axis=0)
shel = shel.dropna(axis=0)
bp = bp.dropna(axis=0)
ba = ba.dropna(axis=0)


#Russell2000 stocks

bset = bset.dropna(axis=0)
kmt = kmt.dropna(axis=0)
abm = abm.dropna(axis=0)
lnn = lnn.dropna(axis=0)
anik = anik.dropna(axis=0)
sasr = sasr.dropna(axis=0)

#Using the simple moving average is a fundamental of this strategy, as such calculation is needed

#setting the window to 7, to match the project strategies weekly parameter
window = 7


def simpleMovingAverage(stock,window=window,column= "Close"):
    return stock[column].rolling(window=window).mean()


def metricCalculation(df,ratioplot,plotMRS):
    #removing NaN Values
    df = df.dropna(axis=0)

    #calculating metrics and creating additional columns
    df["SMA"] = simpleMovingAverage(df,window,"Close")
    df["Simple_Returns"] = df.pct_change(1)["Close"]
    df["Log_Returns"] = np.log(1+df["Simple_Returns"])
    df["Ratios"] = df["Close"] / df["SMA"]

    #now to calculate the percentiles of the ratios column
    ratios = df["Ratios"].dropna()
    percentiles = [10,25,50,75,90]
    percentileValues = np.percentile(ratios,percentiles)

    if ratioplot == 1:
        plotRatios(percentileValues,df)

    sell = percentileValues[-1]
    buy = percentileValues[0]

    df["Positions"] = np.where(df.Ratios > sell, -1, np.nan)
    df["Positions"] = np.where(df.Ratios < buy, 1, df["Positions"])
    df["Positions"] = df["Positions"].ffill()

    df["Buy"] = np.where(df.Positions == 1, df["Close"],np.nan)
    df["Sell"] = np.where(df.Positions == -1, df["Close"],np.nan)

    if plotMRS == 1:
        mrsPlot(df)

    # Documenting Mean reversion returns
    df["MRS Returns"] = df.Positions.shift(1) * df.Log_Returns

    returns = np.exp(df["MRS Returns"].dropna()).cumprod()[-1] - 1

    print(f'The returns for MRS per $1 invested:',round(returns,2))

    return df



def plotRatios(percentileValues,df):
    plt.figure(figsize=(14,7))
    plt.title("Ratios")
    df["Ratios"].dropna().plot(legend=True)
    plt.axhline(percentileValues[0], c='green',label='10th Percentile')
    plt.axhline(percentileValues[2], c='orange',label='50th Percentile')
    plt.axhline(percentileValues[-1], c='red',label='90th Percentile')
    plt.show()


def mrsPlot(df):
    plt.figure(figsize=(14,7))
    plt.title('Close price w/ Buy & Sell signals')
    plt.plot(df["Close"],alpha=0.5,label="Close Price")
    plt.plot(df["SMA"],alpha=0.5, label="SMA")
    plt.scatter(df.index, df["Buy"], color='green',label="Buy",marker="+",alpha=1)
    plt.scatter(df.index, df["Sell"], color='red',label="Sell",marker="_",alpha=1)
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.legend()
    plt.show()
    

if __name__ == "__main__":

    metricCalculation(amp,1,1)
    metricCalculation(ivz,1,1)
    metricCalculation(trow,1,1)
    metricCalculation(amg,1,1)
    metricCalculation(ben,1,1)
    metricCalculation(blk,1,1)

    metricCalculation(adm,1,1)
    metricCalculation(azn,1,1)
    metricCalculation(tsco,1,1)
    metricCalculation(shel,1,1)
    metricCalculation(bp,1,1)
    metricCalculation(ba,1,1)

    metricCalculation(bset,1,1)
    metricCalculation(kmt,1,1)
    metricCalculation(abm,1,1)
    metricCalculation(lnn,1,1)
    metricCalculation(anik,1,1)
    metricCalculation(sasr,1,1)
   

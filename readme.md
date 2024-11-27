
# ML4T Final Study Guide
Topics in ML4T to study for Final - feel free to add your notes!

# NOTES FROM LECTURES:  

## 02-07 Dealing with Data
- [ ] Aggregated  - how data is combined and reported
- [ ] Tick- a successful buy/sell transaction at different exchanged (prices and volumes are different)
- [ ] Data can be represented as open, high, low, close and volume within a time period (daily data)
- [ ] Open - first transaction within the time period
- [ ] High - Highest price in the time period
- [ ] Low - Lowest price in the time Period
- [ ] Close = Last transaction within the time period
- [ ] Volume - Total volume in that time period
- [ ] Stock Split may be the reason why prices may drop suddenly
- [ ] Reason fo Stock Splits:
  -  Prices too high - cannot buy too many stocks at once and stock becomes less liquid
  -  Portfolio Evaluation because hard to finely tune when stock prices are too high
-  [ ] Solution for accounting for Split - Adjusted Close
-  [ ] Last day stays the same price, and previous "higher" (prices due to splits) - are divided by the ratio to accurately account for increase in value
-  [ ] Dividends - paid out to owners
-  [ ] Significant effect on what happens to stock
-  [ ] Day Before Dividend is paid: value of the stock goes up to match that of the divident pay out
-  [ ] Day of Dividend paid - price goes back to valuation without dividend amount
-  [ ] Adjusted Price for Dividends: We adjust prices down as we go back in history by the dividend amount
-  [ ] Last day of file: The actual close and adjusted close are the exact same, back in time, the prices will diverge
-  [ ] Survivor Bias - Start with today's universe stock: SP500 and apply algo to choose stocks to buy based on that price in the past
-  [ ] Mistake to look at membership of universe stock TODAY and use those specific stocks to create a strategy in historic data - because the SP500 in that past date - Must use Survivor Bias Free Data
## 02-08 Efficient Markets Hypothesis
- [ ] Jules Regnault in 1863
- [ ] EMH Assumptions:
  - Large Number of Investors interacting in the market for profit - they have an incentive to find opportunities where the price is out of line with value
  - New Information arrives at random times, and at random rates constantly for different stocks and investors are paying attention to that information
  - Prices adjust quickly and prices reflect all available information about that stock
- [ ] Places where information comes from:
  -  Technical Analysis Data: Price / volume (information is rapid, quick, and everybody can see it)
  -  Fundamental Data: reported quarterly, points more to the root of the value of the company
  -  Exogenous Data- Information about the world that would effect the price of the stock (Price of oil is exogenous data that would effect the price of an airline stock)
  -  Company insiders - secretive and important information - information someone has inside the company that the general public doesn't have
- [ ] 3 Forms of the EMH (Weak to Strong)
  -  Weak: Future prices cannot be predicted by analyzing historical prices, meaning you cannot profit from looking at historical data (prohibites profitting from technical analysis, but leaves room for fundemental analysis and insider information)
  -  Semi-Strong: Prices adjust immediately to new public information (prohibites profitting from technical analysis and fundemental analysis, but leaves room for insider information)
  -  Strong: Prices reflect all information public and private (even insider information won't change the price of the stock) prohibites technical analysis, fundemental analysis, and insider information
    - If strong is correct, then there is no way to make money by holding a portfolio, other than the market portfolio
- [ ] Is the EMH correct? Well, some hedge funds make a lot of money so the Strong version is probably not true since people have profited from insider trading
- [ ] P/E ratio - low price to earnings (PE ratio) corresponds to higher annualized returns - which means that PE ratios are very predictive of returns for many different stocks. This means that Fundemental analysis does have a correlation to Returns - refuting Semi Strong EMH. 
## 02-09 The Fundamental Law of active portfolio management
- [ ] 
## 02-10 Portfolio optimization and the efficient frontier
- [ ] 
## 03-05 Reinforcement learning
- [ ] 
## 03-06 Q-Learning
- [ ] 
## 03-07 Dyna
- [ ] 


# NOTES FROM READING:  

## What Hedge Funds Really Do Ch. 9
- [ ] 
## Machine Learning – Chapter 13
- [ ] 
## Introduction to Statistical Learning (2nd Edition) – Deep Learning Chapter 10
- [ ] 
## Handbook of AI and Big Data Applications in Investments - Chapter 8 
- [ ] 
## Handbook of AI and Big Data Applications in Investments - Chapters 10 and 11
- [ ] 
## Keynote on Algorithmic Bias (Drs. Isbell and Littman)
- [ ] 
## Interview with Tammer Kamel (Ed Lessons)
- [ ] 

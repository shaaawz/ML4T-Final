
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
- [ ] Warren Buffet, "Only When the tide goes out, do you discover who's been swimming naked"
- [ ] "Wide Diversification not necessary unless the investor does not know what he/she is doing" - Talking about two things: Investor Skill and Breadth (# of investments)
- [ ] Grinold's Law - Performance, Skill, Breadth
- [ ] Fundemental Law of Active Portfolio Management: Performance = skill * Sqrt(Breadth)
- [ ] To improve performace, you can improve your skill or find more opportunities for applying that skill
- [ ] Performance is summarized as Information Ratio - Sharpe Ratio of Excess Returns, the manner in which the portfolio is exceeding market ratio
- [ ] Return on the portfolio on a particular day is equal to the market component of the return (Beta) times market turn on that day  plus residual return (alpha). The alpha is due to the skill of the fund manager - the different of the market return for that day vs the price of the stock for that date
- [ ] the IR (skill )is the mean(alpha) / stdev (alpha) (reward / risk)
- [ ] Skill is summarized in IC (Information Coefficent) and Breadth is how many trading opporunities we have
- [ ] IC is the correlation of forecasts to returns (0 - 1) 0 - no correlation between what is forecased and what occures, 1 - very strong correlation
- [ ] BR, Breadth number of trading opportunities per year
- [ ] IR = IC * Sqrt(BR) 
## 02-10 Portfolio optimization and the efficient frontier
- [ ] Mean Variance Optimization or Portfolio Optimization - Given a set of equities and target return, find an allocation to each equity that minimizes risk
- [ ] What is risk? Volatility used as a measure of risk - the standard deviation of historical daily returns
- [ ] Visualizing return vs risk in a scatterplot - createa portfolio by combining multiple stocks on a risk/return scatterplot and weighting it depending on allocation within the portfolio
- [ ] Relationship between stocks in terms of covariance - relationship of risk is not just a factor or blend of the various risks, but it has to do with how they interact day to day
- [ ] Harry Markowitz: Key to it is to pick the right stocks in the right proportions - a blend of stocks and bonds is the lowest risk 
- [ ] Importance of covariance - how they move together - if they move similarily: high positive correlation coefficiant (close to 1) - if they move opposite - high negative correlation (close to -1)
- [ ] Important to blend anti-correlated stocks together to create lower volatility
- [ ] MVO Inputs: Factors it considers:
    - Expected return - What we think the future returns are of each asset
    - Volatility - Historically, how volatile each one of these assets have been
    - Covariance - A matrix which shows within each asset and every other asset, what is the correltaion of daily returns
    - Target Return - level of return between max return to min return we expect
- [ ] MVO Outputs:
    - A set of weights for each asset in the portfolio that minimizes risk, but meets the target return
- [ ] The Efficient Frontier: For any particular return level, there is an optimal portfolio
- [ ] As you reduce the return, the curve of the Efficient Frontier comes back indicating that risk is increaing as we reduce the return too much (this part of the curve is always excluded from the Efficient Frontier since no one wants increased risk for less reward)
- [ ] Tangent line from orgin to the lowest part of the Efficient Frontier, where it hits is the max sharp ratio portfolio of all these assets
## 03-05 Reinforcement learning
- [ ] Sense, Think, Act cycle: Observe s' state, processes s π, outputs a action - affects environment transitions t to new state
- [ ] R reward - happens when robot takes a certain action: take actions to maximize reward π - simple look up table
- [ ] Mapping trading to RL:
    - State: market features, prices, our current holdings of stock, historical value
    - Action: Buy, Sell, Do Nothing
    - Reward: Money/Return we make
- [ ] Markov decision Problem (MDP)
    - Set of states S
    - Set of actions A
    - Transition function T[s, a, s']: Probability we go from state s to state s', if we take action a
    - Sum of all possible s' must equal 1 in the Transition Function
    - Reward Function R[s,a]: If we are in state s and we take action a, we get a reward
    - Goal is to Find: Optimal Policy π*(s) to a that will maximize r
      - The Algorithm that leads to π* if you know T and R: policty iteration or value iteration
- [ ] Unknown Transition and Rewards: must interact with the world, observe and build policy in unknown areas
- [ ] Experience Tuple <s, a, s', r> : When you experience s, and take action a, you end up in state s' and get reward r
    - s' becomes the s of the next experience tuple and it continues down until you have a complete collection of experience tuples
- [ ] Model Based RL uses Policy and/or Value Interation: Build models T[s, a, s'] and R[s,a] based on previous experience tuples - a tabular representation
- [ ] Model Free Method uses Q-Learning: Develop policy just directly by looking at data
- [ ] What to Optimize?
    - Infinite Horizon: sum of all rewards from (1, infinity)
    - Finite Horizon: limited "moves" or time rewards from (1, n) 
    - Discounted Reward: Summation of Gamma * Reward
      - Gamma is similar to interest rate. Gamma: as it increases, reward decreases a little bit (0 < Gamma <= 1.0) The closer to 1, the more value to the rewards in the future 
## 03-06 Q-Learning
- [ ] Model Tree Approach (Doesn't konw or use T[] or R[])
- [ ] Builds table of utility values as agent interacts with the world
- [ ] What is Q? Value of taking action a in state s
- [ ] Q Function Q[s,a] = immediate reward + discounted reward (reward for future action)
- [ ] How to Use Q
      - Policy: What action do we take in state s?
      - π*(s) - Q*[s,a]
- [ ] Q Learner Procedure
    - Big Picture:
    - Select Training Data from older time series
    - Iterate over time and collect experience tuples throughout training data <s, a, s', r>
      - Details of Iteration:
      - Set start time, init Q[]
      - Compute s
      - Select a
      - Observe r, s' and update Q table
    - Test your policy π - how well it performs in back testing
    - Repeat until Converge / Policy does not get better with more interations
- [ ] Update Rule: Q'[s,a] = (1 - 	α ) * Q[s,a] + (	α * improved estimate / 	α * (r + λ * later reward)
    - Alpha 	α : Learning Rate [0, 1.0] - usually .02. Large alpha means learning more quickly
    - Lambda λ : Discount Rate [0, 1.0] Low value: values immediate rewards, High Value: value later rewards more (A reward 20 setps in the fugure is owrth the same as reward right now)
- [ ] New Value of Q =  Old value of Q + immediate reward + Future Discounted Rewards for Future Actions
- [ ] Last Part of Equation asks: What is the value of those future rewards if we reach s' and act appropriately?
- [ ] Two Finer Points:
    - Success Depends on Exporation
    - Choose Random Action w Probability c (.3)
    - Over iterations, make probability c smaller until we don't choose randomly anymore
- [ ] Two Possible "Flips of the Coin" during each iteratoin:
    - First Flip determines if we will use randomness or go off highest Q-value
    - Section Flip: if choosing randomly, determines random action to select. Random Actions force system to try different actions and states, which improves exploration 
## 03-07 Dyna
- [ ] 


# NOTES FROM READING:  

## What Hedge Funds Really Do Ch. 9
- [ ] 
## Machine Learning – Chapter 13 - Deep Learning
- [ ] The cornerstone of deep learning is the neural network
- [ ] Neural networks resurfaced after 2010 with the new name deep learning, with new architectures,
additional bells and whistles, and a string
of success stories on some niche problems such as image and video classification, speech and text modeling.
#### 10.1 Single Layer Neural Networks
- [ ] A neural network takes an input vector of p variables X = (X1, X2, . . . , Xp)
and builds a nonlinear function f(X) to predict the response Y .
- [ ] feed-forward neural network for modeling a quantitative response
using predictors. In the terminology of neural networks, the four features X1, . . . , X4 make up the units in the input layer. Each of the inputs from the input layer feeds into each of the K hidden input layer units (we get to pick K). 
- [ ] The preferred choice in modern neural networks is the ReLU (rectified linear unit) activation function
- [ ] So in words, the model depicted in Figure 10.1 derives five new features
by computing five different linear combinations of X, and then squashes
each through an activation function g(·) to transform it. The final model
is linear in these derived variables.
The name neural network originally derived from thinking of these hidden
units as analogous to neurons in the brain — values of the activations
Ak = hk(X) close to one are firing, w
#### 10.2 Mutlilayer Neural Networks
- [ ] Modern neural networks typically have more than one hidden layer, and
often many units per layer. In theory a single hidden layer with a large
number of units has the ability to approximate most functions. However,
the learning task of discovering a good solution is made much easier with
multiple layers each of modest size.
- [ ]  Example is number classification
- [ ]  More generally, in multi-task learning one can predict different responses simultaneously with a single network; they
all have a say in the formation of the hidden layers.
#### Convolutional Neural Networks
- [ ]  Image Classification
- [ ]  A special family of convolutional neural networks (CNNs) has evolved for
classifying images such as these, and has shown spectacular success on a
wide range of problems. CNNs mimic to some degree how humans classify
images, by recognizing specific features or patterns anywhere in the image
that distinguish each particular object class.
- [ ] The network first identifies low-level features in the input image, such
as small edges, patches of color, and the like. These low-level features are
then combined to form higher-level features, such as parts of ears, eyes,
and so on. Eventually, the presence or absence of these higher-level features
contributes to the probability of any given output class.
- [ ]  How does a convolutional neural network build up this hierarchy? It combines two specialized types of hidden layers, called convolution layers and
pooling layers. Convolution layers search for instances of small patterns in
the image, whereas pooling layers downsample these to select a prominent
subset. In order to achieve state-of-the-art results, contemporary neuralnetwork architectures make use of many convolution and pooling layers.
We describe convolution and pooling layers next.
- [ ] Convolution Layers: made up of a large number of convolution filters, each of which is a template that determines whether a particular local feature is
present in an image. A convolution filter relies on a very simple operation,
called a convolution, which basically amounts to repeatedly multiplying
matrix elements and then adding the results.
- [ ] Convolution filters find local features in an image, such as edges
and small shapes.
- [ ] We typically apply the ReLU activation function (10.5) to the convolved image. This step is sometimes viewed as a separate layer in
the convolutional neural network, in which case it is referred to as a
detector layer
#### Pooling Layers
- [ ] A pooling layer provides a way to condense a large image into a smaller
summary image. While there are a number of possible ways to perform
pooling, the max pooling operation summarizes each non-overlapping 2 × 2
block of pixels in an image using the maximum value in the block. This
reduces the size of the image by a factor of two in each direction, and it
also provides some location invariance: i.e. as long as there is a large value
in one of the four pixels in the block, the whole block registers as a large
value in the reduced image.
#### Architecture of a Convolutional Neural Network
- [ ] The number of convolution filters in a
convolution layer is akin to the number of units at a particular hidden layer
in a fully-connected neural network
- [ ] This number also defines the number of channels in the resulting threedimensional feature map. We have also described a pooling layer, which
reduces the first two dimensions of each three-dimensional feature map.
- [ ] After this first round of convolutions, we
now have a new “image”; a feature map with considerably more channels
than the three color input channels (six in the figure, since we used six
convolution filters).
- [ ] Each subsequent convolve layer is similar to the first. It takes as input
the three-dimensional feature map from the previous layer and treats
it like a single multi-channel image. Each convolution filter learned
has as many channels as this feature map
- [ ] Since the channel feature maps are reduced in size after each pool
layer, we usually increase the number of filters in the next convolve
layer to compensate.
- [ ] Sometimes we repeat several convolve layers before a pool layer. This
effectively increases the dimension of the filter.
- [ ] These operations are repeated until the pooling has reduced each channel
feature map down to just a few pixels in each dimension. At this point the
three-dimensional feature maps are flattened — the pixels are treated as
separate units — and fed into one or more fully-connected layers before
reaching the output layer, which is a softmax activation for the 100 classes
#### Data Augmentation
- [ ] An additional important trick used with image modeling is data augmentation. Essentially, each training image is replicated many times, with each replicate randomly distorted in a natural way such that human recognition
is unaffected. 
- [ ] At face value this is a way of increasing the training set
considerably with somewhat different examples, and thus protects against
overfitting. In fact we can see this as a form of regularization: we build a
cloud of images around each original image, all with the same label. This
kind of fattening of the data is similar in spirit to ridge regularization.

- [ ]
- [ ]
- [ ]
- [ ]
- [ ]
- [ ]
- [ ]
- [ ]
- [ ]
- [ ]
## Introduction to Statistical Learning (2nd Edition) Chapter 10 Deep Learning – 
- [ ] 

## Handbook of AI and Big Data Applications in Investments - Chapter 8 
- [ ] 
## Handbook of AI and Big Data Applications in Investments - Chapters 10 and 11
- [ ] 
## Keynote on Algorithmic Bias (Drs. Isbell and Littman)
- [ ] 
## Interview with Tammer Kamel (Ed Lessons)
- [ ] 

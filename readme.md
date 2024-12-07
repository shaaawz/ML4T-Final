
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
- [ ] Dyna is a reinforcement learning architecture that combines model-based and model-free methods, where learning is done using both real experience and simulated experiences.
- [ ] The core idea is to use the agent's learned model of the environment to generate additional synthetic experiences (hallucinated experiences), which are then used to update the value function or Q-table.
- [ ] Big Picture: learn T & R:
- [ ] T[s,a,s'] - Transition model, which estimates the probability of reaching state s' from state s by taking action a
- [ ] R[s,a] - Reward model, which estimates the expected reward when taking action a in state s.
- [ ] The model can be learned using the agent's real experience or be pre-defined
- [ ] Hallucinate experience and update Q table based on hallucinations
- [ ] Hallucinated experience: Generated by using the model to simulate state transitions. These experiences are typically created using the current model of the environment (T[s, a, s']) and reward function (R[s, a]).
- [ ] s = random, a = random, s' = infer from T[] using argmax, r = R[s,a]


# NOTES FROM READING:  

## What Hedge Funds Really Do Ch. 9
- [ ] Buffett invests strongly in a small number of companies: As of September 2010, 54 percent of BRK-A’s holdings were in just three stocks: Coca Cola (KO), American Express (AXP), and Wells Fargo (WFC). Ninety percent of their holdings were in just 12 stocks.
- [ ] In the 1980s, Richard Grinold introduced what he calls the Fundamental Law of Active Portfolio Management. It is described nicely in his book with Ronald Kahn, Active Portfolio Management . For the moment we will offer a simplified version of this law. We paraphrase it as follows:
- [ ] performance = skill * √breadth
- [ ] Skill is a measure of how well a manager transforms information about an equity into an accurate prediction of future return, and breadth represents the number of investment decisions (trades) the manager makes each year. This law suggests that as a manager’s skill increases, returns increase. That is not surprising. What is interesting and perhaps surprising is that to double performance at the same level of skill, a manager must find four times as many opportunities to trade.
- [ ] There are two ways to increase the breadth of a portfolio: We can choose to hold more equities at once, or we can turn over those assets more frequently. The first approach is more suitable for a conventional managed portfolio approach to investing, while the second relates more to an active trading strategy
- [ ] Two Categories of Risk:
    - [ ] Systematic risk is that risk undertaken by exposer to any asset in the asset class. example: Market Falls
    - [ ] Specific risk i sthe risk associated with a particular asset. Example: Oil company stock falls because its new oil field fails - despite other occurances in the stock market
- [ ] Diversification mutes specific risk. Volatility of a portoflio declines as more individaul stocks are included into the portfolio
- [ ] Yet Experts argue that diversification beyond 20 to 40 separate issues provides little additional risk reduction.
- [ ] There is a trade-off, however. The more breadth in an investor’s portfolio, the less expertise they can apply to each of its contents. Since alpha is assumed to stem from research and knowledge— that is, from investment-specific information— the broader the portfolio, the less alpha can be generated
- [ ] Another approach to adding breadth to a portfolio or strategy is through more trading opportunities.
- [ ] Coin Flip example: If heads was 51% and you had $1000 to bet on a coin flip, would you bet $1000 in one flip or $1 on 1000 flips? The expected the return is $20 on both, but the risk is very different.
- [ ] First, for the single bet option, there is a 49 percent chance that you will lose your entire $1,000 stake. For the multiple bet option, the probability of a total loss— the coin coming up tails each of 1,000 times— is 0.49 ^ 1000, which is infinitesimal (too small to be displayed on our spreadsheet, even to 23 significant digits).
- [ ]  So, for this measure, risk is substantially lower for the 1000-bet scenario. But we can also use standard deviation— a measure of the range of plausible returns— as a measure of risk. For a single $1,000 bet, the standard deviation is $31.62. For the 1,000 $1 bets, the standard deviation is $1. So for this measure, we also see significantly lower the risk for the 1000 bet case. In finance, we often compare strategies in terms of risk adjusted return, which is to say return divided by the risk. So the return to risk ratios of the two options are as follows:
- [ ]  Single bet: $20 / $31.62 risk = 0.6325 Return/Risk Ratio (Similar to Sharpe Ratio)
- [ ]  Thousand Bets: $20 / $1 risk = 20.0 Return/Risk Ratio
- [ ]  Information Ratio: Similar to Sharpe Ratio - It adjusts return for risk by dividing them
- [ ]  IR = Excess return per period / standard deviation of excess return per period
- [ ]  Excess Return seeks to measure return that is due to the investor's skill rather than return due to the market, the alpha component or residual
- [ ]  Total Return has two risk components: Market Risk (captured by standard deviation of beta x return)
- [ ]  Investor skill risk (Capured by the stdv of alpha)
- [ ]  Hedging investing strategies seek to minimize or eliminate the market risk, leaving a clear field to exploit the investor’s alpha. A summary measure of skill is the information ratio of an investor’s alpha, or: IR (alpha) = Mean (alpha)/Standard deviation (alpha).
- [ ]  Information Coefficient: correlation of a manager's predictions about asset prices with their actual future prices. Perfect predictor would have IC of 1.0, Perfectly wrong predictor would be -1.0
- [ ]  Breadth - Number of trading opportunities presented over time
- [ ]  Fundemental Law in Full: IR = IC * √breadth
- [ ]  An increment of added skill, reflected in the information coefficient, has a greater reflect on risk-adjusted return (the information ratio) than does an equal increment in portfolio breadth. This is because the portfolio breadth affects the IR as a square root, whereas IC affects IR proportionally. So for an investor like Buffett, whose IC is probably among the highest on the planet, he is absolutely correct— for him— to disparage diversification.
## Machine Learning – Chapter 13 - Reinforcement Learning
- [ ] Reinforcement learning addresses the question of how an autonomous agent that
senses and acts in its environment can learn to choose optimal actions to achieve its
goals. Each time the agent performs an action in its environment, a trainer may provide a
reward or penalty to indicate the desirability of the resulting state. The
task of the agent is to learn from this indirect, delayed reward, to choose sequences
of actions that produce the greatest cumulative reward. This chapter focuses on
an algorithm called Q learning that can acquire optimal control strategies from
delayed rewards, even when the agent has no prior knowledge of the effects of
its actions on the environment. 
- [ ] The robot, or agent, has a set of sensors to
observe the state of its environment, and a set of actions it can perform to alter
this state. Its task is to learn a control
strategy, or policy, for choosing actions that achieve its goals.
- [ ] This reward function may be built into the robot, or known only to an
external teacher who provides the reward value for each action performed by the
robot. The task of the robot is to perform sequences of actions, observe their consequences, and learn a control policy. The control policy we desire is one that, from
any initial state, chooses actions that maximize the reward accumulated over time
by the agent
- [ ] An agent interacting with its environment.
The agent exists in an environment described
by some set of possible states S. It can
perform any of a set of possible actions
A. Each time it performs an action a, in
some state st the agent receives a real-valued
reward r, that indicates the immediate value
of this state-action transition. This produces
a sequence of states si, actions ai, and
immediate rewards ri as shown in the figure.
The agent's task is to learn a control policy,
n : S + A, that maximizes the expected
sum of these rewards, with future rewards
discounted exponentially by their delay.
- [ ]  This reinforcement learning problem differs from other function
approximation tasks in several important respects:
    -  Delayed reward: In
reinforcement learning, however, training information is not available in this
form. Instead, the trainer provides only a sequence of immediate reward values as the agent executes its sequence of actions. The agent, therefore, faces
the problem of temporal credit assignment: determining which of the actions
in its sequence are to be credited with producing the eventual rewards.
    - Exploration: In reinforcement learning, the agent influences the distribution
of training examples by the action sequence it chooses. This raises the question of which experimentation strategy produces most effective learning. The
learner faces a tradeoff in choosing whether to favor exploration of unknown
states and actions (to gather new information), or exploitation of states and
actions that it has already learned will yield high reward (to maximize its
cumulative reward).
    -  Partially observable states: Although it is convenient to assume that the
agent's sensors can perceive the entire state of the environment at each time
step, in many practical situations sensors provide only partial information.
    - Life-long learning: Unlike isolated function approximation tasks, robot learning often requires that the robot learn several related tasks within the same
environment, using the same sensors
- [ ] In a Markov decision process (MDP) the agent can perceive a set S of distinct
states of its environment and has a set A of actions that it can perform.At each
discrete time step t, the agent senses the current state st, chooses a current action
a,, and performs it. The environment responds by giving the agent a reward r =
r (s, a,) and by producing the succeeding state s(t+l) = ϴ(s(t), a()).
Here the functions ϴ  and r are part of the environment and are not necessarily known to the agent. 
- [ ] In an MDP, the functions  ϴ (st, a,) and r(s,, a,) depend only on the current state
and action, and not on earlier states or actions. 
- [ ] The task of the agent is to learn a policy, π : S -> A, for selecting its next
action a, based on the current observed state s; that is, π(s) = a,
- [ ] Discounted cumulative reward (V) achieved by policy π from initial state s. It is reasonable to
discount future rewards relative to immediate rewards because, in many cases,
we prefer to obtain the reward sooner rather than later. 
- [ ] We are now in a position to state precisely the agent's learning task. We
require that the agent learn a policy n that maximizes V"(s) for all states s.
We will call such a policy an optimal policy and denote it by  π*.
optimal policy  π* =   argmax V" (s), (Vs) 
- [ ] Note
the immediate reward in this particular environment is defined to be zero for
all state-action transitions except for those leading into the state labeled G. It is
convenient to think of the state G as the goal state, because the only way the agent
can receive reward, in this case, is by entering this state. Note in this particular
environment, the only action available to the agent once it enters the state G is
to remain in this state. For this reason, we call G an absorbing state. 
- [ ] What evaluation function should the agent attempt to learn? One obvious
choice is V*. The agent should prefer state sl over state s2 whenever V*(sl) >
V*(s2), because the cumulative future reward will be greater from sl. Of course
the agent's policy must choose among actions, not among states. However, it can
use V* in certain settings to choose among actions as well. The optimal action
in state s is the action a that maximizes the sum of the immediate reward r(s, a)
plus the value V* of the immediate successor state, discounted by y. 

- [ ] Equation (13.3): n*(s) = argmax[r(s, a) f y V*(G(s, a))] 
- [ ] Thus, the agent can acquire the optimal policy by learning V*, provided it has
perfect knowledge of the immediate reward function r and the state transition
function 6. When the agent knows the functions r and 6 used by the environment
to respond to its actions, it can then use Equation (13.3) to calculate the optimal
action for any state s.
- [ ] cases where either
ϴ or r is unknown, learning V* is unfortunately of no use for selecting optimal
actions because the agent cannot evaluate Equation (13.3). What evaluation function should the agent use in this more general setting? The evaluation function Q,
defined in the following section, provides one answer. 
- [ ]  In other words, the value of Q is the reward received
immediately upon executing action a from state s, plus the value (discounted by
y) of following the optimal policy thereafter.   π*(s) = argmax Q (s , a) 
- [ ] Why is this rewrite important? Because it shows that if the agent learns the Q
function instead of the V* function, it will be able to select optimal actions even
when it has no knowledge of thefunctions r and ϴ. it need only consider each available action a in its current state s and choose the action that maximizes Q(s, a). 
- [ ] Part of
the beauty of Q learning is that the evaluation function is defined to have precisely
this property-the value of Q for the current state and action summarizes in a
single number all the information needed to determine the discounted cumulative
reward that will be gained in the future if action a is selected in state s.
To illustrate, Figure 13.2 shows the Q values for every state and action in the
simple grid world. Notice that the Q value for each state-action transition equals
the r value for this transition plus the V* value for the resulting state discounted by
y. Note also that the optimal policy shown in the figure corresponds to selecting
actions with maximal Q values.
- [ ] The key problem is finding a reliable way to estimate training values for
Q, given only a sequence of immediate rewards r spread out over time. This can
be accomplished through iterative approximation.
- [ ] which allows rewriting Equation (13.4) as
Q(s, a) = r(s, a) + alpha max Q(theta(s, a), a')
- [ ] In this algorithm the learner represents its hypothesis Q by a large table
with a separate entry for each state-action pair. The table entry for the pair (s, a)
stores the value for ~(s, a)-the learner's current hypothesis about the actual
but unknown value Q(s, a). The table can be initially filled with random values
(though it is easier to understand the algorithm if one assumes initial values of
zero). The agent repeatedly observes its current state s, chooses some action a,
executes this action, then observes the resulting reward r = r(s, a) and the new
state s' = 6(s, a). It then updates the table entry for ~(s, a) following each such
transition, according to the rule: Note this training rule uses the agent's current Q values for the new state
s' to refine its estimate of ~(s, a) for the previous state s. 
- [ ] Each time the agent moves forward from an old state to a new one, Q
learning propagates Q estimates backward from the new state to the old. At the
same time, the immediate reward received by the agent for the transition is used
to augment these propagated values of Q.
- [ ] How will the values of Q evolve as the Q learning algorithm is applied in
this case? With all the Q values initialized to zero, the agent will make no changes
to any Q table entry until it happens to reach the goal state and receive a nonzero
reward. This will result in refining the Q value for the single transition leading
into the goal state. On the next episode, if the agent passes through this state
adjacent to the goal state, its nonzero Q value will allow refining the value for
some transition two steps from the goal, and so on. Given a sufficient number of
training episodes, the information will propagate from the transitions with nonzero
reward back through the entire state-action space available to the agent, resulting
eventually in a Q table containing the Q values shown

## Introduction to Statistical Learning (2nd Edition) Chapter 10 Deep Learning 
- [ ] - [ ] The cornerstone of deep learning is the neural network
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
#### Document Classification
- [ ] A new type of example that has important
applications in industry and science: predicting attributes of documents.
Examples of documents include articles in medical journals, Reuters news
feeds, emails, tweets, and so on
- [ ] Example: IMBD Reviews - Each review can be a different length, include slang or non-words, have
spelling errors, etc. We need to find a way to featurize such a document. This is modern parlance for defining a set of predictors.
- [ ] The simplest and most common featurization is the bag-of-words model. We score each document for the presence or absence of each of the words in
a language dictionary — in this case an English dictionary. If the dictionary
contains M words, that means for each document we create a binary feature
vector of length M, and score a 1 for every word present, and 0 otherwise
- [ ] The bag-of-words model summarizes a document by the words present,
and ignores their context. There are at least two popular ways to take the
context into account
  - [ ]  The bag-of-n-grams model. For example, a bag of 2-grams records the consecutive co-occurrence of every distinct pair of words. “Blissfully long” can be seen as a positive phrase in a movie review, while
“blissfully short” a negative.
- [ ] Treat the document as a sequence, taking account of all the words in
the context of those that preceded and those that follow
#### Recurrent Neural Networks
- [ ] In a recurrent neural network (RNN), the input object X is a sequence.
- [ ] The order of
the words, and closeness of certain words in a sentence, convey semantic
meaning. RNNs are designed to accommodate and take advantage of the
sequential nature of such input objects, much like convolutional neural networks accommodate the spatial structure of image inputs. The output Y
can also be a sequence (such as in language translation), but often is a
scalar, like the binary sentiment label of a movie review document.
- [ ] There are many variations and enhancements of the simple RNN we
used for sequence modeling. One approach we did not discuss uses a onedimensional convolutional neural network, treating the sequence of vectors
(say words, as represented in the embedding space) as an image. The convolution filter slides along the sequence in a one-dimensional fashion, with
the potential to learn particular phrases or short subsequences relevant to
the learning task.
#### When To use Deep Learning
- [ ] The performance of deep learning in this chapter has been rather impressive. It nailed the digit classification problem, and deep CNNs have really
revolutionized image classification. We see daily reports of new success stories for deep learning. Many of these are related to image classification
tasks, such as machine diagnosis of mammograms or digital X-ray images,
ophthalmology eye scans, annotations of MRI scans, and so on. Likewise
there are numerous successes of RNNs in speech and language translation,
forecasting, and document modeling. The question that then begs an answer is: should we discard all our older tools, and use deep learning on every
problem with data?
- [ ] We see similar performance for all three
models. We report the mean absolute error on the test data, as well as
the test R2 for each method, which are all respectable (see Exercise 5).
We spent a fair bit of time fiddling with the configuration parameters of
the neural network to achieve these results. It is possible that if we were to
spend more time, and got the form and amount of regularization just right,
that we might be able to match or even outperform linear regression and
the lasso. But with great ease we obtained linear models that work well.
Linear models are much easier to present and understand than the neural
network, which is essentially a black box. The lasso selected 12 of the 19
variables in making its prediction. So in cases like this we are much better
off following the Occam’s razor principle: when faced with several methods
that give roughly equivalent performance, pick the simplest


## Handbook of AI and Big Data Applications in Investments - Chapter 8 
- [ ] The larger
a trade is, the more impact it tends to have on market prices.
This impact can be measured as slippage (i.e., the difference
between a reference price before the start of the trade and
the prices at which trades are executed). To minimize this
slippage cost, which can lead to a significant performance
degradation over time, machine learning (ML) methods can
be deployed to improve execution algorithms in various ways.
- [ ] To minimize the adverse impact
of trades on the market price, this large order is split into
smaller slices that are then executed over the available time
horizon. The role of the algorithm is to choose an execution
schedule that reduces the slippage bill as much as possible. If the order was not split up this way and distributed
over time but instead was executed as a market order at the
moment the trade instruction was presented, then large buy
orders would push prices up, sell orders would push them
down, or there might simply not be enough liquidity in the
market at the time to complete the trade, leading to a less
favorable slippage cost or an incomplete execution.
- [ ] TWAP (time-weighted average price) execution, which
serves as a benchmark for more advanced execution algorithms to surpass.
- [ ] Another commonly used execution strategy makes use of information on trading volume,
or market turnover, because a higher volume allows larger
trades to be executed for the same amount of price impact.
This type of strategy targets execution of a block of shares
at the volume-weighted average price (VWAP) by splitting up
execution over time proportionately to the expected volume
profile.
- [ ] One role for ML algorithms in execution is therefore to
compute forecasts of short-term price movements and
expected volume that an execution algorithm can use
to front- or back-load the execution schedule—in effect,
locally speeding up or slowing down trading activity.
- [ ] Most electronic exchanges involved in the trading of cash
equities, futures, or options use limit order books (LOBs) to
match buy and sell orders using a price–time priority matching mechanism. In this prioritization scheme, orders are first
matched by their price levels, with orders at the same price
on a first-come, first-served basis. Every limit order to buy
with a price lower than any sell order in the LOB is added to
the bid side of the LOB, where levels are ordered from best
(highest price and earliest arrival) to worst (lowest price and
latest arrival).
- [ ] The bid level with
the highest price is called the best bid, and the ask level with the lowest price is called the best ask. The difference
in price between the best ask and best bid is called the
spread. If a market order is placed to buy (sell), it is first executed against the best price level of the ask (bid) side, then
executed in order of arrival time of the corresponding limit
order, and finally executed against the next level if the order
is larger than the number of shares available at the best
price
- [ ] Given this order matching mechanism, LOBs are often
described as double-sided continuous auctions, since a
continuous order flow changes the book dynamically over
time
- [ ] Over the short term, one of the best indicators of immediate
price moves in the LOB is the order flow imbalance (Cont,
Kukanov, and Stoikov 2014). The definition of the order flow
imbalance (OFI) is the order flow on the buy side: incoming
limit buy orders at the best bid, Lb, net of order cancellations, Cb, and market orders, Mb, minus the opposing sellside flow, within a period of time: . OFI = (Lb - Cb - Ms) - (Ls - Cs - Mb)
- [ ] This measure captures an imbalance between demand and
supply in the market, which is an essential determinant
of the market price. A high positive order flow imbalance
indicates excessive buy pressure at current prices, thereby
making it more likely that prices will rise imminently. 
- [ ] order book imbalance (OBI): OBI = (d^b - d^s) / (d^b + d^s)
- [ ] Here, db and ds
 are the depths of the best bid and best
ask, respectively. The OBI calculates the
normalized difference between the depth (number of
available shares) on the buy side at the best bid, db, and
the number of shares posted on the sell side at the best
ask, ds
- [ ] A classical approach in statistical modeling is to start out
with simple, perhaps linear, models and a small set of variables, or features, that are likely to have some predictive
power for the quantity of interest. Over the modeling process, model complexity is gradually increased and features
are further engineered and refined to extract more information from the raw data.
- [ ] ML literature by deep learning and artificial neural network
(ANN) models, most recent approaches, however, have
moved away from handcrafted feature engineering and
instead approached prediction problems using raw data
directly. This trend has also taken hold in financial ML and
quantitative trading.
- [ ] Deep Order FLow Imbalance model: only extracted order flow features from the top 10 levels of the order books and were able to show great performance. This implies
that practitioners might be able to get away with simpler
models in some cases by performing an input data transformation from raw data to order flows. These results contrast with those of the same simple neural network models
that instead use raw order book features, which cannot
achieve any predictive power on the same task, implying
that the data transformation is essential. 
- [ ] DeepLOB - CCN deep learning model, This type of model currently provides the most
accurate short-term price signals, which can be used to
improve execution trajectories. To improve the robustness
of forecasts, DeepLOB can also be extended to perform
quantile regression on the forward return distribution 
which uses a deep neural network architecture with convolutional layers and an inception module
- [ ] Convolutional neural networks (CNNs) were originally developed for visual classification tasks, such as handwritten
digit recognition or classifying images based on their content.
- [ ] Convolutional layers act as
local filters on an image, aggregating local information in
every special region. During learning, the weights of many
such filters are updated as the overall system learns to
recognize distinct features in the data, such as horizontal or
vertical lines, corners, and regions of similar contrast.
- [ ] To translate written text
from one language to another, the idea of the sequence-tosequence model (Sutskever, Vinyals, and Le 2014) is to use
an LSTM encoder to learn a representation of a sentence as
a fixed-length vector and then use a separate LSTM-based
decoder to again translate this vector representation into
the target language.
- [ ] Adapting this idea to predict return
trajectories, the model in Zhang and Zohren (2021) uses
the DeepLOB network (Zhang et al. 2019a) as an encoder,
while an attention mechanism (Vaswani, Shazeer, Parmar,
Uszkoreit, Jones, Gomez, Kaiser, and Polosukhin 2017)
allows using selected hidden states of the encoder layers
in the decoding step, producing the forecast time series.
- [ ] Using a combination of ML models, we can thus engineer
a complete execution strategy. An example algorithm
might work as follows. Volatility forecasts, obtained using
any method from historical averaging over generalized
autoregressive conditional heteroskedasticity (GARCH)
models to deep learning, can be used to schedule either
the time allowed for an execution ticket or the amount of
front-loading over a fixed time horizon by controlling how
execution slice sizes decrease over time. Using ML forecasts of trade volume, we can further modulate execution
schedules by proportionately changing future slice sizes
with expected volume forecasts. Predicted price paths can
then be used to fine-tune the strategy by varying placement levels (prices) dynamically. Should we expect a favorable price move, we would place a passive limit order in the
book at the first level—or even deeper into the book if th eexpected price move is sufficiently large.
- [ ] reinforcement learning (RL) algorithms to plan
the execution trajectory. RL—and especially deep RL using
deep neural networks—has been tremendously successful
- [ ] On a theoretical level, the execution problem can be
framed as a partially observable Markov decision process
(POMDP), which can be amenable to being solved using RL
algorithms. The RL learning paradigm works analogously
to biological learning processes in animals and humans. 
- [ ] The agent, our execution algorithm, interacts with a market
environment in state s, which describes all information
necessary to characterize the current state of the market,
by performing an action, a, thereby transitioning the state
to s' = T(s,a). The environment state is further assumed to satisfy the Markov property, which means that past states
do not add any further relevant information for the future.
The agent, however, perceives not the entire state of the
world but only an observation, o’ = obs(s’), and hence does
usually not know the underlying state exactly. In addition
to the new observation o’ at each step, the learner also
receives a reward signal, r. Based on the reward signal, the
RL algorithm learns over time which sequence of actions
leads to the highest expected cumulative rewards.
- [ ] The most basic kind of “simulator” simply uses
historical market prices. This approach limits the action
space to timing market orders, because past prices alone
cannot determine whether a limit order would have been
executed or not. Another shortcoming of relying solely on
historical prices for simulation is that trades do not generate any market impact, because neither do they take away
liquidity in the book nor can they cause any other market
participant to react to the trade. To alleviate the latter problem, simulation environments are sometimes enhanced
with stochastic models of price impact to represent more
realistic costs of aggressive trading. 
- [ ] However, this alone does not solve the problem of counterfactual behavior by other agents. For example, if one of
our orders is executed, it might imply that someone else’s
order was not executed. They then might have placed
another order at a different price; however, this is not represented in the historical data. One approach to handle these
counterfactual scenarios is agent-based modeling, which
represents individual traders explicitly in a simulation.
These simulated agents follow their own trading strategies
and can react to changes in the market caused by our
execution algorithm, as well as to actions and reactions of
other agents. Capturing realistic trading behavior remains
a challenging task, and building realistic LOB models is the
subject of active research.

## Handbook of AI and Big Data Applications in Investments - Chapters 10 and 11
### Chapter 10: ACCELERATED AI AND USE CASES IN INVESTMENT MANAGEMENT
- [ ] Investment professionals are increasingly integrating AI and big data into their investment processes, using advanced technologies like AI "factories" and simulation platforms. This trend is driven by the potential for AI to generate alpha (investment returns), improve risk management, enhance client access, and provide customization opportunities. Embracing these technologies has become a key differentiator in the industry.
- [ ] Massive amounts of data, such as text-based or satellite imagery data, need to be collected, cleaned, streamed, and analyzed—tasks that machines can perform very efficiently. AI technologies help process this data, providing transparency and traceability, which are crucial for human–machine interaction and validation. This enables data narratives and supports subject matter experts. Manual analysis of such vast data is not feasible.
- [ ] The investment firms of the future will succeed by strategically integrating AI and big data into their processes. Key elements for an effective AI strategy include:

    - Infrastructure: Developing appropriate infrastructure for AI training, simulation, risk management, algorithmic trading, and backtesting.
    - Scalable Workflow: Creating a scalable process for developing and deploying AI models across the enterprise.
    - Robust AI Models: Building models that are not only powerful but also explainable and verifiable, increasing confidence and adoption of AI technologies.
- [ ] A crucial technology to implement these elements is accelerated computing, which enhances simulation, data manipulation, AI model building, and deployment. This technology boosts productivity, return on investment (ROI), model quality, and scalability while reducing costs, time to insight, energy consumption, and infrastructure complexity.
- [ ] The need for accelerated computing becomes evident when combining and analyzing multiple data sources, such as remote sensors, IoT devices, social media, and satellite data. Accelerated computing technologies enable faster aggregation, correlation, analysis, and visualization of these data sources by significantly reducing analytical latency, often to milliseconds.
- [ ] New machine learning (ML) forecasting techniques that leverage complex, unstructured datasets (e.g., satellite images) and generative methods for synthetic data are transforming strategy development and backtesting.
- [ ] Investors are increasingly focused on real-time insights into company performance, sustainability, and environmental risk factors. To effectively analyze these large, complex datasets, traditional CPU-based servers may struggle, as they are slower and more energy-intensive than graphics processing units (GPUs), which are better suited for high-speed, multilayered, and multimodal analysis. This makes GPUs crucial for efficiently processing vast amounts of data.
- [ ] GPUs can be found in many compute clusters, ranging from
supercomputing to public clouds and even enterprise data
centers. They can accelerate the processing of huge amounts
of structured and unstructured large datasets and execute
large training and inferencing workloads. These ultra-fast
processors are designed for massively accelerated training
epochs, queries, complex image rendering, and interactive visualization. Combined with purpose-built analytics
software, they deliver the kind of speed and zero-latency
interactivity that professional investors need.
- [ ] To summarize, accelerated computing can build more
model alternatives, with potentially higher accuracy and
at the same time at lower cost and energy consumption and with greater flexibility. Such approaches as “fail
fast, fail forward” can be implemented with accelerated computing, which can be viewed as a kind of “time
machine” by speeding up the iterations required for model
development.
- [ ] Natural Language Processing (NLP) is a key AI technique used to process text for tasks like named entity recognition, sentiment analysis, language translation, and text summarization. In the financial and ESG (Environmental, Social, and Governance) space, NLP is applied to extract and digitize relevant information from complex documents, such as financial news, ESG reports, and disclosures, which often include images, tables, and varied formats.
- [ ] An ESG and risk data repository can centralize this information, making it easier to operationalize and adapt to changing regulatory requirements.
- [ ] A real-time ESG analytics process incorporates news and media screening to monitor adverse events and help investors stay on top of ESG reporting and assessment. Advanced NLP technologies can analyze unstructured data from news and social media, offering insights that, when combined with ESG scores from rating agencies, provide a comprehensive view for more informed decision-making.
- [ ] Earth Observation (EO) involves gathering data about the Earth's physical, chemical, and biological systems using remote sensing technologies, primarily via satellites with imaging devices. This data provides reliable, repeatable insights into environmental conditions and changes. When combined with machine learning (ML), EO has the potential to revolutionize information availability within financial systems.
- [ ] Spatial finance refers to the integration of geospatial data and analysis into financial services, enabling better management of climate-related and environmental risks such as biodiversity loss, water quality threats, and deforestation. According to the EU Space Programme (EUSPA), by 2031, the insurance and finance sector will become the largest contributor to global EO revenues, with an estimated EUR 994 million and an 18.2% market share.
- [ ] Modern portfolio theory (MPT) and portfolio diversification often face practical issues, such as backtest overfitting and reliance on noisy covariance estimates for optimization. To address these, alternative approaches aligned with the Monte Carlo backtesting paradigm are recommended, which could help mitigate the replication crisis. One such approach is the use of synthetic datasets to develop investment algorithms, similar to how synthetic data is used for training autonomous machines like robots or self-driving cars. This technique helps investors navigate unknown data-generating processes and test AI-driven strategies.
- [ ] While significant progress has been made in generating synthetic asset return data with realistic characteristics, less attention has been paid to generating correlated returns that align with empirical covariance matrices. This is crucial for pricing and managing risks of correlation-dependent financial instruments. Techniques like Generative Adversarial Networks (GANs) and evolutionary multi-objective optimization (e.g., "matrix evolutions") are being used to generate realistic correlation scenarios for financial modeling. These methods can create millions of unique, yet realistic, correlation matrices that have never been observed, allowing for robust testing of investment portfolios.
### Chapter 11: SYMBOLIC AI: A CASE STUDY
- [ ] Samuel is a composite AI14 system that collaborates with
humans. It acts as a digital colleague that guides the
human investment team with transparent, systematized,
and well-substantiated advice. Its decisions are transparently substantiated and can be tracked down to each
datapoint used, allowing for efficient reconciliation with the
thought process of the human team.
- [ ] Symbolic AI, or
classical AI, is a collection of techniques that are based
on human-readable and high-level representations of the
problem. E
- [ ] The knowledge base is a database that contains all data
related to the decision-making process and includes all
principles that need to be applied to those data in order
to get the outcomes. 
- [ ] The calculation engine applies the rules to the data and
stores the outputs in the knowledge base. The calculation
engine requires human oversight and needs to be configured by humans. 
- [ ] Interaction tools enable the interaction between Samuel
and humans. The output of Samuel needs to be interpreted
by humans, and the input needs to be given by humans.
Interaction tools can have many different forms, ranging
from dashboards accessible via the web browser to search
bars and forms for input
- [ ] The data pipelines are ETL (extract, transform, load)15 flows
that ingest data from various sources outside the team.
Here the IDs are matched, and data are prepared for further usage
- [ ] The knowledge pipelines are an active collaboration
between humans and Samuel. Knowledge is constantly
evolving, and as such, the principles need to be updated
regularly. Thus, it is important that whenever new knowledge is created after a group discussion, this knowledge is
made explicit and codified in the form of principles
- [ ] Interaction with Samuel during the investment process can
be grouped into four types: evaluating the buy/hold/sell
decision, preparing the proposal, portfolio monitoring, and
letting Samuel learn.
- [ ] The evaluation of the final decision is the moment where
the human-proposed action is ranked by Samuel versus all
other potential actions. If other actions are more favorable,
the portfolio managers that do the proposal will have to
explain why they deviate from Samuel.
- [ ] Compared to the human team, Samuel excels in portfolio
monitoring. Samuel collects the characteristics to monitor
from each individual investment and contains the portfolio
construction principles.
- [ ] Whenever
after reconciliation of the proposed action with Samuel
a consistent omission in Samuel’s reasoning is found, its
principles need to be adjusted. Doing so requires that the
omission be made explicit, and it has to be made clear
what new principles need to be added.
- [ ] Collaborating with a digital colleague like Samuel offers three key advantages:

    - Improved Decision Making: A digital colleague helps counter human cognitive biases, which can distort decision-making. Rule-based models, like Samuel, often outperform human judgment because they consistently apply best practices and guidelines. Cognitive biases can cause inconsistency in decisions, both between individuals and within the same individual over time. Samuel mitigates this variability by making decisions based on agreed-upon principles, providing a consistent and reliable benchmark.

    - Transparency: Samuel enhances transparency by making the decision-making process more understandable. This is crucial for explaining investment decisions to clients and stakeholders. For example, investors can trace how responsible investment factors, like ESG (Environmental, Social, and Governance) considerations, influence decision-making. This transparency helps ensure that the reasoning behind investment choices is clear and accessible.

    - Improved Efficiency: Samuel streamlines the decision-making process by automating lower-level decisions, which saves time and improves the overall efficiency of the investment process. It provides contextualized suggestions, like rental growth forecasts or discount rates, embedded in the workflow. These recommendations can be automatically updated, such as with new data for discount rate building blocks, improving both speed and accuracy in decision-making.
## Keynote on Algorithmic Bias (Drs. Isbell and Littman)
- [ ] You cant escape hyperparameters and latent variables: Machine Learning as a Software Engineer Enterprise
- [ ] Basic Conceit: As a community, ew are complier hackers. We should send at least as much time being software engineers, language nerds and ethnographers 
- [ ] Algorithmic Bias: "Math likes some people, and hates other people" Machine Learning can be biased towards certain people - Is it the data or the algorithm that is biased? ML issue or Implementation issue?
- [ ] Past technology: Decisions are made in selection and creation in each stage - Kodak made decisoins to create the best "printed photos" but those decisions resulted in too dark representation of black skined people on film. "Some decisions are justified in making tradeoffs, but they are tradeoffs nonetheless and they make some things better, and some things worse".
- [ ] Training algorithms Intervention: recognize an anomaly and deal with different groups differently: receognize when the learner was consistently wrong for a certain parameter, and change for that parameter. 
- [ ] Software Engineering: Systematic application of methods, pricples and techniques to build software in a rigorous way - provides a framework for factoring in the consequences of technical and non-technical decisions
- [ ] Current technology: Negative Applications of predictive algorithms. PULSE / StyleGAN: took low res photots and built high res systems - often would change the feaures of people of color when it upsampled images to look more "white" 
- [ ] "If you want the ML to do something, you have to specifically ask it to do that"
- [ ] Facial Recognition issues with gender and races. The way we use algos can create technical bias. 
- [ ] Security issues, who's data is being used for facial recognition systems? 
- [ ] The drive to improve model accuracy has coincided with both an increase in model parameters and demand for data. This demand has led to data collection from an increasing number of people, often without their knowledge
- [ ] Theres many kinds of bias, technical bias, objective bias, personal bias. Bias is a problem and it can creep in anywhere in the pipeline and needs to be addressed
- [ ] Tools to solve the Bias Problem already exist: Environment-Independent Task Specifications via GLTL, Accessibility for nonML people when the models are making decisions for people. Interpretable ML models are better than "black box" 
- [ ] If we apply computation specification thinking, we can bring more people along and that would help us to identify and fix bias
- [ ] Adaptive Programming - Probability-based, utility-based, trained(learning)
- [ ] Computational Thinking teaches us how to specify what we want exactly and more diversity in ML ensures that all perspectives are heard
- [ ] Larger sample of people from different background showing an agent how to do things removes bias
- [ ] important to pay attention to whole learning pipeline, there are interesting technical problems to be solved, moral concerns about what happens to people if we get this wrong, taking the long view is essential
- [ ] Bias stems from the decisions we make, but we can identify it and be cognizant of it 
## Interview with Tammer Kamel (Ed Lessons)
- [ ] Quandl platform - gets the data that professional investors need into their hand and into the format they need it in to use for managing money
- [ ] Methods for getting data: Low level API and they pull db into their database or analysis tool.
- [ ] Mix of historical and updated daily data
- [ ] heart is no SQL database, with an API built ontop. With a website Ruby on Rails websigte.
- [ ] No SQL because of scalability - deliver a lot of numerical data to lots of people very quickly. 10-50 million
- [ ] Timestamping - risk of creating a look-ahead bias.
- [ ] Flaws in data: no perfect data set, but publishers are keen to correct errors quickly because theyve got customers who know its them that provide the data. removes opacity from end user and provider which creates improvement in quality
- [ ] Jump Diffusion: model that caputres sixth sigma event. If you mix normal distributes and drew from them at different probabilities, mixing two Gaussian distributes mimics reality better, especially if one has higher SD - Draw from both is great for risk management because it simulates rare events and creates fatter tails.
- [ ] Strategies that Hedge funds use: quantitative fianancing, yield per arbitrage (model swap curve with 2-3 factor model you could discover at any given time thers deviation between current curve and what a sensible yeild curve would look like). Mispriced relative to its peers on that yield curve.
- [ ] A yield curve is a graphical representation that shows the relationship between the interest rates (or yields) of bonds with the same credit quality but different maturities (i.e., the time to maturity). It typically plots the yields of government bonds, such as U.S. Treasury bonds, at various maturities ranging from short-term (e.g., 1 month) to long-term (e.g., 30 years).
- [ ] A credit swap (more formally known as a credit default swap, or CDS) is a financial derivative contract that allows one party to transfer the credit risk of a debt instrument to another party. It's essentially a form of insurance against the default of a borrower (e.g., a corporation or government) or a specific debt (e.g., bonds or loans).
- [ ] 
- [ ] Distribution of returns for one strategy or one stock, combine distributions for multiple different strategies or stocks. 
- [ ] A Six Sigma event refers to a process or occurrence that is six standard deviations away from the mean, which in statistical terms indicates a very rare or extreme event.
- [ ] In the context of Six Sigma methodology, which focuses on improving processes by identifying and eliminating defects, "Six Sigma" itself represents a level of process quality where the defect rate is incredibly low. Specifically, achieving Six Sigma means that the number of defects in a process is 3.4 defects per million opportunities.
- [ ] In the context of financial markets, a Six Sigma event is often used to describe a highly improbable event, such as a massive stock market drop or a rare financial crisis. It implies an event that is so unlikely that it typically happens once every few million instances.
- [ ] Find new data sources to find alpha
- [ ] theortically sound, empirically testable, and simple  GOOD
- [ ] biggest trap: calibrating the same model with teh same data and seeing "good reults" can be caused by curve fitting

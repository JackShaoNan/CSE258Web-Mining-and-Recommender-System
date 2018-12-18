#Models

Here are our experiments with different models. In this assignment, we explored three different machine learning algorithms: linear regression, neural network and lightGBM. Each has various strengths and weakness that we believe would help provide insights into this problem.

##Linear regression
The basic intuition that comes to our mind is using linear regression for this predictive problem. <br>

linear regression is all about a linear approach to modelling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables).<br>

When extracting input parameters, the following questions comes out:
For PUBG, what is the best strategy to win in it? Should we just sit in one spot and hide our way into victory, or do we need to be the top shot? Having taken the best strategy into consideration, we select the following parameters for linear predictor function: <br>

 * damageDealt
 * headshotKills
 * killPlace
 * killStreaks
 * rideDistance
 * roadKills
 * swimDistance
 * vehicleDestroys
 * walkDistance
 * weaponsAcquired
 * winPlacePerc
 
PROS: <br>
We have to admit that linear regression is studied rigorously and used extensively in practical applications. This is because the statistical properties of the resulting estimators are easier to determine. And the goal of PUBG problem is prediction, that why we choose lieaner regression that is good at fitting a predictive model to an observed data set of values of the reponse and explanatory variables. Also after developing such model, if additional values of the explanatory variables are collected without an accompanying response value, the fitteed model can be used to make a prediction of the response easily.<br>

CONS:<br>
But, at the same time, we have seen that the result is not that satisfied. We got a really small mean squared error value at training set, while applied it to test set, we just got a result lower than average. The reason behind that maybe we have a little overfitting by using linear regression or LR is just too simple to handle such complex problems with multiple parameters.


##Neural Network
When linear model fails, we turn to non-linear model: neural network. Artificial neural networks (ANN) or connectionist systems are computing systems vaguely inspired by the biological neural networks that constitute animal brains. The neural network itself is not an algorithm, but rather a framework for many different machine learning algorithms to work together and process complex data inputs. For this task, we choose the Multi-layer Perceptron regressor. This model optimizes the squared-loss using LBFGS or stochastic gradient descent.<br>
A multilayer perceptron (MLP) is a class of feedforward artificial neural network. An MLP consists of, at least, three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training. Its multiple layers and non-linear activation distinguish MLP from a linear perceptron. It can distinguish data that is not linearly separable.<br>

After hyper-parameter tuneing, we spercify the parameters as follows:

* solver='adam'
* hidden_layer_sizes=(10,10,10)
* alpha=0.1
* random_state=1

PROS:<br>

The part of reasons that LR fails in this problem is that the parameters involved in are not lineraly separable, so we can not get accurate prediction with linear method. But MLP does it well.
MLPRegressor trains iteratively since at each time step the partial derivatives of the loss function with respect to the model parameters are computed to update the parameters. It can also have a regularization term added to the loss function that shrinks model parameters to prevent overfitting. In this way, we improved the performance massively on test set.

CONS:<br>
While we got the good result through MLP, we find that, on the other hand, it is really time-consuming and low efficiency. It takes long time to train model and performs not well on big data.



##LightGBM
To make it more efficient and less costly, we try to find other substitutes. We find that LightGBM may be a good fit for solving our problem. Light GBM is just a gradient boosting framework that uses tree based learning algorithm. Light GBM grows tree vertically while other algorithm grows trees horizontally meaning that Light GBM grows tree leaf-wise while other algorithm grows level-wise. It will choose the leaf with max delta loss to grow. When growing the same leaf, Leaf-wise algorithm can reduce more loss than a level-wise algorithm. That is an important difference between lightGBM and other boosting algorithms.<br>

For parameter choosing, we spercified the following:

* objective: regression
* metric: mae
* other: default

PROS:<br>
Light GBM is prefixed as ‘Light’ because of its high speed. Light GBM can handle the large size of data and takes lower memory to run. Another reason of why Light GBM is chosen here is because it focuses on accuracy of results. LGBM also supports GPU learning and thus data scientists are widely using LGBM for data science application development. So basicaly, we can get good result in a short time period with limited resources.

CONS:<br>
Implementation of Light GBM is easy, the only complicated thing is parameter tuning. Light GBM covers more than 100 parameters. So we have to find out the important ones and focus more on parameter tuning.<br>
Due to the limited time and resources, we only explored two main parameters: application(regression) and metric parameter(mean absolute error). 
Actually, there are still many other parameters that maybe very helpful, such as: boosting(defines the type of algorithm you want to run), num_boost_round(Number of boosting iterations), learning_rate(This determines the impact of each tree on the final outcome)and num_leaves(number of leaves in full tree). If we spend more time on them, we believe we can get better performance.

































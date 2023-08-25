# Perceptron_Learning_Algo

## The Perceptron Algorithm

The perceptron is a classic learning algorithm for the neural model of learning.

```Python
 PerceptronTrain(D,MaxIter) 

w_d <- 0 for all d= 1 to D               // initialize weights
b <- 0                                 // initialize bias
for iter=1 to MaxIter do
  for all (x,y) ∈ D do
     a ← ∑ w_d * x_d + b              // compute activation for this example
     if ya <= 0 then
       w_d ← w_d + yx_d, for all d = 1...D   // update weights
       b <- b + y                          // update bias 
     endif
   endfor
endfor
return w_1,w_2,w_3,...,w_D,b
```
```Python

PerceptronTest(w_0,w_1,...,w_D,b,x̂ )

a ← ∑ w_d * x̂ d + b      // compute activation for the test example
return sing(a)
```

The algorithm is actually quite different than either the decision tree algorithm or the KNN algorithm. First, it is online. 
This means that instead of considering the entire data set at the same time, it only ever looks at one example. It processes that example and then goes on to the next one. Second, it is error-driven. This means that, so long as it is doing well, it doesn’t bother updating its parameters.

The algorithm maintains a “guess” at good parameters (weights and bias) as it runs. It processes one example at a time. For a given example, it makes a prediction. It checks to see if this prediction is correct (recall that this is training data, so we have access to true labels). If the prediction is correct, it does nothing. Only when the prediction is incorrect does it change its parameters, and it changes them in such a way that it would do better on this example the next time around. It then goes on to the next example. Once it hits the last example in the training set, it loops back around for a specified number of iterations.

There is one “trick” in the training algorithm, which probably seems silly, but will be useful later. It is in line 6, when we check to see if we want to make an update or not. We want to make an update if the current prediction (just SIGN(a)) is incorrect. The trick is to multiply the true label y by the activation a and compare this against 0. Since the label y is either +1 or -1 , you just need to realize that ya is positive whenever a and y have the same sign. In other words, the product ya is positive if the current prediction is correct.

The particular form of update for the perceptron is quite simple. The weight w_d is increased by yx_d and the bias is increased by y. The goal of the update is is to adjust the parameters so that they are “better” for the current example. In other words, if we saw this example twice in a row, we should do a better job the second time around.

To see why this particular update achieves this consider the following scenario. We have some current set of parameters w_1,…,w_D,b. We observe an example (x,y). For simplicity, suppose this is a positive example, so y = +1. We compute an activation a, and make an error. Namely, a < 0.We now update our weights and bias. Let’s call the new weights w’_1,w’_2,…,w’_D, b’. Suppose we observe the same example again and again and need to compute a new activation a’. We proceed with some algebra:

So the difference between the old activation a and the new activation a’ is:

But x_d² ≥ 0 , since its squared. So this value is always at least one. Thus, the new activation is always at least the old activation plus one. Since this was a positive example, we have successfully moved the activation in the proper direction.

The only hyperparameter of the perceptron algorithm is MaxIter, the number of passes to make over the training data. If we make too many passes over the training data, then the algorithm is likely to overfit. On the other hand, going over the data only one time might lead to underfitting.

One aspect of the perceptron algorithm that is left underspecified is line 4, which says: loop over all the training examples. The natural implementation of this would be to loop over them in a constant order. This is actually a bad idea.

Consider what the perceptron algorithm would do on a dataset that consisted of 500 positive examples followed by 500 negative examples. After seeing the first few positive examples (maybe 5), it would likely decide that every example is positive, and would stop learning anything. It would do well for a while (next 495 examples), until it hit the batch of negative examples. Then it would take a while (maybe ten examples) before it started predicting everything as negative. By the end of one pass through the data, it would really only have learned from a handful of examples (fifteen in this case).

So one thing we need to avoid is presenting the examples in some fixed order. This can easily be accomplished by permuting the order of examples once in the beginning and then cycling over the dataset in the same (permuted) order each iteration. However, it turns out that you can actually do better if you re-permute the examples in each iteration. In practice, permuting each iteration tends to yield about 20% savings in number of iterations. In theory, you can actually prove that it’s expected to be about twice as fast.
Geometric Interpretation

A question that arises now is: what does the decision boundary of a perceptron look like? You can actually answer the question mathematically. For a perceptron, the decision boundary is precisely where the sign of the activation, a, changes from -1 to +1. In other words, it is the set of points x that achieve zero activation. The points that are not clearly positive nor clearly negative. For simplicity, we’ll first consider the case where there is no “bias” term (or, equivalently, the bias is zero). Formally, the decision boundary beta is:

We can now apply some linear algebra. Recall that ∑ w_d · x_d is just the dot product between the vectors w = <w1,w2,…wD> and the vector x. This is written as w · x. Two vectors have a zero dot product only if they are perpendicular. Thus, if we think of the weights as a vector w, then the decision boundary is simply the plane perpendicular to w.

This is shown pictorially in the image below. Here the weight vector is shown, together with its perpendicular plane. This plane forms the boundary between negative points and positive points. The vector, w, points in the direction of the positive examples and away from the negative examples.

One thing to notice is that the scale of the weight vector is irrelevant to classification. Suppose you take a weight vector w and replace it with 2w. All activations are now doubled. But their sign does not change. This makes sense geometrically, since all that matters is which side of the plane a test point falls on, not how far it is from the plane. For this reason, it is common to work with normalized weight vectors, w, that have length one; i.e. ||w|| = 1.

The geometric intuition can help us even more when we realize that dot products compute projections. That is, the value w · x is just the distance of x from the origin when projected onto the vector w. This is shown in the figure below:

In that figure, all the data points are projected onto w. Now we can think of this as a one-dimensional version of the data, where each data point is placed according to its projection along w. This distance is exactly the activation of that example, with no bias.

From here, we can start thinking about the role of the bias term. Previously, the threshold would be at zero. Any example with a negative projection onto w would be classified negative; any example with a positive projection, positive. The bias simply moves this threshold. Now, after the projection is computed, b is added to get the overall activation. The projection plus b is then compared against 0.

Thus, from a geometric perspective, the role of the bias is to shift the decision boundary away from the origin, in the direction of w. It is shifted exactly -b units. So if b is positive, the boundary is shifted away from w, and if b is negative, the boundary is shifted toward w. This makes intuitive sense: a positive bias means that more examples should be classified positive. By moving the decision boundary in the negative direction, more space yields a positive classification.
Perceptron Convergence and Linear Separability

We already have an intuitive feeling for why the perceptron works: it moves the decision boundary in the direction of the training example. An important question is: does the perceptron converge? If so, what does it converge to? And how long does it take?

It is easy to construct data sets on which the perceptron algorithm will never converge. In fact, consider the (very uninteresting) learning problem with no features. You have a data set consisting of one positive example and one negative example. Since there are no features, the only thing the perceptron algorithm will ever do is adjust the bias. Given this data, you can run the perceptron for a bajillion iterations and it will never settle down. As long as the bias is non-negative, the negative example will cause it to decrease. As long as it is non-positive, the positive example will cause it to increase. Ad infinitum. (Yes, this is a very contrived example.)

What does it mean for the perceptron to converge? It means that it can make an entire pass through the training data without making any more updates. In other words, it has correctly classified every training example. Geometrically, this means that it was found some hyperplane that correctly segregates the data into positive and negative examples.

In this case, this data is linearly separable. This means that there
exists some hyperplane that puts all the positive examples on one side
and all the negative examples on the other side. If the training is not
linearly separable then the perceptron has no hope of converging. It could never possibly classify each point correctly.

The somewhat surprising thing about the perceptron algorithm is that if the data is linearly separable, then it will converge to a weight vector that separates the data. (And if the data is inseparable, then it will never converge). This is great news. It means that the perceptron converges whenever it is even remotely possible to converge.

The second question is: how long does it take to converge? By “how long,” what we really mean is “how many updates?” As is the case for much learning theory, you will not be able to get an answer of the form “it will converge after 5293 updates.” This is asking too much. The sort of answer we can hope to get is of the form “it will converge after at most 5293 updates.”

What we might expect to see is that the perceptron will converge more quickly for easy learning problems than for hard learning problems. This certainly fits intuition. The question is how to define “easy” and “hard” in a meaningful way. One way to make this definition is through the notion of margin. If I give you a dataset and hyperplane that separates it, then the margin is the distance between the hyperplane and the nearest point. Intuitively, problems with large margins should be easy (there’s a lot of “wiggle room” to find a separating hyperplane) ; and problems with small margins should be hard (you really have to get a very specific well tuned weight vector).

Formally, the margin is only defined if w, b actually separate the data (otherwise it is -∞). In the case that it separates the data, we find the point with the minimum activation, after the activation is multiplied by the label.

For some historical reason, margins are always demoted by the Greek letter γ (gamma). Often one talks about the margin of the dataset. The margin of a dataset is the largest attainable margin on this data. Formally,

margin (D) = max (margin(D, w, b))

In words, to compute the margin of a dataset, you “try” every possible w, b pair. For each pair, you compute its margin. We then take the largest of these as the overall margin of the data. If the data is not linearly separable, then the value of the max, and therefore the value of the margin, is −∞.

There is a famous theorem by Rosenblatt that shows that the number of errors that the perceptron algorithm makes is bounded by γ^-2.
Perceptron Convergence Theorem

Suppose the perceptron algorithm is run on a linearly separable dataset D with margin γ >0 . Assume that ||x|| ≤ 1for all x ∈ D. Then the algorithm will converge after at most 1/ γ² updates.

The proof of this theorem is is somewhat complicated, but the idea behind the proof is as follows. If the data is linearly separable with margin γ,
then there exists some weight vector w* that achieves this margin. Obviously, we don’t know what w* is, but we know it exists. The perceptron algorithm is trying to find a weight vector w that points roughly in the same direction as w*. Every time the perceptron makes an update, the angle between w and w* changes. What can be proved is that the angle actually decreases. This is shown in two steps. First, the dot product w · w* increases a lot. Second, the norm ||w|| does not increase very much. Since the dot product is increasing, but w isn’t getting too long, the angle between them has to be shrinking.
Limitations of the Perceptron

Although the perceptron is very useful, it is fundamentally limited in a way that neither decision trees nor KNN are. Its limitation is that its decision boundaries can only be linear. The classic way of showing this limitation is through the XOR problem. This is shown graphically in the image below:

It consists of four data points, each at a corner of the unit square. The labels for these points are the same, along the diagonals. You can try, but you will not be able to find a linear decision boundary that perfectly separates these data points.

One question you might ask is: do XOR-like problems exist in the real world? Unfortunately for the perceptron, the answer is yes. Consider a sentiment classification problem that has three features that simply say whether a given word is contained in a review of a course. These features are: excellent, terrible and not. The excellent feature is indicative of positive reviews and the terrible feature is indicative of negative reviews. But in the presence of the not feature, this categorization flips.

One way to address this problem is by adding feature combina tions. We could add two additional features: excellent-and-not and terrible-and-not that indicate a conjunction of these base features. By assigning weights as follows, you can achieve the desired effect:

w(excellent) = +1 w(execllent-and-not) = −2
w(terrible) = −1 w(terrible-and-not) = +2
w(not) = 0

In this particular case, we have addressed the problem. However, if we start with D-many features, if we want to add all pairs, we’ll blow up to O(D²) features through this feature mapping. And there’s no guarantee that pairs of features is enough. We might need triples of features, and now we’re up to O( D³) features. These additional features will drastically increase computation and will often result in a stronger propensity to overfitting.

In fact, the “XOR problem” is so significant that it basically killed
research in classifiers with linear decision boundaries for a decade
or two. In some future post, I will discuss two alternative approaches to
taking key ideas from the perceptron and generating classifiers with
non-linear decision boundaries. One approach is to combine multiple perceptrons in a single framework: this is the neural network approach. The second approach is to find computationally efficient ways of doing feature mapping in a computationally and statistically efficient way: this is the kernels approach.

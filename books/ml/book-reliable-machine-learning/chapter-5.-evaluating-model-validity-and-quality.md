# Chapter 5. Evaluating Model Validity and Quality

OK, so our model developers have created a model that they say is ready to go into production. Or we have an updated version of a model that needs to be swapped in to replace a currently running version of that model in production. Before we flip the switch and start using this new model in a critical setting, we need to answer two broad questions. The first establishes model _validity_: will the new model break our system? The second addresses model _quality_: is the new model any good?

These are simple questions to ask but may require deep investigation to answer, often necessitating collaboration among folks with various areas of expertise. From an organizational perspective, it is important for us to develop and follow robust processes to ensure that these investigations are carried out carefully and thoroughly. Channeling our inner Thomas Edison, it is reasonable to say that model development is 1% inspiration and 99% verification.

This chapter dives into questions of both validity and quality, and provides enough background to allow MLOps folks to engage with both of these issues. We will also spend time talking about how to build processes, automation, and a strong culture around ensuring that these issues are treated with the appropriate attention, care, and rigor that practical deployment demands.

[Figure 5-1](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch05.html#a\_simplified\_view\_of\_the\_repeated\_cycle) outlines the basic steps of model development and the role that quality plays in it. While this chapter focuses on evaluation and verification methods, it is important to note that these processes will likely be repeated over time in an iterative fashion. See [Chapter 10](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch10.html#continuous\_ml) for more on these topics.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098106218/files/assets/reml_0501.png" alt="A simplified view of the repeated cycle of model development" height="355" width="600"><figcaption></figcaption></figure>

**Figure 5-1. A simplified view of the repeated cycle of model development**

## Evaluating Model Validity

It has been said that all humans crave validation of one form or another, and we MLOps folks are no different. Indeed, _validity_ is a core concept for MLOps, and in this context we cover the concept of whether a model will create system-level failures or crashes if put into production.

The kinds of things that we consider for validity checks are distinct from model quality issues. For example, a model could be horribly inaccurate, incorrectly guessing that every image shown should be given the label `chocolate pudding` without causing system-level crashes. Similarly, a model might be shown to have wonderful predictive performance in offline tests, but rely on a particular feature version that is not currently available in the production stack, or use an incompatible version of some ML package, or rarely give values of `NaN` that cause downstream consumers to crash. Testing for validity is a first step that allows us to be sure that a model will not cause catastrophic harm to our system.

Here are some things to test for when verifying that a model will not bring our system to its knees:

Is it the right model?Surprisingly easy to overlook, it is important to have a foolproof method for ensuring that the version of the model we are intending to serve is the version we are actually using. Including timestamp information and other metadata within the model file is a useful backstop. This issue highlights the importance of automation and shows the difficulties that can arise with ad hoc manual processes.Will the model load in a production environment?

To verify this, we create a copy of the production environment and simply try to load the model. This sounds pretty basic, but it is a good place to start because it is surprisingly easy for errors to occur at this stage. As you’ll learn in [Chapter 7](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#training\_systems), we are likely to take a trained version of a model and copy it to another location where it will be used either for offline scoring by a large batch process, or for online serving of live traffic on demand. In either case, the model is likely to be stored as a file or set of files in a particular format that are then moved, copied, or replicated for serving. This is necessary because models tend to be large, and we also want to have versioned checkpoints around that can be used as artifacts both for serving and for future analysis or as recovery options in case of an unforeseen crisis or error. The problem is that file formats tend to change slightly over time as new options are added, and there is always at least some chance that the format a model is saved in is not a format compatible with our current serving system.

Another issue that can create loading errors is that the model file may be too large to be loaded into available memory. This is especially possible in memory-constrained serving environments, such as on-device settings. However, it can also occur when model developers zealously increase model size to pursue additional accuracy, which is an increasingly common theme in contemporary ML. It is important to note that the size of the model file and the size of the model instantiated in memory are correlated, but often only loosely so, and are absolutely not identical. We cannot just look at the size of the model file to ensure that the resulting model is sufficiently small to be successfully loaded in our environment.

Can the model serve a result without crashing the infrastructure?

Again, this seems like a straightforward requirement: if we feed the model one minimal request, does it give us a result back of any kind, or does the system fail? Note that we say “one minimal request” as opposed to many requests intentionally, because these sorts of tests are often best done with single examples and single requests to start. This both minimizes risk and makes debugging easier if failures do arise.

Serving the result for one request might cause a failure, for several reasons:

Platform version incompatibility

Especially when using third-party or open source platforms, the serving stack could easily be using a different version of the platform than the training stack used.

Feature version incompatibility

The code for generating features is often different in the training stack from the serving stack, especially when each stack has different memory, compute cost, or latency requirements. In such cases, it is easy for the code that generates a given feature to get out of sync in these different systems, causing failures—one form of a general class of problems often referred to as _training-serving skew_. For example, if a dictionary is used to map word tokens to integers, the serving stack might be using a stale version of that dictionary even after a newer one was created for the training stack.

Corrupted model

Errors happen, jobs crash, and in the end our machines are physical devices. It is possible for model files to become corrupted in one way or another, either through error at write time or by having `NaN` values written to disk if there were not sufficient sanity checks in training.

Missing plumbing

Imagine that a model developer creates a new version of a feature in training, but neglects to implement or hook up a pipeline that allows that feature to be used in the serving stack. In these cases, loading in a version of the model that relies on that feature will result in crashes or undesirable behavior.

Results out of range

Our downstream infrastructure might require the model predictions to be within a given range. For example, consider what might happen if a model is supposed to return a probability value between 0 and 1, not inclusive, but instead returns a score of exactly 0.0, or even –0.2. Or consider a model that is supposed to return 1 of 100 classes to signify the most appropriate image label, but instead returns 101.

Is the computational performance of the model within allowable bounds?

In systems that use online serving, models must return results on the fly, which typically means tight latency constraints must be met. For example, a model intended to do real-time language translation might have a budget of only a few hundred milliseconds to respond. A model used within the context of high-frequency stock trading might have significantly less.

Because such constraints are often in tension with each other, for changes made by model developers in the search for increased accuracy, it is critical to measure latency before deployment. In doing so, we need to keep in mind that the production settings are likely to have peculiarities around hardware or networking that create bottlenecks that might differ from a development environment, so latency testing must be done as close to the production setting as possible. Similarly, even in offline serving situations, overall compute cost can be a significant limiting factor, and it is important to assess any cost changes before kicking off huge batch-processing jobs. Finally, as discussed previously, storage costs, such as size of the model in RAM, are another critical limitation and must be assessed prior to deployment. Checks like this can be automated, but it may also be useful to verify manually to consider trade-offs.

For online serving, does the model pass through a series of gradual canary levels in production?Even after we have some confidence in a new model version based on the validation checks, we will not want to just flip a switch and have the new model suddenly take on the full production load. Instead, our collective stress level will be reduced if we first ask the model to serve just a tiny trickle of data, and then gradually increase the amount after we have assurance that the model is performing as expected in serving. This form of canary ramp-up is a place where model validation and model monitoring, as discussed in [Chapter 9](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch09.html#monitoring\_and\_observability\_for\_models), overlap to a degree: our final validation step is a controlled ramp-up in production with careful monitoring.

## Evaluating Model Quality

Ensuring that a model passes validation tests is important, but by itself does not answer whether the model is good enough to do its job well. Answering these kinds of questions takes us into the realm of evaluating model quality. Understanding these issues is important for model developers, of course, but is also critical for both organizational decision makers and for MLOps folks in charge of keeping systems running smoothly.

### Offline Evaluations

As discussed in the whirlwind tour of the model development lifecycle in [Chapter 2](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch02.html#data\_management\_principles), model developers typically rely on offline evaluations, such as looking at accuracy on a held-out test set as a way to judge how good a model is. Clearly, this kind of evaluation has limitations, as we will talk about later in this chapter—after all, accuracy on held-out data is not the same thing as happiness or profit. Despite their limitations, these offline evaluations are the bread and butter of the development lifecycle because they offer a reasonable proxy while existing in a sweet spot of low cost and high efficiency that allows developers to test many changes in rapid succession.

Now, what is an evaluation? An _evaluation_ consists of two key components: a performance metric and a data distribution. _Performance metrics_ are things like accuracy, precision, recall, area under the ROC curve, and so on—we will talk about a few of these later on if they are not already familiar. _Data distributions_ are things like “a held-out test set that was randomly sampled from the same source of data as the training data” that we have talked about before, but held-out test data is not the only distribution that might be important to look at. Others might include “images specifically from roads in snowy conditions” or “yarn store queries from users in Norway” or “protein sequences that have not previously been identified by biologists.”

An evaluation is always composed of both a metric and a distribution together—the evaluation shows the model’s performance on the data in that distribution, as computed by the chosen metric. This is important to know because folks in the ML world sometimes use shorthand and say things like “this model has better accuracy” without clarifying what the distribution is. This sort of shorthand can be dangerous for our systems because it neglects to mention which distribution is used in the evaluation, and can lead to a culture in which important cases are not fully assessed. Indeed, many of the issues around fairness and bias that emerged in the late 2010s can likely be tracked down to not giving sufficient consideration to the specifics of the data distribution used at test time during model evaluations. Thus, when we hear a statement like “accuracy is better,” we can always add value by asking _on what distribution?_

### Evaluation Distributions

Perhaps no question is more important in the understanding of an ML system than deciding how to create the evaluation data. Here are some of the most common distributions used, along with some factors to consider as their strengths and weaknesses.

#### Held-out test data

The most common evaluation distribution used is _held-out test data_, which we covered in [Chapter 3](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch03.html#basic\_introduction\_to\_models) when reviewing the typical model lifecycle. On the surface, this seems like an easy thing to think about—we randomly select some of our training data to be set aside and used only for evaluation. When each example in the training data has an equal and independent chance of being put into the held-out test data, we call this an _IID test set_. The _IID_ term is statistics-speak that means _independently and identically distributed_. We can think of the IID test set process as basically flipping a (maybe biased) coin or rolling a die for each example in the training data, and holding each one out for the IID test set based on the result.

The use of IID test data is widely established, not because it is necessarily the most informative way to create test data, but because it is the way that respects the assumptions that underpin some of the theoretical guarantees for supervised ML. In practice, a purely IID test set might be inappropriate, though, because it might give an unrealistically rosy view of our model’s expected performance in deployment.

As an example, imagine we have a large set of stock-price data, and we want to train a model to predict these prices. If we create a purely IID test set, we might have training data from 12:01 and 12:03 from a given day in the training data, while data from 12:02 ends up in the test data. This would create a situation in which the model can make a better guess about 12:02 because it has seen the “future” of what 12:03 looks like. In reality, a model that is guessing about 12:02 would be unable to have access to this kind of information, so we would need to be careful to create our evaluations in a way that does not allow the model to train on “future” data. Similar examples might exist in weather prediction, or yarn product purchase prediction.

The point here is not that IID test distributions are always bad, but rather that the details here really do matter. We need to apply careful reasoning and common sense to the creation of our evaluation data, rather than relying on fixed recipes.

#### Progressive validation

In systems with a time-based ordering to data—like our preceding stock-price prediction example—it can be quite helpful to use a progressive validation strategy, sometimes also called _backtesting_. The basic idea is to simulate how training and prediction would work in the real world, but playing the data through to the model in the same order that it originally appeared.

For each simulated time step, the model is shown the next example and asked to make its best prediction. That prediction is then recorded and incorporated to the aggregate evaluation metric, and only then is the example shown to the model for training. In this way, each example is first used for evaluation and then used for training. This helps us see the effects of temporal ordering, and ask questions like, “What would this model have done last year on election day?”

The drawback is that this setup is somewhat awkward to adapt if our models require many passes over the data to train well. A second drawback is that we must be careful to make comparisons between models based on evaluation data from exactly the same time range. Finally, not all systems will operate in a setting in which the data can meaningfully be ordered in a canonical way, like time.

#### Golden sets

In models that continually retrain and evaluate using some form of progressive validation, it can be difficult to know whether the model performance is changing or whether the data is getting easier or harder to predict on. One way to control this is to create a _golden set_ of data that models are not ever allowed to train on, but that is from a specific point in time. For example, we might decide that the data from October 1 of last year might be set aside as golden set data, never to be used for training under any circumstance, but held aside.

When we set aside the golden set of data, we also keep with it either the results of running that set of data through our model or, in some cases, the result of having humans evaluate the golden set. We might sometimes treat these results as “correct” even if they are really just the predictions for those examples from a specific process and at a particular point in time.

Performance on golden set data like this can reveal any sudden changes in model quality, which can aid debugging greatly. Note that golden set evaluations are not particularly useful for judging absolute model quality, because their relevance to current performance diminishes as their time period recedes into the past. Another problem can arise if we are not able to keep golden set data around for very long (for example, to respect certain data privacy laws or to respond to requests for deletion or expiration of access). The primary benefit of golden sets is to identify changes or bugs, because, typically, model performance on golden set data changes only gradually as new training data is incorporated into the model.

#### Stress-test distributions

When deploying models in the real world, one worry is that the data they may encounter in reality may differ substantially from the data they were shown in training. (These issues are sometimes described by different names in the literature, including _covariate shift_, _nonstationarity_, or _training-serving skew_). For example, we might have a model trained largely on image data from North America and Western Europe, but that is then later applied in countries across the globe. This creates two possible problems. First, the model may not perform well on the new kinds of data. Second, and even more important, we may not know that the model would not perform well because the data was not represented in the source that supplied the (supposedly) IID test data.

Such issues are especially critical from a fairness and inclusion standpoint. Imagine we are building a model to predict the yarn color preferred by a user, given a provided portrait image. If our training data does not include portrait images with a wide range of skin tones, an IID test set might not have sufficient representation to uncover problems if the model does not do well for images of people with especially dark skin tones. (This example harkens back to seminal work by Buolamwini and Gebru.)[1](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch05.html#ch01fn39) In cases like this, it’s important to create specific stress-test distributions in which carefully constructed test sets each probe for model performance on different skin tones. Similar logic applies to testing any other area of model performance that might be critical in practice, from snowy streets for navigation systems developed in temperate climates, to a broad range of accents and languages for speech-recognition systems developed in a majority English-speaking workplace.

#### Sliced analysis

One useful trick to consider is that any test set—even an IID test set—can be sliced to effectively create a variety of more targeted stress-test distributions. By _slicing_, we mean filtering the data based on the value of a certain feature. For example, we could slice images to look at performance on images with only snowy backgrounds, or stocks from companies that were only in their first week of trading on the market, or yarns that were only a shade of red. So long as we have at least some data that conforms to these conditions, we can evaluate performance on each of these cases through slicing. Of course, we need to take care not to slice too finely, in which case the amount of data we would be looking at would be too small to say anything meaningful in a statistical sense.

#### Counterfactual testing

One way to understand a model’s performance at a deeper level involves learning what the model would have predicted if the data had been different. This is sometimes called _counterfactual testing_ because the data that we end up feeding to the model runs counter to the actual data in some way. For example, we might ask what the model would predict if the dog in a given image were not on a grassy background, but instead shown against a background of snow or of clouds.[2](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch05.html#ch01fn40) We might ask if the model would have recommended a higher credit score if the applicant had lived in a different city, or if the model would have predicted a different review score if the pronoun for the lead actor in a movie had been switched from _he_ to _she_.

The trick here is that we might not have any examples that match any of these scenarios, in which case we would take the step of creating synthetic counterfactual examples by manipulating or altering examples that we do have access to. This tends to be most effective when we want to test that a given alteration does _not_ substantially change model prediction. Each of these tests might reveal something interesting about the model and the kinds of information sources it relies on, allowing us to use judgment about whether it is appropriate model behavior and whether we need to address any issues prior to launch.

### A Few Useful Metrics

In some corners of the ML world, there is a tendency to look at a single metric as the standard way to view a model’s performance on a given task. For example, accuracy was, for years, the one way that models were evaluated on ImageNet held-out test data. And indeed, this mindset is most often seen in benchmarking or competition settings, in which the use of a single metric simplifies comparisons between different approaches. However, in real-world ML, it is often ill-advised to myopically consider only a single metric. It is better to think of each metric as a particular perspective or vantage point. Just as there is no one best place to watch the sun rise, there is no one best metric to evaluate a given model, and the most effective approach is often to consider a diverse range of metrics and evaluations, each of which has its own strengths, weaknesses, blind spots, and peculiarities.

Here, we will try to build up our intuition around some of the more common metrics. We divide them into three broad categories:

Canary metricsAre great at indicating that something is wrong with a model, but are not so effective at distinguishing a good model from a better one.Classification metricsHelp us understand the impact of a given model on downstream tasks and decisions, but require fiddly tuning that can make comparisons between models more difficult.Regression and ranking metricsAvoid this tuning and make comparisons easier to reason about, but may miss specific trade-offs that might be available when some errors are less costly than others.

#### Canary metrics

As we’ve noted, this set of metrics offers a useful way to tell when something is horribly wrong with our model. Like the fabled canary in the coal mine, if any of these metrics is not singing as expected, then we definitely have a problem to deal with. On the flip side, if these metrics look good, that does not necessarily mean that all is well or that our model is perfect. These metrics are just a first line of detection for potential issues.

**Bias**

Here we use _bias_ in the statistical sense rather than the ethical sense. Statistical bias is an easy concept—if we add up all the things we expect to see based on the model’s predictions, and then add up all the things we actually see in the data, do we get the same amount? In an ideal world, we would, and typically a well-functioning model will show very low bias, meaning a very low difference between the total expected and observed values for a given class of predictions.

One of the nice qualities of bias as a metric is that unlike most other metrics, there is a “correct” value of 0.00 that we do expect most models to achieve. Differences here of even a few percent in one direction or another are often a sign that something is wrong. Bias often couples well with sliced analysis as an evaluation strategy to uncover problems. We can use sliced analysis to identify particular parts of the data that the model is performing badly on as a way to begin debugging and improving the overall model performance.

The drawback of bias as a metric is that it is trivial to create a model with perfectly zero bias, but that is a terrible model. As a thought experiment, this could be done by having a model that just returns the average observed value for every example—zero bias in aggregate, but totally uninformative. A pathologically bad model like this can be detected by looking at bias on more fine-grained slices, but the larger point remains. Bias as a metric is a great canary, but just having zero bias is not by itself indicative of a high-quality model.

**Calibration**

When we have a model that predicts a probability value or a regression estimate, like a probability that a user will click a given product or a numerical prediction of tomorrow’s temperature, creating a _calibration plot_ can provide significant insight into the overall quality of the model. This is done essentially in two steps, which can be thought of roughly as first bucketing our evaluation data in a set of buckets and then computing the model’s bias in each of these buckets. Often, the bucketing is done by model score—for example, the examples that are in the lowest tenth of the model’s predictions go in one bucket, the next lowest tenth in the next bucket, and so on—in a way that brings to mind the idea of sliced analysis discussed previously. In this way, calibration can be seen as an extension of the approach of combining bias and sliced analysis, in a systematic way.

Calibration plots can show systemic effects such as overprediction or underprediction in different areas, and can be an especially useful visualization to help understand how a model performs near the limits of its output range. In general, calibration plots can help show areas where a model may systematically overpredict or underpredict, by plotting the observed rates of occurrence versus their predicted probabilities. This can be useful to help detect situations where the model’s scores are either more or less reliable. For example, the plot in [Figure 5-2](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch05.html#an\_example\_calibration\_plot) shows a model that gives good calibration in the middle ranges, but does not do as well at the extremes, overpredicting on actual low probability examples and underpredicting on actual high-probability examples.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098106218/files/assets/reml_0502.png" alt="An example calibration plot" height="318" width="600"><figcaption></figcaption></figure>

**Figure 5-2. An example calibration plot**

#### Classification metrics

When we think of model evaluation metrics, classification metrics like accuracy are often the first ones that come to mind. Broadly speaking, a _classification metric_ helps measure whether we’ve correctly identified that a given example belongs to a specific category (or _class_). Class labels are typically discrete—things like `click` or `no_click`, or `lambs_wool`, `cashmere`, `acrylic`, `merino_wool`—and we tend to judge a prediction as a binary correct or incorrect on getting a given class label right.

Because models typically report a score for a given label, such as `lambs_wool: 0.6`, `cashmere: 0.2`, `acrylic: 0.1`, `merino_wool: 0.1`, we need to invoke a decision rule of some kind to decide when we are going to predict a given label. This might involve a threshold, like “predict acrylic whenever the score for `acrylic` is above 0.41 for a given image,” or it might ask which class label gets the highest score out of all available options. Decision rules like these are a choice on the part of the model developer, and are often set by taking into account the potential costs of different kinds of mistakes. For example, it may be significantly more costly to miss identifying stop signs than to miss identifying merino wool products.

With that background, let’s look at a couple of classic metrics.

**Accuracy**

In conversation, many folks use the term _accuracy_ to mean a general sense of goodness, but accuracy also has a formal definition that shows the fraction of predictions for which the model was correct. This satisfies an intuitive desire—we want to know how often our model was right. However, this intuition can sometimes be misleading without appropriate context.

To place an accuracy metric into context, we need to have some understanding of how good a naive model that always predicts the most prevalent class would be, and also to understand relative costs of different types of errors. For example, 99% accuracy sounds pretty good, but may be completely terrible if the goal is figuring out when it is safe to cross a busy street—we would be almost sure to be in an accident quite soon. Similarly, 0.1% accuracy sounds horrible but would be an amazingly good performance if the goal was to predict winning lottery number combinations. So, when we hear an accuracy value quoted, the first question should always be, “What is the base rate of each class?” It is also worth noting that seeing 100% accuracy—or perfect performance on any metric—is most often cause for concern rather than celebration, as this may indicate overfitting, label leakage, or other problems.

**Precision and recall**

These two metrics are often paired together, and are related in an important way. Both metrics have a notion of a _positive_, which we can think of as “the thing we are trying hard to find.” This can be finding spam for a spam classifier, or yarn products that match the user’s interest for a yarn store model. These metrics answer the following related questions:

PrecisionWhen we said an example was a positive, how often was that indeed the case?RecallOut of all of the positive examples in our dataset, how many of them were identified by our model?

These questions are especially useful to ask and answer when positives and negatives are not evenly split in the data. Unlike accuracy, the intuitions of what a precision of 90% or a recall of 95% might mean scale reasonably well even if positives are just a small fraction of the overall data.

That said, it is important to notice that the metrics are in tension with each other in an interesting way. If our model does not have sufficient precision, we may be able to increase its precision by increasing the threshold it uses to make a decision. This would cause the model to say “positive” only when it is even more sure, and for reasonable models would result in higher precision. However, this would also mean that the model refrains from saying “positive” more often, meaning that it identifies fewer of the total possible number of positives and results in lower recall because of the increased precision. We could also trade off in the other direction, lowering thresholds to increase recall at the cost of less precision. This means that it is critical to consider these metrics together, rather than in isolation.

**AUC ROC**

This is sometimes just referred to as _area under the curve_ (_AUC_). _ROC_ is an abbreviation for _receiver operating characteristics_, a metric that was first developed to help measure and assess radar technology in the Second World War, but the acronym has become universally used.

Despite the confusing name, it has the lovely property of being a threshold-independent measure of model quality. Remember that accuracy, precision, and recall all rely on classification thresholds, which must be tuned. The choice of threshold can impact the value of the metric substantially, making comparisons between models tricky. AUC ROC takes this threshold tuning step out of the metric computation.

Conceptually, AUC ROC is computed by creating a plot showing the true-positive rate and the false-positive rate for a given model at every possible classification threshold, and then finding the area under that plotted curved line; see [Figure 5-3](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch05.html#an\_roc\_curve\_demonstrating\_performance) for an example. (This sounds expensive, but efficient algorithms can be used for this computation that don’t involve actually running a lot of evaluations with different thresholds.) When the area under this curve is scaled to a range from 0 to 1, this value also ends up giving the answer to the following question: “If we randomly choose one positive example and one negative from our data, what is the probability that our model gives a higher prediction score to the positive example rather than the negative?”

No metric is perfect, though, and AUC ROC does have a weakness. It is vulnerable to being fooled by model improvements that change the relative ordering of examples far away from any reasonable decision threshold, such as pushing an already low-ranked negative example even lower.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098106218/files/assets/reml_0503.png" alt="An ROC curve demonstrating performance of a model on a set of classifications" height="534" width="600"><figcaption></figcaption></figure>

**Figure 5-3. An ROC curve demonstrating performance of a model on a set of classifications**

**Precision/recall curves**

Just as an ROC curve maps out the space of trade-offs between true-positive rate and false-positive rate at different decision thresholds, many folks plot precision/recall curves that map out the space of trade-offs between precision and recall at different decision thresholds. This can be useful to get an overall sense of comparison between two models across a range of possible trade-offs.

Unlike the AUC ROC, the computed area under the precision/recall curve does not have a theoretically grounded statistical meaning, but is often used in practice as a quick way to summarize the information nonetheless. In cases of strong levels of class imbalance, there is a case to be made that the area under the precision/recall curve is a more informative metric.[3](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch05.html#ch01fn41)

#### Regression metrics

Unlike classification metrics, regression metrics do not rely on the idea of a decision threshold. Instead, they look at the raw numerical output that represents a model’s prediction, like predicted price for a given skein of yarn, number of seconds a user might spend reading a description, or the probability that a given picture contains a puppy. They are most often used when the target value itself is continuously valued, but have utility in discrete valued label settings like click-through prediction as well.

**Mean squared error and mean absolute error**

When comparing predictions from a model to a ground-truth value, the first metric we might look at is the difference between our prediction and reality. For example, in one case, our model might predict 4.3 stars for an example that had 4 stars in reality, and in another case it might predict 4.7 stars for an example that had 5 stars in reality. If we were to aggregate those values without thinking about it, so we could look at averages over many values, we would run into the mild annoyance that in the first example the difference was 0.3 and in the second it was –0.3, so our average error would appear to be 0, which feels misleading for a model that is clearly imperfect.

One fix for this is to take the absolute value of each difference—creating a metric called _mean absolute error_ (_MAE_) to average these values across examples. Another fix is to square the errors—raising them to the power of two—to create a metric called _mean squared error_ (_MSE_). Both metrics have the useful quality that a value of 0.0 shows a perfect model. MSE penalizes larger errors much more than smaller errors, which can be useful in domains where you do not want to make big mistakes. MSE can be less useful if the data contains noise or outlier examples that are better ignored, in which case MAE is likely a better metric. It can be especially useful to compute both metrics and see if they yield qualitatively different results for a comparison between two models, which can provide clues into a deeper level of understanding their differences.

**Log loss**

Some people think of _log loss_ as an abbreviation for _logistic loss_, because it is derived from the _logit_ function, which is equivalent to the log of the odds ratio between two possible outcomes. A more convenient way to think of it might be as _the loss we want to use when we think about our model outputs as actual probabilities_. Probabilities are not just numbers restricted to the range from 0.0 to 1.0, although this is an important detail. Probabilities also meaningfully describe the chance that a given thing will happen to be true.

Log loss is great for probabilities because it will highlight the difference between a prediction of 0.99, 0.999, and 0.9999, and will penalize each more confident prediction significantly more if it turns out to be incorrect—and the same thing happens at the other end of the range for predictions like 0.01, 0.001, and 0.0001. If we do indeed care about using the model outputs as probabilities, this is quite helpful. For example, if we are creating a risk-prediction model predicting the chance of an accident, there is an enormous difference between an operation being 99% reliable and 99.99% reliable—and we could end up making very bad pricing decisions if our metrics did not highlight these differences. In other settings, we might just loosely care how likely a picture is to contain a kitten, and 0.01 and 0.001 probabilities might both be best interpreted as “basically unlikely,” in which case log loss would be a poor choice of final metric. Lastly, it is important to note that log loss can give infinite values (which show up as `NaN` values and destroy averages) if our models were to predict values of exactly 1.0 or 0.0 and be in error.

## Operationalizing Verification and Evaluation

We have just taken a whirlwind tour through the world of evaluating model validity and model quality. How do we turn this knowledge into something actionable?

Assessing model validity is something that anyone who cares about production should know how to do. This is possible, even if you don’t do model evaluation daily, with a combination of training, checklists/processes, and automated support code for the simpler cases (which itself saves human expertise and judgment for more demanding cases).

For questions of model quality evaluations, things are perhaps a little more ambiguous. Obviously, it is highly useful for MLOps folks to have a working knowledge of the various distributions and metrics that are most critical for assessing model quality for our system. An organization may go through a few phases.

In the earliest days of model development for an organization, the biggest questions are often much more around getting something working rather than about how to evaluate it. This can lead to relatively coarse strategies for evaluation. For example, the main problems in developing the first version of a yarn store product recommendation model are much more likely to be around creating a data pipeline and a serving stack, and model developers might not have bandwidth to choose carefully between varying classification or ranking metrics. So our first standard evaluation might just be AUC ROC for predicting user clicks within a held-out test set.

As the organization develops, a greater understanding develops of the drawbacks or blind spots that a given evaluation might have. Typically, this results in additional metrics or distributions being developed that help shed light on important areas of model performance. For example, we might notice a cold-start problem in which new products are not represented in the held-out set, or we might decide to look at calibration and bias metrics across a range of slices by country or product listing type to understand more about our model’s performance.

At a later stage, the organization may start to go back and question basic assumptions, such as whether the chosen metrics reflect the business goals with sufficient veracity. For example, in our imaginary yarn store, we may come to realize that optimizing for clicks is not actually equivalent to optimizing for long-term user satisfaction. This may require a full reworking of the evaluation stack and careful reconsideration of all associated metrics.

Are these questions within the realm of model developers or MLOps folks? Opinions here may vary, but we believe that a healthy organization will encourage multiple points of view and rich discussions on these questions.

## Conclusion

This chapter has focused on establishing an initial viewpoint of model validity and model quality, both of which are critical to assess before moving a new version of a model into production.

Validity tests help establish that a new version of the model will not break our system. These include establishing compatibility with code and formats, and making sure that the resource requirements of computation, memory, and latency are all within acceptable limits.

Quality tests help give assurance that a new version of the model will improve predictive performance. This most often involves assessment of the model’s performance on some form of held-out or test data, with the appropriate choice of evaluation metric suited for the application task.

Together, these two forms of testing establish a decent level of trust in a model and will be a reasonable starting point for many statically trained ML systems. However, systems that deploy ML in a continuous loop will require additional verification, as detailed in [Chapter 10](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch10.html#continuous\_ml).

[1](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch05.html#ch01fn39-marker) See [“Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification”](https://oreil.ly/g37lG) by Joy Buolamwini and Timnit Gebru.

[2](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch05.html#ch01fn40-marker) See, for example, [“Noise or Signal: The Role of Image Backgrounds in Object Recognition”](https://arxiv.org/abs/2006.09994) by Kai Xiao. The [What-If Tool](https://oreil.ly/M07ff) is also an excellent example of tooling that allows for counterfactual probing.

[3](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch05.html#ch01fn41-marker) See, for example: [“Precision-Recall Curve Is More Informative Than ROC in Imbalanced Data: Napkin Math & More,”](https://oreil.ly/wZA17) by Tam D. Tran-The.

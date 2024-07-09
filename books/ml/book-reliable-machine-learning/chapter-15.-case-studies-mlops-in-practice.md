# Chapter 15. Case Studies: MLOps in Practice

This book has laid out principles and best practices for MLOps, and we’ve done our best to provide examples throughout. But there is nothing like hearing stories from folks working in the field to help see how these principles play out in the real world.

This chapter provides a set of case studies from different groups of practitioners, each detailing a specific issue, challenge, or crisis that they have lived through from an MLOps perspective. Each story was written by the practitioners themselves, so we can hear in their own words what they went through. We can see what they faced, how they dealt with it, what they learned, and what they might do differently next time. Indeed, it is striking to see how things as deceptively simple as load testing, or as seemingly unrelated as a launched update to an entirely different mobile app, can cause headaches for those in charge of daily care and feeding of ML models and systems. (Note that some of the details may have been glossed over or omitted to protect trade secrets.)

## 1. Accommodating Privacy and Data Retention Policies in ML Pipelines

By Riqiang Wang, Dialpad

### Background

The automatic speech recognition (ASR) team at Dialpad is responsible for the end-to-end speech transcription system that generates live transcripts for various AI features (collectively known as _Dialpad AI_) for our customers across the world. Various subcomponents of our AI system heavily rely on the ASR outputs to make further predictions, so any error in the transcripts gets propagated to other downstream natural language processing (NLP) tasks, such as real-time assists or named entity recognition (NER). Therefore, we continually aspire to improve the ASR models in our ML pipelines.

### Problem and Resolution

In 2020, our system was achieving great accuracy for typical North American dialects, but our benchmarking as well as anecdotal evidence showed that other dialects were often mistranscribed. As we expanded our business to other major English-speaking countries like the United Kingdom, Australia, and New Zealand, we needed to at least reach the same bar set for North American dialects. Consequently, we started looking into how to improve ASR accuracy for specific dialects, or even the plethora of dialects within North America. This included transfer learning experiments and using specialized lexicons, but on their own they were not enough. Privacy is at the center of everything we do at Dialpad, which is also a major challenge in most of the modern ML ecosystems. In this case study, we discuss a couple of challenges we’ve come across and the solutions we’ve implemented as we worked toward deploying a model for multiple dialects while respecting user privacy.

**NOTE**

In a departure from the relevant literature, we mainly use the term _dialects_ instead of _accents_ because we recognize variations beyond the accent (i.e., the sound of the speech). For example, New York and New Zealand dialects differ also in vocabulary, phrasal expressions, and even grammar. We want to ideally address all these aspects in making ASR more inclusive.

#### Challenge 1: Which dialects?

At Dialpad, we value user privacy, and to power various AI features, we need massive amounts of data for model training. Therefore, accommodations have to be made within our Dialects pipeline. Specifically, we keep as little metadata related to calls as possible, and remove calls when needed for privacy reasons.

But for training a good Dialects model, we wanted to know which of our users speak with a given dialect, be it British, Australian, or others. Thus, we needed as much metadata as feasible so we could sample accordingly for each dialect.

We first considered ideas such as letting human transcribers annotate the accent being spoken in each call, but then realized that crucially, it is extremely difficult to pinpoint accents without having experience with them, especially with non-native speakers, who are more likely to have idiolects than dialects (i.e., each speaker has their own way of speaking). Then we thought about letting users self-report dialects, but such classification self-evidently raises data privacy concerns that cut against our motivating goal of inclusiveness.

#### Solution: Get rid of the concept of dialects!

Ultimately, we realized that regardless of a given user’s dialect, we just wanted our model to do better with the speech that it was then struggling with. The ASR model did well with North American dialects because we had been feeding it North American speech, so we could also improve this existing model by adding undersampled data, building a model agnostic of dialects. We ended up simply getting more data that our model was doing poorly on, filtered by the model’s own confidence measure. We manually transcribed this underrepresented data and trained a new model with this dataset plus the original training dataset.

Within a few rounds of model tuning and evaluations, the ASR models started performing better on the underrepresented dialects test set that we manually curated, without any changes to the training techniques or model architecture. More importantly, this extra dialect dataset was only a tiny fraction of the larger original training data, but made a significant difference in performance. This shows the importance of intentionality and diversity in data collection. It also suggests that we can rely on confidence/uncertainty measures as a pseudo-diversity measure for data collection, when true diversity is difficult to measure.

#### Challenge 2: Racing the clock

Making Dialpad’s no-cost, customizable data-retention policy available as standard to all customers means that a customer can request their data to be deleted anytime, or schedule new data to be available for only a specific period of time. These substantial privacy wins, however, require equally substantial cleverness across the entire ASR system in terms of model testing and experimentation, and especially so for the Dialects ML pipeline that consists of multiple steps: collection of audio, transcription, data preparation, experimentation, and final productionization. These steps can together span over multiple quarters, longer than the lifespan of some of the collected data. That means training data and test sets are not constant, making it difficult to reproduce experimental results, sometimes leading to delays in training the models and launching the desired improvements for customers.

Late in the process of rolling out the new Dialects model, we saw that it performed well on multiple test sets, but performed significantly worse with one single test set across multiple internal trials (compared to the model in production, released six months earlier). This halted deployment of the model while we investigated why. We used multiple methods, including training the new Dialects model from scratch and checking data partitioning (after a previous misadventure inadvertently mispartitioning between training data and test data).

We also wanted to reproduce the results from the production model by using the same process to train a model, but 11 months later, the data subject to retention policies had begun expiring, and we didn’t have the exact training dataset anymore. This made it difficult to reproduce the results of past model builds, and we gained only inconclusive results. Ultimately, the key insight to resolving the discrepancy was that the previous model that had performed well on the test set was actually _in use_ during the time the data from production was taken to make the test set. Since our human transcribers create test-set transcripts by editing production model transcripts, this means that the reference transcript of this test set is biased toward the output of the old model. We will never know for sure, however, because the transient nature of data subject to arbitrary data retention regimes compounds the problem of building and maintaining a sufficient corpus for underrepresented data.

#### Solutions (and new challenges!)

We see this experience as a good step toward integrating our respect for data privacy goals with the rigor of reproducible R\&D. By the end of this project, we had created a separate, specialized data team that handles ASR or the NLP team’s data tasks, redefining our whole data collection and annotation preparation process. The data team’s task is to standardize our test-set creation process, creating dynamic test sets that provide high reproducibility even if some test data needs to be removed because of data retention policies. For example, data with time-based retention policies are no longer considered when creating a test set, and the data team also handles manual data deletions by backfilling, while monitoring how performance metrics on our test sets change over time.

The team has also standardized training data collection: instead of each ASR engineer writing their own query to get data from our database, we can now submit a request to the data team, and it will provide structured data as needed, including accounting for (and even avoiding) data flagged for deletion. As confidence in the accuracy and integrity of our human annotation pipeline improves, we are also exploring the possibility of identifying personal data elements at scale so that they could be removed or tokenized in lieu of fully deleting the transcript. While difficult, this challenge suggests a way in which privacy-promoting, data-minimizing techniques could secure much more robust access to ML training data.

### Takeaways

Integrating with privacy and data retention policies undoubtedly introduces challenges in ML pipelines, especially those powering the primary use cases of a customer-facing product/service. In our use case, working toward a more inclusive ASR model for Dialects, we first learned that even a little diversity in our training data makes the model more robust. In traditional ML practices, we tend to emphasize the size of the training data, but our results demonstrate that quality—and specifically, diversity—is irreplaceable. More importantly, we can get diversity without probing into users’ privacy by using the model confidence measure.

Secondly, although these efforts at diversity were complicated by our commitment to honoring customer choice in their data usage, we discovered that with careful curation, we could engineer robustness and reproducibility into our ML pipeline, alongside efficiency gains, by standardizing dataset creation with a dedicated team. We believe abandoning the “trade-off” narrative ultimately improves our access to needed customer data by demonstrating we are willing to put in the effort to be good stewards. Efforts like our Diversity and Dialects initiatives likewise demonstrate to customers the value of wide participation in, and representation by, ML training sets.

## 2. Continuous ML Model Impacting Traffic

By Todd Phillips, Google

### Background

Here’s a story from an incident within Google from several years ago. For confidentiality, we obscure some of the details and don’t say exactly which system was impacted, but the broad strokes of the story are still worth retelling.

The system in question included a continuous ML model that helped predict the likelihood of clicks on certain kinds of results in a search engine setting, continually updating on new data as it came in. Data came in from several sources, including web browsers and specialized apps on mobile devices. (See [Chapter 10](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch10.html#continuous\_ml) for more background on continuous ML models.)

One day, an improvement was made to one of the apps that contributed a particularly large amount of traffic to the system. As part of the improvement, code was included that asked the app to issue the most recent query a second time after an app update was made, in order to make sure that the served results had the freshest state. This improvement was pushed out to all installations of the app as one instantaneous update. For the sake of ominous foreshadowing, we will call this mass instantaneous update _issue A_.

### Problem and Resolution

What happened over the next day and a half was interesting. The moment each device in the world received the update, it reissued the most recent query. This resulted in a huge spike in traffic, but because the queries were being issued automatically, there were no additional user result clicks. This data with many more queries but no additional clicks was staged for retraining of the continuous ML model.

Because of an unrelated issue, which we will call _issue B_, the original push was rolled back. Each device that got the original update now updated back to the previous version. This, of course, caused each one to follow the protocol of reissuing the most recent query yet again, causing a third round of duplicate queries and even more data that had no associated clicks.

At this point, the continuous ML model was now happily training on all the corrupted data. Because the corrupted data included a lot of traffic with no associated clicks, the model was getting a signal that the overall click-through rate in the world was now about half of what it had been just a few hours before. The resulting learning within the model soon led to changes in the served models and, not surprisingly, lower served predictions than normal. This soon set off numerous alerts, and the model ops folks started to notice a problem with no obvious root cause because the app owners and the model owners had no visibility into each other’s systems—indeed, they were in completely different areas of a much larger organization.

Meanwhile, the unrelated issue B that had caused the rollback was fixed, and the app folks pushed the update again. This caused—you guessed it!—another round of updates on each mobile device with the app, and still yet another round of duplicate queries.

By this time, the ops folks had pushed the stop button on the continuous ML model training, and all model training was stopped. The most up-to-date version of the model was therefore one that has been impacted by the corrupted data.

Also by this time, word has gotten through the organization, and the root cause has been traced to the recent app push, but the specific cause of the changes in model predictions due to the app update was not readily apparent. The behavior that an app update caused a duplicate query was not widely known, and those who knew about it on the app side did not make the connection to the way that it could impact training data in a continuous ML model. Thus, it was assumed that the update may have contained another bug, and the decision was made to re-roll back the re-update of the app and observe for a few hours after that. And of course, this re-rollback created still yet one more round of duplicate queries and corrupted data.

Once the rollback was completed and several hours of system observations were done, there was enough information available to be sure that the problem was just in the way that the pushes were being done. The mitigation turned out to be simple: stop making updates to the app in terms of rollbacks and re-updates, and then let the continuous ML system roll forward and catch up to the new state of the world. The ML model eventually saw clean data with the appropriate click-through rates.

### Takeaways

One thing we took away from this study is that the many attempted mitigating actions actually ended up doing as much harm as good, and in some ways extended the period of impact. In retrospect, if we had just allowed the model to roll through, the system likely would have recovered much more smoothly and gracefully than it did in the wake of all our attempts to fix things. Sometimes it pays to just hold tight and roll through.

## 3. Steel Inspection

By Ivan Zhou, Landing AI

### Background

Manufacturers in many industries rely on visual inspections to detect critical defects during production of steel rolls. I am an ML engineer at Landing AI and wrote this case study to show some data-centric techniques we used to develop deep learning models for visual inspection tasks.

Recently, my team and I worked on a steel inspection project ([Figure 15-1](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch15.html#this\_case\_study\_focuses\_on\_detecting\_de)). The customer had been developing a visual inspection model for years and had it running in production. But their models achieved only 70% to 80% accuracy. I was able to rapidly prototype a new deep-learning model that achieved 93% accuracy at detecting defects in the project.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098106218/files/assets/reml_1501.png" alt="This case study focuses on detecting defects in steel rolls that may have occurred in production" height="421" width="600"><figcaption></figcaption></figure>

**Figure 15-1. This case study focuses on detecting defects in steel rolls that may have occurred in production**

The goal of this project was to accurately classify the defects among the customer’s hot rolling dataset; _hot rolling_ is a key stage in the production pipeline of steel rolls. The defects were spread across 38 classes, and many defect classes had only a couple hundred examples.

The customers had been working on this problem for almost 10 years, and the best performance their model was able to achieve was only 80% accuracy, which was not sufficient for the client’s needs. Over the years, the customers had tried to bring on several other AI teams to improve the accuracy of their models. All attempted to improve the performance by architecting several state-of-the-art models, but ultimately none were able to make any improvements.

### Problem and Resolution

I went onsite to work on this project. I hired three local interns to help me label those images. In the first week, I spent almost all of my time learning about defect classes, managing the labeling work, and reviewing their labels. I gave data to interns in small batches. Every time they finished, I would review their labels and share with them feedback if there was a labeling error.

We didn’t label all data at once. We labeled 30% of the data per class in the first iteration, pinpointed all the ambiguities, addressed them, and then labeled the next 30%. So we had three iterations of labeling over two weeks. We focused on defects that might be introduced in the “roll” stage of the manufacturing pipeline, after the metal is cast but before it is finished. Defects can occur from a variety of physical conditions in the hot process, and are grouped by category. In the end, we labeled 18,000 images and threw away more than 3,000 that we thought were confusing (Figures [15-2](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch15.html#details\_of\_the\_hot\_rolling\_setting) and [15-3](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch15.html#timeline\_for\_data\_labelingcomma\_label\_r)).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098106218/files/assets/reml_1502.png" alt="Details of the hot rolling setting" height="800" width="521"><figcaption></figcaption></figure>

**Figure 15-2. Details of the hot rolling setting**

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098106218/files/assets/reml_1503.png" alt="Timeline for data labeling, label review, and model training" height="193" width="600"><figcaption></figcaption></figure>

**Figure 15-3. Timeline for data labeling, label review, and model training**

One of the challenges that took us lots of time was to manage and update the defect consensus. Out of 38 defect classes, many pairs of classes looked very similar at first glance, so they easily confused the labelers. We had to constantly discuss ambiguous cases when disagreement occurred, and we had to update our defect definitions to maintain defect consensus among three labelers and ML engineers. For example, can you tell there are three distinctive defect classes from the nine images in [Figure 15-4](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch15.html#visually\_identifying\_the\_distinctions\_a)?

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098106218/files/assets/reml_1504.png" alt="Visually identifying the distinctions among the three classes (black line, black stripe, and scratch) from nine images is far from trivial" height="255" width="600"><figcaption></figcaption></figure>

**Figure 15-4. Visually identifying the distinctions among the three classes (black line, black stripe, and scratch) from nine images is far from trivial**

So here are the answers. After we saw more samples of these three classes, we could continuously correct the boundaries between these three defect types and update their defect definitions ([Figure 15-5](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch15.html#three\_defect\_types\_with\_updated\_definit)). We spent lots of effort for labelers to maintain a defect consensus. For samples that were really hard to identify, we had to remove them from our training dataset.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098106218/files/assets/reml_1505.png" alt="Three defect types with updated definitions" height="351" width="600"><figcaption></figcaption></figure>

**Figure 15-5. Three defect types with updated definitions**

Besides the defect definition, it was also critical to establish labeling consensus. The labelers were not only expected to tell defect classes accurately, but since we were doing object detection, we also wanted their bounding boxes labeling to be tight and consistent.

For example, the samples shown in [Figure 15-6](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch15.html#the\_third\_image\_shows\_a\_well\_annotated) were from a defect class called _roller iron sheet_, which featured very dense holes or black dots. When labelers labeled the images, they were expected to draw tight bounding boxes around all areas with clear patterns of defects. If discontinuity occurred, they needed to annotate with separate boxes, like the third image ([Figure 15-6](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch15.html#the\_third\_image\_shows\_a\_well\_annotated)). However, the fourth image was rejected during labeling reviewing, because the box was too wide and loosely covered a defective area. If we allowed this label to be added to our training set, it would mislead the model when calculating the losses, and we should avoid that.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098106218/files/assets/reml_1506.png" alt="The third image shows a well-annotated discontinuity, while the fourth image was rejected during labeling review because the box too loosely covers a defective area" height="376" width="600"><figcaption></figcaption></figure>

**Figure 15-6. The third image shows a well-annotated discontinuity, while the fourth image was rejected during labeling review because the box too loosely covers a defective area**

### Takeaways

We spent less than 10% of our time doing model iterations. After each time we trained a model, I spent most of my time reviewing falsely predicted examples and identified root causes of errors. Then I took those insights back to further clean the dataset. After two iterations like this, we achieved 93% accuracy on the test set, or a 65% reduction in error rate. This far exceeded the baseline and the expectations that the customers had at that time and met their needs.

## 4. NLP MLOps: Profiling and Staging Load Test

By Cheng Chen, Dialpad

### Background

Dialpad’s AI team builds NLP applications to help users get more from their calls, including real-time transcript formatting, sentiment detection, action-item extraction, and more. Developing and deploying large NLP models systems is a challenge. Fitting them within the constraints of real-time, cost-effective performance makes that challenge significantly more complex.

In 2019, the large language model BERT achieved state-of-the-art NLP performance.[1](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch15.html#ch01fn145) We planned to leverage it to provide more accurate NLP capabilities, including punctuation restoration and date, time, and currency detection. To reduce cloud cost, however, our real-time production environment has very limited resources assigned to it (GPU is not an option, and we have one CPU at most for many models). In addition, our hard limit on model inference is 50 ms per utterance. Meanwhile, the BERT base model has 12 layers of transformer blocks with 110 million parameters. We knew it would be challenging to optimize it to fit into our real-time environment, but we still overlooked one critical piece: the difficulty of obtaining an accurate estimate on how much faster the model has to be to meet our real-time demand.

### Problem and Resolution

Our team needed to perform local profiling in order to benchmark various NLP models, which ran model inference over a large number of randomly sampled utterances and calculated average inference time per utterance.

Once the average inference speed met a fixed threshold, the packaged model would be handed over to our data engineering (DE) team, which would then do canary deployment in our Google Kubernetes Engine (GKE) cluster, and monitor a dashboard of various real-time metrics ([Figure 15-7](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch15.html#a\_dashboard\_monitoring\_real\_time\_metric)) with an ability to drill down into specific metrics ([Figure 15-8](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch15.html#drill\_down\_to\_a\_specific\_metric\_for\_a\_g)).

This was how we gradually deployed a new BERT-based punctuator model (whose goal is to restore punctuation) into production, and this was where the confusion started. For a large language model based on BERT, often DE teams discovered that the latency or queue time bumped up significantly and they had to roll back the deployment. Clearly, the local profiling configuration was not aligned with the actual pattern occurring in the production system. This discrepancy may come from two sources: cluster compute resource allocation and/or traffic pattern. However, the reality was that scientists didn’t have the right tools to properly benchmark model inference. This was resulting in time-consuming deployments and rollbacks with lots of wasted effort on repeated manual work. Add to that the anxiety about deploying a potentially underperforming model into the system, with the resulting system congestion and service outages. As a stopgap measure, applied scientists and engineers agreed to increase the compute resource, such as adding one more CPU for inference, but we clearly needed a better approach to benchmarking.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098106218/files/assets/reml_1507.png" alt="A dashboard monitoring real-time metrics" height="619" width="600"><figcaption></figcaption></figure>

**Figure 15-7. A dashboard monitoring real-time metrics**

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098106218/files/assets/reml_1508.png" alt="Drill-down to a specific metric for a given model" height="200" width="600"><figcaption></figcaption></figure>

**Figure 15-8. Drill-down to a specific metric for a given model**

#### An improved process for benchmarking

What we needed was a way to allow NLP applied scientists to efficiently obtain benchmarking results that were close to production metrics. (Note that canary deployment was still required in production deployment.)

Apart from the production system, the DE team also maintained a staging environment where NLP models were deployed and integrated with the product interface (reference) prior to production deployment. Our QA team made test calls to test various call features, and applied scientists leveraged this environment to ensure that the model ran properly with the product UI. However, they had not used it to thoroughly benchmark large models.

The DE team proposed a comprehensive and self-serve load-test tool to help applied scientists benchmark model inference in the staging environment. When designing the load-test tool, we kept the following high-level points in mind:

* Load-test data should contain trigger phrases for the model.
* Load-test data should contain a healthy mix of utterance lengths. It is probably better to have longer utterances so as to give a better approximation of how the system will perform under stress.
* Load-test data should trigger model inference and not get short-circuited by optimizations/caches that would lead to misleadingly low runtime latencies.
* We use CircleCI workflows to control automatic deployments to staging.
* (Optional) Load-test data should have similar characteristics to data expected in production.

After the tool was developed, applied scientists had two options for performing a load test on staging:

* Audio-based (end-to-end) load test
  * This is a full end-to-end test.
  * This simulates calls on the system.
  * Data is sampled automatically from calls in the staging environment (e.g., QA calls) and tries to provide good coverage on certain features.
  * This audio can be customized so we can put NLP-specific audio datasets if needed.
* Text-based (model specific) load test
  * This targets only a single microservice (e.g., the punctuator or the sentiment model).
  * This allows us to pick the most difficult inputs to stress-test our models.

After scientists decided on the type of load test and had all the necessary data in place, they then deployed the changes to staging and started the load test.

Once the load test began, scientists could monitor the live dashboard for important metrics such as the Runtime 95th, as shown in [Figure 15-9](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch15.html#runtime\_ninefiveth\_percentile). That is the most significant value when evaluating inference speed. As a rule of thumb, anything below 1 second satisfies the requirement. Currently, most models tend to be clustered at or below 0.5 seconds.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098106218/files/assets/reml_1509.png" alt="Runtime 95th percentile" height="209" width="600"><figcaption></figcaption></figure>

**Figure 15-9. Runtime 95th percentile**

With this tool in place, scientists could launch staging tests themselves, without asking for help from the DE team. The Datadog dashboard also provided a comprehensive breakdown of the runtime performance of each model so that applied scientists could monitor the metric numbers more closely. Therefore, the load-test tool significantly reduced communication overhead during our rapid development cycle.

### Takeaways

Cramming state-of-the-art NLP performance into a resource-constrained real-time production environment requires very high confidence that benchmarks during testing will be borne out in production. When our testing methods started failing on the more resource-intensive BERT model, we reached into our staging environment to give our scientists a more representative environment to test against and made it self-serve so they could iterate rapidly. The automatic staging benchmarking step has since become a standard process in the model development process. Both teams are now relieved as applied scientists are able to obtain close-to-production estimates on model inference with great confidence.

## 5. Ad Click Prediction: Databases Versus Reality

By Daniel Papasian, Google

### Background

Google’s ad-targeting systems aim to help maximize the long-term value of shown ads, which includes minimizing the frequency of displaying unwanted ads to users. They do so in part by using models to predict the probability that a given ad will be clicked. When there are opportunities to show ads, an auction is conducted, and as part of this auction a server uses a model to predict the probability that certain ads will be clicked. These probabilities are one of several inputs to the auction, and if the model underperforms, both user and advertiser experience suffers. The act of displaying the ad results in us inserting a row to our database. The row in the database corresponds to an ad being shown to the user, with columns associated with the features used for model training. Additionally, a Boolean column represents whether the ad resulted in a click. This column is defaulted to `false` when the row is inserted.

If the ad is clicked, it generates a record in a click log. The click-logging team runs a process to remove fraudulent clicks, and publishes a feed of “clean” clicks for consumption by other internal users. This feed is used to issue an update to the already created row in the database to mark the ad impression as having resulted in a click.

The model was trained by looking at the rows in this database as examples, and using the click bit as the label. The rows in the data were the raw inputs to the model and the label for each event recording either “resulted in a click” or “was not clicked.” If we were to never update the models, they would work well for some time, but eventually degrade in accuracy because of changing user behaviors and advertiser inventories. To improve the overall behavior, we automated the retraining and deployment of the click prediction model.

Before a retrain is pushed to production, we validate that we are improving the model’s accuracy. We hold back a portion of the dataset from training and use it as a test set. Our training process handled this with Bernoulli sampling: for each ad shown in the 48 hours before training, there would be a 99% chance we’d train on it, and a 1% chance we would reserve it for our test set. This was implemented with a `test_set` bit on the row that we’d set to `true` 1% of the time. When the training system read the events table, it would filter out all rows where this was true. Newly trained models would be sent to this validation system, which generated inferences on recent events with the `test_set` bit. It compared these inferences to the observed labels to generate statistics about model accuracy. Models would be pushed to production only if the new models performed better than the old model over the recent test set events.

### Problem and Resolution

One Monday, we came in and were greeted by an automatic alert: we were showing ads at a rate far below normal. We quickly realized that our mean predicted probability of a click was one-tenth of what it typically was. Whether we show an ad at all was gated in part on how probable we thought it was that the ad would be clicked. As such, the widespread underprediction of click probability explained the alert on the rate of ads being shown. But questions still remained: did our users’ behavior change? Or had our model update somehow hurt us, despite our validation measures?

We queried the database for all rows in the 48 hours before the training cutoff. No matter how we aggregated it, we saw click rates that were astoundingly typical. The model was acting as if clicks were far more rare, but the data in our database didn’t reflect that. But why didn’t our validation system block the model from going to production? We tasked both our data science and production engineering teams to dig into the situation to understand what happened.

The data science team started by looking at the validation system: this was supposed to keep us from pushing out models that performed worse than the versions we were replacing. The validation system computed a loss metric by generating inferences over the test set. Lower loss was supposed to mean better models. The logs from Sunday’s validation run indicated we processed the test events as expected, and that the loss statistic was lower for the new model than our old model. With a hunch, someone decided to rerun the validation system with the same pair of models across the test set. The test set was reread from the database, and the inferences were generated as expected. This time, the loss metric indicated the new model was worse than the old model—the opposite result from Sunday. What changed?

The production engineering teams checked a range of data from a large set of systems, trying to see whether any unexplained anomalies were in relevant systems. Curiously, a graph showed revenue of $0 for Wednesday through Sunday, and then a spike to very large amounts of revenue in the early hours of Monday. The graph was produced by a system that watched the feed of verified clicks.

When the production engineer and data scientist teams conferred with each other and shared their findings, they realized the model was underpredicting because of a failure of the infrastructure responsible for processing the raw click logs and distributing the clean click feed to consumers. The clean clicks arrived to the ML training system late—not until early Monday morning, after the model had last trained. Without any evidence to the contrary, the model believed that every ad shown in this period didn’t result in a click. Every event was a true negative or a false negative, and that’s all our test set contained as well. The model that we trained concluded that the ads were awful and no one would click on any of them, which was accurate given the data that we had when we trained the model. When the click-processing feed caught up, the validation data was relabeled so that ads that resulted in clicks were labeled that way. This explained why subsequent attempts at validation of the already trained model were still failing. The issue was resolved by retraining on the corrected data.

### Takeaways

In retrospect, it’s important to note that our model never directly predicted the probability of a click. Rather, it predicted the probability of a shown ad being marked as “was clicked” in our database at the time that training happens. Indeed, while we expect that our database is an accurate reflection of reality, bugs or failures may cause differences. In this case, a production failure in an upstream system caused the meaning of this label to diverge from what we wished it to reflect. Our models built using supervised learning techniques predict labels in our training set, and it’s of critical importance that our labels reflect reality.

The teams collaborated to write a postmortem to analyze what happened and how to prevent it. This turned out to be a significant learning experience for our organization. We gathered the timeline: from the perspective of people working on the click prediction models, the problem wasn’t detected until Monday. We later learned that the team that worked on the click logs noticed their pipeline was broken on Wednesday with a software release, and were aware of the problem the same day when their alerting indicated the feed wasn’t being processed. They put in place mitigations to ensure that clicks would still be eventually billed, and figured they’d fix the rest of the data-feed processing first thing on Monday. They hadn’t realized their system was a data dependency of an ML process downstream, and how that system was making assumptions about the completeness of data. We believe many ML pipelines make assumptions about the completeness of input data and correctness of provided labels without verifying these assumptions, and as such are at risk for similar problems.

We listed every potential cause of the outage we could muster; we knew we’d prioritize these based on effort to make an improvement and expected value of the improvement. The causes included the lack of integration testing leading to the click logs breaking, the training system’s reliance on the click log processing being more reliable than agreed-upon service availability targets, and our assumption that the most recent events would be representative of all events.

Our follow-ups included establishing an availability target for the click log processing systems, expanding our validation system to check that the test set’s positive ratio wasn’t suspiciously low or high, and establishing a process for the click log team to communicate outages and pause training if serious problems with model health occurred.

## 6. Testing and Measuring Dependencies in ML Workflow

By Harsh Saini, Dialpad

### Background

At Dialpad, we have a speech recognition and processing engine that has several ML dependencies. Audio comes in from our telephony backend and gets transcribed in real time through our proprietary ASR models, where formatting and readability improvements are also made. The output is then fed into our language-specific NLP models like NER and sentiment analysis. This pipeline can be simplified as a flowchart ([Figure 15-10](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch15.html#flowchart\_of\_dialpad\_speech\_recognition)).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098106218/files/assets/reml_1510.png" alt="Flowchart of Dialpad speech-recognition and processing pipeline" height="60" width="600"><figcaption></figcaption></figure>

**Figure 15-10. Flowchart of Dialpad speech-recognition and processing pipeline**

However, in reality, the pipeline is not as straightforward as shown in this simplified diagram. Multiple speech-recognition models may be used, depending on the user’s location and/or the product line they are using within Dialpad. For instance, a user from the UK using the call-center product line will be provided with a speech-recognition model fine-tuned on the UK dialect of English and trained on call-center domain-specific knowledge. Similarly, a user from the US using the sales product line will have their call transcribed using a speech-recognition model trained on the US English dialect and domain-specific knowledge for sales calls.

Additionally, for the NLP, several task-specific models run in parallel to perform tasks such as sentiment analysis, question detection, and action-item identification. With this in mind, the simplified flowchart can be extended to highlight the diversity of models in Dialpad’s production ML pipeline.

### Problem and Resolution

[Figure 15-11](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch15.html#some\_of\_the\_ml\_dependencies\_that\_exist) highlights some of the ML dependencies that exist for NLP task-specific models with regards to the upstream speech-recognition models. NLP model performance is sensitive to ASR model output artifacts. While most NLP models are not overly sensitive to minor changes to ASR model outputs, over time the data changes significantly enough, and NLP models experience a degradation in performance due to regression and data drift. A few common updates to ASR that result in a change in input data distribution for NLP models are as follows:

* Modifications in the vocabulary of the ASR system (e.g., the addition of the word _coronavirus_)
* Changes in the output of the ASR system (e.g., people are saying the same things, but we’re getting better at accurately transcribing them)
* Topic drift, whereby people are actually talking about different things (e.g., suddenly everyone starts talking about elections in the US)

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098106218/files/assets/reml_1511.png" alt="Some of the ML dependencies that exist for NLP task-specific models" height="232" width="600"><figcaption></figcaption></figure>

**Figure 15-11. Some of the ML dependencies that exist for NLP task-specific models**

To combat this phenomenon, the DE team at Dialpad, in collaboration with the data science team, built an offline testing pipeline that could measure NLP model performance for a given ASR model.

#### Building the regression-testing sandbox

Some of the key requirements for our regression testing and monitoring system were as follows:

* Ensure that monitoring of NLP model performance would happen automatically whenever newer ASR models are released.
* Simulate the behavior as observed in production by the models.
* Collect and report metrics submitted via the evaluation that can be viewed by stakeholders.
* Collect model inference artifacts and logs so as to assist in troubleshooting by scientists.
* Allow for ad hoc evaluation by data science teams when they wish to evaluate a model prerelease.
* Ensure that we could establish comparable baselines, since datasets for evaluation could be modified out of band.
* Be a scalable system so that multiple evaluations could occur simultaneously, and also not be bottlenecked as we increase either the dataset sizes or the number of models being tested.

Given these requirements, the following design decisions were made:

* Kubeflow Pipelines (KFP) was chosen as the platform to host the sandbox:
  * KFP allows users to write custom directed acyclic graphs (DAGs) called _pipelines_.
  * Each pipeline is sandboxed, and the platform as a whole can independently scale to the demands of all running pipelines.
  * The engineering teams at Dialpad are heavily invested in [Kubernetes](https://kubernetes.io/) and [Argo Workflows](https://argoproj.github.io/workflows), which are the underlying technology powering KFP, so it seemed prudent to use this platform.
* The pipelines in KFP will build the correct infrastructure for evaluation by selecting the correct model deployment artifacts, given the evaluation criteria.
  * This will be done on the fly and will not be persisted to reduce cost.
  * The testbed will be decoupled from model versions and be aware of the order of dependencies only for correct orchestration.
* Outputs from every model will be persisted for 30 days for debugging purposes.
* Datasets for every task-specific NLP model would be versioned so as to track changes in evaluation data.
* Metrics will be collected for every combination of ASR model version, NLP model version, and dataset version.
  * This ensures that we can disambiguate among different dependencies correctly.
  * These metrics would then be visualized in a dashboard for observability.
* The input to the testing pipeline is raw audio recordings of conversations, since the idea was to capture whether an ASR model has changed in such a way that it alters the output enough that the downstream NLP model has varied performance.
  * Once audio samples were collected, they would be annotated to state whether they contain a specific NLP moment. For instance, a given audio snippet would be annotated by a human to verify whether it contained positive, negative, or neutral sentiment for the sentiment analysis task.
  * As you can see, this is an arduous task and is still one of the biggest bottlenecks for this project. It is extremely time-consuming to correctly slice, annotate, and store such samples for every NLP task ([Figure 15-12](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch15.html#the\_regression\_testing\_environment\_on\_a)).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098106218/files/assets/reml_1512.png" alt="The regression-testing environment at a high level" height="363" width="600"><figcaption></figcaption></figure>

**Figure 15-12. The regression-testing environment at a high level**

And within KFP, a pipeline would simulate evaluation for a single combination of ASR model version, NLP model version, and dataset version. Since KFP allows us to run multiple pipelines in parallel, this would allow us to scale to all combinations of evaluation we would like to perform ([Figure 15-13](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch15.html#the\_architecture\_of\_the\_kfp\_pipeline\_as)).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098106218/files/assets/reml_1513.png" alt="The architecture of the KFP pipeline as a DAG" height="308" width="600"><figcaption></figcaption></figure>

**Figure 15-13. The architecture of the KFP pipeline as a DAG**

#### Monitoring for regression

Once the pipelines were built on KFP, the next part of the project was to automatically perform regression tests whenever dependencies changed for NLP models. Luckily, at Dialpad we have mature CI/CD workflows managed by engineering, and they were updated to trigger KFP pipelines whenever ASR models were updated in the transcription service. The CI/CD workflow would send a signal to KFP with information about the ASR models, NLP models, etc., and the evaluation would then commence on KFP. Metrics would be stored, and Slack messages would be emitted containing a summary of the evaluation.

Once operational, this process captures performance evaluation data for all NLP task-specific models that have testing data available on the platform. For example, the F1-score of the NLP sentiment analysis model degraded by \~25% over the course of a year, as shown in [Figure 15-14](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch15.html#fone\_score\_of\_the\_nlp\_sentiment\_analysi); the graph highlights the absolute difference from a baseline. This observation alerted the NLP team to investigate the issue and discover that accumulated data drift was the cause of the degradation. A new sentiment model was retrained using the latest ASR model output and released to production in just a few months.

Another tangential benefit of this process is that it allows for ad hoc evaluation of NLP models against different ASR models prior to production release. For instance, it is possible to measure the accuracy of a sentiment analysis model, prior to release, against an ASR model trained on new English dialects, such as Australian or New Zealand English.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098106218/files/assets/reml_1514.png" alt="F1-score of the NLP sentiment analysis model" height="391" width="600"><figcaption></figcaption></figure>

**Figure 15-14. F1-score of the NLP sentiment analysis model**

### Takeaways

This ML regression-testing platform developed at Dialpad has provided data scientists and engineers with much improved visibility on the impact of new model releases on all dependent components in our production stack. Even with an incomplete knowledge of all the deployed production models, people are able to understand whether a proposed release is going to impact the stability and performance of other models in the production pipeline. This reduces the chances of a rollback and can provide an early indication if more work needs to be done to improve compatibility with existing components.

The testing platform is under active development. Other moving pieces are being addressed, one of which is keeping the sandbox orchestration in sync with production and allowing for other “live data” that only transiently lives during a call on production and is difficult to simulate in the regression-testing platform. Another feature being considered is how to provide automated alerting when a proposed release has significant impact on downstream models rather than the current human-in-the-loop approach.

[1](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch15.html#ch01fn145-marker) See the 2019 paper [“BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”](https://oreil.ly/WEh8t) by Jacob Devlin et al.

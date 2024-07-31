---
description: Normalized discounted cumulative gain
---

# NDCG

Ref: [https://towardsdatascience.com/demystifying-ndcg-bee3be58cfe0#:\~:text=models%20in%20production.-,What%20Is%20NDCG%20and%20Where%20Is%20It%20Used%3F,or%20other%20information%20retrieval%20system.](https://towardsdatascience.com/demystifying-ndcg-bee3be58cfe0)

## 1. What is NDCG

NDCG is a measure of ranking quality. This is used to evaluate the performance of a search engine, recommendation, or other information retrieval system.

The value of NDCG is determined by comparing the relevance of the items returned by the search engine to the relevance of the item that a hypothetical “ideal” search engine would return.

## 2. Example

Here is a simple version dataset for a ranking model. There are two different search queries: **x** and **y**. Within each group, there are five different items shown as the result of search and each item has rank based on the position they are at the result list. Lastly, there are gains for each item representing the relevance of each item within the search.

<figure><img src="../../../.gitbook/assets/image (8) (1).png" alt=""><figcaption></figcaption></figure>

### 2.1 Cumulative Gain (CG)

Cumulative Gain is a sum of _gains_ associated for items within a search query. Here is the formula for it:

<figure><img src="../../../.gitbook/assets/image (9) (1).png" alt=""><figcaption></figcaption></figure>

Using the dataset above, we can calculate CG for each group:

<figure><img src="../../../.gitbook/assets/image (10).png" alt=""><figcaption></figcaption></figure>

In this example, both groups have the same CG — 3 — so we are still not able to tell which search groups are better. In order to do that, we need to take consideration of rank in the formula — which brings us into the next part: DCG.

### 2.2 Discounted Cumulative Gain (DCG)

DCG is the same concept as CG but takes the additional step of discounting the gains by rank. Here is the formula for DCG:

<figure><img src="../../../.gitbook/assets/image (11).png" alt=""><figcaption></figcaption></figure>

Using the dataset above, we can calculate DCG for each group:

<figure><img src="../../../.gitbook/assets/image (12).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../../.gitbook/assets/image (13).png" alt=""><figcaption></figcaption></figure>

Good news! Now we can see the DCG of _**y**_ is better than the DCG of _**x**_. It also makes sense that group _**y**_ has better DCG because the items in the higher rank are more relevant (higher gain) to the search group _**y**_. So why do we still need NDCG? To answer this question, let’s introduce another search groups _**z**_ to the count example:

<figure><img src="../../../.gitbook/assets/image (14).png" alt=""><figcaption></figcaption></figure>

Then, let’s practice DCG calculation one more time:

<figure><img src="../../../.gitbook/assets/image (15).png" alt=""><figcaption></figcaption></figure>

The DCG of _**z**_ is 1, but it has the most relevant item at the first rank. If we compare the data, it should be at least better than group _**x**_. The problem is group _**x**_ has three relevant items and group _**z**_ only has one, and it’s not fair to just compare the DCG since it’s cumulative sum.

This is how NDCG comes into play since it normalizes the DCG before comparing — but the problem is how to normalize it to make a fair comparison. For this task, we need IDCG.

### 2.3 Ideal Discounted Cumulative Gain (IDCG)

IDCG stands for **ideal discounted cumulative gain**, which is calculating the DCG of the ideal order based on the gains.

when a user searches for something online they always want to have the most relevant item at the top and above any irrelevant items. That is, all the relevant information should always be at the top, and it should have the best DCG.

Using IDCG for each search group from the dataset above:

<figure><img src="../../../.gitbook/assets/image (16).png" alt=""><figcaption></figcaption></figure>

### 2.4 Normalized Discounted Cumulative Gain (NDCG)

NDCG normalizes the DCG by the IDCG of the group. It can be interpreted as the comparison of the actual relevance order and the ideal relevance order.

<figure><img src="../../../.gitbook/assets/image (17).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../../.gitbook/assets/image (18).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../../.gitbook/assets/image (19).png" alt=""><figcaption></figcaption></figure>

With this, we can confidently say group _**z**_ has the best NDCG. It also makes sense that all its relevant items are at the top of the list.

## 3. What is NDCG@K?

_K_ means the top _K_ ranked item of the list, and only top _K_ relevance contributes to the final calculation.

Now, let’s calculate the NDCG@3 for the group **x**:

<figure><img src="../../../.gitbook/assets/image (20).png" alt=""><figcaption></figcaption></figure>

## 4. How does NDCG compare to other ranking metrics?

* **NDCG** (**normalized discounted cumulative gain)**: NDCG is a measure of the effectiveness of a ranking system, taking into account the position of relevant items in the ranked list. It is based on the idea that items that are higher in the ranking should be given more credit than items that are lower in the ranking. NDCG is calculated by dividing the discounted cumulative gain (DCG) of the ranked list by the DCG of the ideal ranked list, which is the list with the relevant items ranked in the most optimal order. _NDCG ranges from 0 to 1, with higher values indicating better performance._
* **MAP (mean average precision)**: [mean average precision](https://arize.com/blog-course/ndcg/) is a measure of the precision of a ranking system, taking into account the number of relevant items in the ranked list. It is calculated by averaging the precision at each position in the ranked list, where precision is defined as the number of relevant items in the list up to that position divided by the total number of items in the list up to that position. _MAP ranges from 0 to 1, with higher values indicating better performance._
* **MRR (mean reciprocal rank)**: MRR is a measure of the rank of the first relevant item in a ranked list. It is calculated by taking the reciprocal of the rank of the first relevant item, and averaging this value across all queries or users. For example, if the first relevant item for a given query has a rank of 3, the MRR for that query would be 1/3. _MRR ranges from 0 to 1, with higher values indicating better performance._

## 5. What does alow NDCG value in production mean?

The example below shows what happens when the performance of a recommendation system in production starts to decline. In the image inset, you may notice that the training and production datasets are nearly identical, with only the first and last recommendations switched in the production dataset. This results in a significant difference in the performance between the two datasets, dropping NDCG from 0.993 to 0.646. NDCG is the most sensitive rank-aware metric to overall graded order and is favorable for cases when you can receive full relevance feedback.

<figure><img src="../../../.gitbook/assets/image (21).png" alt=""><figcaption></figcaption></figure>






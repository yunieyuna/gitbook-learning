# Assessment Test

1. You are migrating your machine learning operations to GCP and want to take advantage of managed services. You have been managing a Spark cluster because you use the MLlib library extensively. Which GCP managed service would you use?
   1. Cloud Dataprep
   2. Cloud Dataproc
   3. Cloud Dataflow
   4. Cloud Pub/Sub
2. Your team is designing a database to store product catalog information. They have determined that you need to use a database that supports flexible schemas and transactions. What service would you expect to use?
   1. Cloud SQL
   2. Cloud BigQuery
   3. Cloud Firestore
   4. Cloud Storage
3. Your company has been losing market share because competitors are attracting your customers with a more personalized experience on their e-commerce platforms, including providing recommendations for products that might be of interest to them. The CEO has stated that your company will provide equivalent services within 90 days. What GCP service would you use to help meet this objective?
   1. Cloud Bigtable
   2. Cloud Storage
   3. AI Platform
   4. Cloud Datastore
4. The finance department at your company has been archiving data on premises. They no longer want to maintain a costly dedicated storage system. They would like to store up to 300 TB of data for 10 years. The data will likely not be accessed at all. They also want to minimize cost. What storage service would you recommend?
   1. Cloud Storage multi-regional storage
   2. Cloud Storage Nearline storage
   3. Cloud Storage Coldline storage
   4. Cloud Bigtable
5. You will be developing machine learning models using sensitive data. Your company has several policies regarding protecting sensitive data, including requiring enhanced security on virtual machines (VMs) processing sensitive data. Which GCP service would you look to for meeting those requirements?
   1. Identity and access management (IAM)
   2. Cloud Key Management Service
   3. Cloud Identity
   4. Shielded VMs
6. You have developed a machine learning algorithm for identifying objects in images. Your company has a mobile app that allows users to upload images and get back a list of identified objects. You need to implement the mechanism to detect when a new image is uploaded to Cloud Storage and invoke the model to perform the analysis. Which GCP service would you use for that?
   1. Cloud Functions
   2. Cloud Storage Nearline
   3. Cloud Dataflow
   4. Cloud Dataproc
7. An IoT system streams data to a Cloud Pub/Sub topic for ingestion, and the data is processed in a Cloud Dataflow pipeline before being written to Cloud Bigtable. Latency is increasing as more data is added, even though nodes are not at maximum utilization. What would you look for first as a possible cause of this problem?
   1. Too many nodes in the cluster
   2. A poorly designed row key
   3. Too many column families
   4. Too many indexes being updated during write operations
8. A health and wellness startup in Canada has been more successful than expected. Investors are pushing the founders to expand into new regions outside of North America. The CEO and CTO are discussing the possibility of expanding into Europe. The app offered by the startup collects personal information, storing some locally on the user’s device and some in the cloud. What regulation will the startup need to plan for before expanding into the European market?
   1. HIPAA
   2. PCI-DSS
   3. GDPR
   4. SOX
9. Your company has been collecting vehicle performance data for the past year and now has 500 TB of data. Analysts at the company want to analyze the data to understand performance differences better across classes of vehicles. The analysts are advanced SQL users, but not all have programming experience. They want to minimize administrative overhead by using a managed service, if possible. What service might you recommend for conducting preliminary analysis of the data?
   1. Compute Engine
   2. Kubernetes Engine
   3. BigQuery
   4. Cloud Functions
10. An airline is moving its luggage-tracking applications to Google Cloud. There are many requirements, including support for SQL and strong consistency. The database will be accessed by users in the United States, Europe, and Asia. The database will store approximately 50 TB in the first year and grow at approximately 10 percent a year after that. What managed database service would you recommend?
    1. Cloud SQL
    2. BigQuery
    3. Cloud Spanner
    4. Cloud Dataflow
11. You are using Cloud Firestore to store data about online game players’ state while in a game. The state information includes health score, a set of possessions, and a list of team members collaborating with the player. You have noticed that the size of the raw data in the database is approximately 2 TB, but the amount of space used by Cloud Firestore is almost 5 TB. What could be causing the need for so much more space?
    1. The data model has been denormalized.
    2. There are multiple indexes.
    3. Nodes in the database cluster are misconfigured.
    4. There are too many column families in use.
12. You have a BigQuery table with data about customer purchases, including the date of purchase, the type of product purchases, the product name, and several other descriptive attributes. There is approximately three years of data. You tend to query data by month and then by customer. You would like to minimize the amount of data scanned. How would you organize the table?
    1. Partition by purchase date and cluster by customer
    2. Partition by purchase date and cluster by product
    3. Partition by customer and cluster by product
    4. Partition by customer and cluster by purchase date
13. You are currently using Java to implement an ELT pipeline in Hadoop. You’d like to replace your Java programs with a managed service in GCP. Which would you use?
    1. Data Studio
    2. Cloud Dataflow
    3. Cloud Bigtable
    4. BigQuery
14. A group of attorneys has hired you to help them categorize over a million documents in an intellectual property case. The attorneys need to isolate documents that are relevant to a patent that the plaintiffs argue has been infringed. The attorneys have 50,000 labeled examples of documents, and when the model is evaluated on training data, it performs quite well. However, when evaluated on test data, it performs quite poorly. What would you try to improve the performance?
    1. Perform feature engineering
    2. Perform validation testing
    3. Add more data
    4. Regularization
15. Your company is migrating from an on-premises pipeline that uses Apache Kafka for ingesting data and MongoDB for storage. What two managed services would you recommend as replacements for these?
    1. Cloud Dataflow and Cloud Bigtable
    2. Cloud Dataprep and Cloud Pub/Sub
    3. Cloud Pub/Sub and Cloud Firestore
    4. Cloud Pub/Sub and BigQuery
16. A group of data scientists is using Hadoop to store and analyze IoT data. They have decided to use GCP because they are spending too much time managing the Hadoop cluster. They are particularly interested in using services that would allow them to port their models and machine learning workflows to other clouds. What service would you use as a replacement for their existing platform?
    1. BigQuery
    2. Cloud Storage
    3. Cloud Dataproc
    4. Cloud Spanner
17. You are analyzing several datasets and will likely use them to build regression models. You will receive additional datasets, so you’d like to have a workflow to transform the raw data into a form suitable for analysis. You’d also like to work with the data in an interactive manner using Python. What services would you use in GCP?
    1. Cloud Dataflow and Data Studio
    2. Cloud Dataflow and Cloud Datalab
    3. Cloud Dataprep and Data Studio
    4. Cloud Datalab and Data Studio
18. You have a large number of files that you would like to store for several years. The files will be accessed frequently by users around the world. You decide to store the data in multi-regional Cloud Storage. You want users to be able to view files and their metadata in a Cloud Storage bucket. What role would you assign to those users? (Assume you are practicing the principle of least privilege.)
    1. roles/storage.objectCreator
    2. roles/storage.objectViewer
    3. roles/storage.admin
    4. roles/storage.bucketList
19. You have built a deep learning neural network to perform multiclass classification. You find that the model is overfitting. Which of the following would not be used to reduce overfitting?
    1. Dropout
    2. L2 Regularization
    3. L1 Regularization
    4. Logistic regression
20. Your company would like to start experimenting with machine learning, but no one in the company is experienced with ML. Analysts in the marketing department have identified some data in their relational database that they think may be useful for training a model. What would you recommend that they try first to build proof-of-concept models?
    1. AutoML Tables
    2. Kubeflow
    3. Cloud Firestore
    4. Spark MLlib
21. You have several large deep learning networks that you have built using TensorFlow. The models use only standard TensorFlow components. You have been running the models on an n1-highcpu-64 VM, but the models are taking longer to train than you would like. What would you try first to accelerate the model training?
    1. GPUs
    2. TPUs
    3. Shielded VMs
    4. Preemptible VMs
22. Your company wants to build a data lake to store data in its raw form for extended periods of time. The data lake should provide access controls, virtually unlimited storage, and the lowest cost possible. Which GCP service would you suggest?
    1. Cloud Bigtable
    2. BigQuery
    3. Cloud Storage
    4. Cloud Spanner
23. Auditors have determined that your company’s processes for storing, processing, and transmitting sensitive data are insufficient. They believe that additional measures must be taken to prevent sensitive information, such as personally identifiable government-issued numbers, are not disclosed. They suggest masking or removing sensitive data before it is transmitted outside the company. What GCP service would you recommend?
    1. Data loss prevention API
    2. In-transit encryption
    3. Storing sensitive information in Cloud Key Management
    4. Cloud Dataflow
24. You are using Cloud Functions to start the processing of images as they are uploaded into Cloud Storage. In the past, there have been spikes in the number of images uploaded, and many instances of the Cloud Function were created at those times. What can you do to prevent too many instances from starting?
    1. Use the --max-limit parameter when deploying the function.
    2. Use the --max-instances parameter when deploying the function.
    3. Configure the --max-instance parameter in the resource hierarchy.
    4. Nothing. There is no option to limit the number of instances.
25. You have several analysis programs running in production. Sometimes they are failing, but there is no apparent pattern to the failures. You’d like to use a GCP service to record custom information from the programs so that you can better understand what is happening. Which service would you use?
    1. Stackdriver Debugger
    2. Stackdriver Logging
    3. Stackdriver Monitoring
    4. Stackdriver Trace
26. The CTO of your company is concerned about the rising costs of maintaining your company’s enterprise data warehouse. The current data warehouse runs in a PostgreSQL instance. You would like to migrate to GCP and use a managed service that reduces operational overhead and one that will scale to meet future needs of up to 3 PB. What service would you recommend?
    1. Cloud SQL using PostgreSQL
    2. BigQuery
    3. Cloud Bigtable
    4. Cloud Spanner

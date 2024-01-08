# Label Studio

Local Test

### 0. Overall

* Each annotation task is separated as different project
  * Example:
    * Image Annotation → Project 1
    * Text Annotation → Project 2
    * OCR Annotation → Project 3

## 1. Image (Bounding Box)

* Can work with different marks at the same time
  * At the example below, images with different marks can be mixed together and annotated at the same time. &#x20;

<figure><img src="../../.gitbook/assets/image (22).png" alt=""><figcaption></figcaption></figure>

### **Export Data**

<figure><img src="../../.gitbook/assets/image (24).png" alt=""><figcaption></figcaption></figure>

JSON

<figure><img src="../../.gitbook/assets/image (25).png" alt=""><figcaption></figcaption></figure>

CSV

<figure><img src="../../.gitbook/assets/image (26).png" alt=""><figcaption></figcaption></figure>

YOLO

* classes: contains all the label information
* images: contains all the original images
* labels: contains annotation information - object class, coordinates, height & width

<figure><img src="../../.gitbook/assets/image (27).png" alt=""><figcaption></figcaption></figure>

## 2. Text

Tips: Pages can scroll up\&down so text with long contents should have no problem.

<figure><img src="../../.gitbook/assets/image (28).png" alt=""><figcaption></figcaption></figure>

Cautions

* Target column that need annotation should use "text" as column name otherwise it won't show on the page.

<figure><img src="../../.gitbook/assets/image (29).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (30).png" alt=""><figcaption></figcaption></figure>

**Output**

<figure><img src="../../.gitbook/assets/image (31).png" alt=""><figcaption></figcaption></figure>

JSON

<figure><img src="../../.gitbook/assets/image (32).png" alt=""><figcaption></figcaption></figure>

CSV

<figure><img src="../../.gitbook/assets/image (33).png" alt=""><figcaption></figcaption></figure>

#### OCR

* Bounding Box + Text are necessary

<figure><img src="../../.gitbook/assets/image (34).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (35).png" alt=""><figcaption></figcaption></figure>

**Output**\


<figure><img src="../../.gitbook/assets/image (36).png" alt=""><figcaption></figcaption></figure>

JSON\


<figure><img src="../../.gitbook/assets/image (37).png" alt=""><figcaption></figcaption></figure>

CSV\


<figure><img src="../../.gitbook/assets/image (38).png" alt=""><figcaption></figcaption></figure>

YOLO

❌ No text data in the output file (only class and coordinate info)



## GCP Deployment

On the official github page, there is a button for cloud instance deployment.&#x20;

<figure><img src="../../.gitbook/assets/image (209).png" alt=""><figcaption></figcaption></figure>

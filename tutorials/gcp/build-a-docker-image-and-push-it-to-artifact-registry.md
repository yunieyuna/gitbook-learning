# Build a Docker Image and Push it to Artifact Registry

In this tutorial, you will get to hands-on of building a docker image and push to the gcp artifact registry. The tutorial will use the following web application repo as example.

\> [label-studio](https://github.com/HumanSignal/label-studio)

## Step 1. Build a Docker Image

### 1. Create a dockerfile if not exist&#x20;

Here is the [official documentation of generating a dockerfile](https://docs.docker.com/get-started/02\_our\_app/#build-the-apps-image). Since there is one already exists in the example repo. We will use it directly.

### 2. Build a docker image

<figure><img src="../../.gitbook/assets/image (188).png" alt=""><figcaption></figcaption></figure>

In your terminal, navigate to the directory containing your Dockerfile. Use the following command to build your Docker image, replacing `Dockerfile.cloudrun` with the name of your Dockerfile, if necessary.

```
$docker build -t label-studio -f Dockerfile.cloudrun .
```

The output is:

<figure><img src="../../.gitbook/assets/image (189).png" alt=""><figcaption></figcaption></figure>

### 3. Check the images you have locally

```
$docker images
```

<figure><img src="../../.gitbook/assets/image (190).png" alt=""><figcaption></figcaption></figure>

## Step 2. Log in to Google Cloud

Use the following command to log in to your Google Cloud account:

```
gcloud auth login
```

And:

```
gcloud auth application-default login
```

## Step 3. Create Repository on Artifact Registry

Artifact Registry: [https://console.cloud.google.com/artifacts/browse/ccbd-ecbdp-bds?project=ccbd-ecbdp-bds](https://console.cloud.google.com/artifacts/browse/ccbd-ecbdp-bds?project=ccbd-ecbdp-bds)

**1. Click Create Repository**

<figure><img src="../../.gitbook/assets/image (191).png" alt=""><figcaption></figcaption></figure>

**2. Add repo name and choose region as "asia-northeast1 (Tokyo)"**

<figure><img src="../../.gitbook/assets/image (192).png" alt=""><figcaption></figcaption></figure>

## Step 4. Configure Docker to authenticate to Artifact Registry

Use the following command to configure Docker to authenticate to the Artifact Registry in the **`asia-northeast1`** region:

<figure><img src="../../.gitbook/assets/image (193).png" alt=""><figcaption></figcaption></figure>

## Step 5: Tag your Docker image

Use the following command to tag your Docker image for the Artifact Registry, replacing `1a429cdb3f06` with your Docker image ID:

```
docker tag 1a429cdb3f06 asia-northeast1-docker.pkg.dev/ccbd-ecbdp-bds/label-studio-test/label-studio:latest
```

<figure><img src="../../.gitbook/assets/image (194).png" alt=""><figcaption></figcaption></figure>

## Step 6. Push your Docker image to the Artifact Registry

Finally, use the following command to push your Docker image to the Artifact Registry:

```
docker push asia-northeast1-docker.pkg.dev/ccbd-ecbdp-bds/label-studio-test/label-studio:latest
```

<figure><img src="../../.gitbook/assets/image (195).png" alt=""><figcaption></figcaption></figure>

After that, you are able to see the docker image on the repository just created.

<figure><img src="../../.gitbook/assets/image (196).png" alt=""><figcaption></figcaption></figure>

## Reference

* [Store Docker container images in Artifact Registry](https://cloud.google.com/artifact-registry/docs/docker/store-docker-container-images#linux)


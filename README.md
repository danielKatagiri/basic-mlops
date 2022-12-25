# basic-mlops
This repo describes how to create a docker image for a ML model. For the full description
check my medium: https://medium.com/@izumi.katagiri/understanding-the-basic-mlops-principle-for-data-scientists-256b8e5f83b1

# Running the code

## Create a new virtual environment
```commandline
python3 -m venv .env 
```

## Training the model
To train and create the model artifacts run
```commandline
python3 -m train_titanic
```

## Building and running the docker image
```commandline
docker build -t titanic .
docker run --rm -it -p 8080:8080 titanic
```
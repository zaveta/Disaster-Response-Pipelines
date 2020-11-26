# Disaster Response Pipeline Project

### Description
In this project, I applied data engineering, natural language processing, and machine learning skills to analyze message data that people sent during disasters to build a model for an API that classifies disaster messages.

Data Processing, ETL Pipeline to extract data from the source, clean data, and save them in a proper database structure.
Machine Learning Pipeline to train and tunning a model able to classify text messages into appropriate categories.
Web App to show model results in real-time.

### File Description
* Jupiter_Notebooks_Pipline_Preparation
  * ETL Pipeline Preparation.ipynb
  * ML Pipeline Preparation.ipynb
  
* data
  * disaster_categories.csv: dataset including all the categories
  * disaster_messages.csv: dataset including all the messages
  * process_data.py: ETL pipeline scripts to read, clean, and save data into a database

* models
  * train_classifier.py: machine learning pipeline scripts to train and export a classifier

* app
  * run.py: Flask file to run the web application
  * templates contains html file for the web application

### Requirements
  * Python 3
#### to run the web app
  * [pandas](https://github.com/pandas-dev/pandas)
  * [nltk](https://github.com/nltk/nltk)
  * [flask](https://github.com/pallets/flask)
  * [plotly](https://github.com/plotly/plotly.py)
  * [sqlalchemy](https://github.com/sqlalchemy/sqlalchemy)

#### to use the pipelines
  * [numpy](https://github.com/numpy/numpy)
  * [sklearn](https://github.com/scikit-learn/scikit-learn)

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Licensing
Apache License 2.0

See the LICENSE file for details

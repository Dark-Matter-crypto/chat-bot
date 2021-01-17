# MGP-Bot

A Django and Python chatbot webapp. Users can ask the bot any number of questions about me. Trained the model with Tensorflow 2, based of a series of questions people ask me on a regular.

![](/repoImages/chatbot.jpg)

### Tech Stack:
* Django 3.1.5
* Tensorflow 2.4.0
* Python 3.8.7
* Bootstrap

### How to run locally:

1. Set up virtual environment:
```
    mkdr chatbot
    cd chatbot
    pip install virtualenv
    python -m venv virtual
    source virtual/Scripts/activate
```


2. Clone project and install requirements:
```
    git clone 'repo_address'
    cd chatbot
    pip install -r requirements.txt
```

3. Migrate and run project:
```
    python manage.py migrate
    python manage.py runserver
```
    



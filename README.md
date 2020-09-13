# Know-Your-Fruit-Public

Welcome to the codebase for Know Your Fruit, available at https://kyfr.herokuapp.com/, : A Python web application written in Flask and hosted on Heroku that displays fruit nutrition, storage, and other information given only the picture of a fruit! After some initial preprocessing, it sends the picture off to Google Cloud Platform (GCP) for AI Platform Prediction, where a Convolutional Neural Network (CNN) has been trained to classify over 30 types of fruit. It uses the Xception network available in Tensorflow and Keras that was trained using transfer learning and image augmentation on hundreds of fruit pictures available on the web. The fruit information was scraped from https://fruitsandveggies.org/ using Selenium.

For security reasons, the actual repository for the application is hidden. This repo has two main folders, one to hold utility scripts and notebooks for creating, testing, and saving the model, and another for the majority of the files that make up the Heroku server. 


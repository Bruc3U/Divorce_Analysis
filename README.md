# 💒 Divorce Analysis Project

![iStock_72610687_MEDIUM](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/54df4e63-4212-4e10-bc92-60d6cfe3acae)

# Python Libraries Used

- pandas
- matplotlib
- numpy
- scikit-learn

## Objective

This project's goal is to determine the main reasons for divorce according to the given dataset. 

## About the dataset

The dataset regroups the records of 170 couples. Those couples were asked the same 53 questions and were given a 5-point scale to answer them.

- 0=never
- 1=seldom
- 2=Averagely
- 3=frequently
- 4=always

The sample contained both married and divorced individuals marked by a 1 for married and a 0 for divorced. 

The data was extracted via the UC Irvine Machine Learning Repository and can be found [here](http://archive.ics.uci.edu/dataset/497/divorce+predictors+data+set).

It is important to note any limitations given by the data itself. In order to find a viable and realistic solution to the problem we must analyze the context.

| Dataset Limitation | 
|---|
| No real information on the couples day to day life, limited context|
| The panel is from a different country (Turkey) |
| The questions had to be translated | 

The panel being from a different country could affect the reason for divorce since the cultural expectation is different. In addition to that, no other context was given for those couples, we do not know their age, their psychological background, their family situation...<br>
Before drawing any conclusion we must refer to the context of the data to avoid any unrealistic conclusions. 

# I/Defining the goal:

Marriage represents a huge chunk of our Western civilization. Marriage can be defined as the legally and formally recognized union of two people as partners in a personal relationship. 
Despite a long history, nowadays nearly 50% of those sacred unions end up in a divorce. 

Why do people get divorced so much?

| The 5 most common reasons for divorce | 
|---|
| 1. Lack of Commitment|
| 2. Infidelity |
| 3. Too much conflict | 
| 4. Getting married too young |
| 5. Financial issues | 

Source [Insider](https://www.insider.com/why-people-get-divorced-2019-1) 

Our goal will be to shed some light on the matter and figure out the main reasons to divorce thanks to machine learning. 

# II/Data Wrangling: 

After setting the right directory and importing the right libraries.

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/d1bbaf9a-fd58-4f8a-9a5b-60dc31f17d86)


The first step is to check for any missing values and duplicates. 

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/23e5a9bd-2776-4164-866c-d6446d92dbbd)

No, duplicates or missing value was found. We will move on to the next step.

# III/ Analysis:
### A/Logistic Regression:

Since the problem is binary, 1 for married and 0 for divorced. 

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/d2334344-5121-49b1-885b-4f6179ef3871)




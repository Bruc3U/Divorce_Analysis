# 💒 Divorce Analysis Project

![iStock_72610687_MEDIUM](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/54df4e63-4212-4e10-bc92-60d6cfe3acae)

# Summary
- Python Libraries Used
- Objective
- About the dataset
- I/Defining the goal
- II/Data Wrangling
- III/ Analysis
- Conclusion

# Python Libraries Used

- pandas
- matplotlib
- numpy
- scikit-learn
- statsmodel

## Objective

Our goal is to determine the main predictors for divorce according to the given dataset. 

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

The panel being from a different country could affect the reason for divorce since the cultural expectations are different. In addition to that, no other context was given for those couples, we do not know their age, their psychological background, their family situation...<br>
Before drawing any conclusion we must refer to the context of the data to avoid any unrealistic diagnosis. 

# I/Defining the goal:

Marriage represents a significant part of our Western civilization.
Marriage can be defined as the legally and formally recognized union of two people as partners in a personal relationship. 
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

Our goal will be to shed some light on the matter and figure out the main reasons for divorce thanks to machine learning. 

# II/Data Wrangling: 

After setting the right directory and importing the right libraries.

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/d1bbaf9a-fd58-4f8a-9a5b-60dc31f17d86)


The first step is to check for any missing values and duplicates. 

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/23e5a9bd-2776-4164-866c-d6446d92dbbd)

No, duplicates or missing value was found. We will move on to the next step.

# III/ Analysis:
### A/Logistic Regression:

The nature of our data is binary, 1 for married and 0 for divorced, we will use a logistic regression for our first model. 

After splitting the data into a train/test format.

The dependent variable will be the column Class, which represents the current marital status of our couples.<br>
The rest of the columns will be our independent variables. 

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/b44d356e-1cb0-4a02-967b-426ab49069e5)


Accuracy: 96.64%


It is pretty accurate but the main issue is the number of features, we do have 53 independent variables.<br>
It could lead to overfitting. 


### B/Principal Component Analysis 

Principal Component Analysis (PCA), is a technique used to reduce the number of features.<br>
PCA requires a set number of components. Those components act as 'compressed' data for easier processing. 

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/8ca8bc59-3ae5-44d2-b042-f5ba47714f33)

The number of components will be two. 

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/4bee1f21-eb1e-4572-b207-47746b69d202)

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/9c8958ee-cf9a-4ba6-8d00-03fee27440ad)


As we can see, 75% of the data is found in the first component.<br>
Having more than two components will have diminished returns.

Let's re-run our logistic regression model with reduced Xs.

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/64ae2811-3ccd-4162-b1c5-0a97498df5d9)


 Accuracy: 96.64%

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/228b1cce-209b-400d-9ba9-151846cff010)

Regarding the confusion matrix, we can observe that our model is highly accurate.<br>
64 True positive and 51 True negative. With only 4 False negatives.


 ### C/Random Forest Classifier 

Let's try another algorithm to see if we can improve our previous model.

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/551073e2-afb5-401b-9c51-4999a96e1a8b)

Accuracy: 96.64%

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/7bbf7bd0-8197-4ce8-8596-72614d35c080)

The random forest classifier gave us the same accuracy rate and a similar confusion matrix.<br>
We will be using this model for our final analysis. 


# Conclusion

After computing several models, we can finally draw conclusions and answer our questions.

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/49396bba-23af-4d48-b060-b5be666672bd)

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/3a157633-7601-4dc2-af99-215d70711c35)

According to our analysis, we can observe that more than half of the questions asked did not have a great impact on the final results.<br>
Only a few questions had a real impact on the predictability of divorce.

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/74a0edd8-2a14-45ca-8556-78675b3a3645)

We can observe two clusters. The first cluster represents the spread of married couples. A tight cluster, in this case, tells us the data did a better job at predicting married couples.<br>
On the other hand, the second cluster seems to be sparse and less grouped. This reflects the data limitation previously stated.<br>
Indeed, we do not have any additional information about the context of those couples. We do not know their age, history, job, social status...<br>
This limits our analysis greatly and is reflected in our final thoughts.


| The 5 most common predictors for divorce according to the dataset | 
|---|
| 1. My spouse and I have similar values that we trust|
| 2. My spouse and I have similar ideas about how marriage should be |
| 3. We have a compatible view about what love should be | 
| 4. I think that one day when I look back, I will remember that I and my spouse have been in harmony with each other |
| 5. I know exactly what my spouse's interests are | 

If we compare our findings with the insiders, we can conclude that the data is heading in the right direction.<br>
Even with a different culture and the lack of context our results make sense and are believable.<br>
But for realistic uses, predictors such as relationship lengths, number of kids, current job, and social status need to be added to the model to ensure long-term reliability. 





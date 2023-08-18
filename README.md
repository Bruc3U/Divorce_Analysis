# 💒 Divorce Analysis Project

![iStock_72610687_MEDIUM](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/54df4e63-4212-4e10-bc92-60d6cfe3acae)

# Python Libraries Used

- pandas
- matplotlib
- numpy
- scikit-learn
- statsmodel

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

The nature of our data is binary, 1 for married and 0 for divorced, we will use a logistic regression for our first model. 

After splitting the data into a train/test format.

The dependent variable will be the column Class, which represents the current marital status of our couples.<br>
The rest of the columns will be our independent variables. 

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/d2334344-5121-49b1-885b-4f6179ef3871)

Accuracy: 96.64%


It is pretty accurate but the main issue is the number of features, we do have 53 independent variables.<br>
It could lead to overfitting the data. 


### B/Principal Component Analysis 

Principal Component Analysis (PCA), is a technique used to reduce the number of features.<br>
PCA requires a set number of components. Those components act as 'compressed' data for easier processing. 

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/8ca8bc59-3ae5-44d2-b042-f5ba47714f33)

The number of components will be two. 

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/4bee1f21-eb1e-4572-b207-47746b69d202)

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/e860d7de-8182-4dc0-b80a-3dcbc04cceff)

As we can see, 75% of the data is found in the first component.<br>
Having more than two components will have diminished returns.

Let's re-run our logistic regression model with reduced Xs.

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/b2eea6ea-501c-475e-90f8-a4370e770bcb)

 Accuracy: 96.64%

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/228b1cce-209b-400d-9ba9-151846cff010)

Regarding the confusion matrix, we can observe that our model is highly accurate.<br>
64 True positive and 51 True negative. With only 4 False negatives.


 ### C/Random Forest Classifier 

Let's try another algorithm to see if we can improve our previous model.

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/f00386b9-4f89-4beb-9b18-e5fac84a67c3)

Accuracy: 96.64%

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/7a1aa6c6-a570-45ae-8498-912a4c465caf)


The random forest classifier gave us the same accuracy rate and a similar confusion matrix.<br>
We will be using this model for our final analysis. 


# Conclusion

After computing several models, we can finally draw conclusions and answer our questions.

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/49396bba-23af-4d48-b060-b5be666672bd)

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/3a157633-7601-4dc2-af99-215d70711c35)

According to our analysis, we can observe that more than half of the questions asked did not have a great impact on the final results.<br>
Only a few questions had a real impact on the predictability of divorce.

![image](https://github.com/Bruc3U/Divorce_Analysis/assets/142362478/74a0edd8-2a14-45ca-8556-78675b3a3645)

With the married cluster being tight we can conclude that the questions really captured the predictability of such actions.<br>
On the other hand, the second cluster, the divorced one seems to be sparse and less grouped. This reflects the data limitation previously stated.<br>
Indeed, we do not have any additional information about the context of those couples. We do not know their age, history, job, social status...<br>
This limits our analysis greatly and is reflected in our final thoughts.


| The 5 most common predictors for divorce according to the dataset | 
|---|
| 1. My spouse and I have similar values that we trust|
| 2. My spouse and I have similar ideas about how marriage should be |
| 3. We have a compatible view about what love should be | 
| 4. I think that one day when I look back, I will remember that I and my spouse have been in harmony with each other |
| 5. I know exactly what my spouse's interests are | 

If we compare our findings with the insiders, we can conclude that the data is heading in the right direction regarding predicting divorce.<br>
Even with a different culture and the lack of context our results make sense and are believable.<br>
But for use in the real world, predictors such as relationship lengths, number of kids, current job, and social status need to be added to the model to ensure long-term reliability. 






# Logistic Regression
It uses conventional gradient decent approach for the convergence of the model and exploits the titanic dataset to calculate the probability of survivals.  

## 1.1) Dataset 
Our dataset includes 711 samples in training dataset while 177 samples in test dataset. Each row in the dataset has seven columns (first 6 columns are features and last column is the class label). First column represents class (1st, 2nd or 3rd), second column represents male or female (0 for male and 1 for female), third column is age , fourth column is siblings/spouses aboard, fifth column is parents/children aboard, fifth column is fare and last column is class label (1 for survived and 0 for died).
not
## 1.2) Training and Testing the Model
After performing 1 lac iterations the model converges at __0.002__ learning rate with training accuracy __80.28%__ while test accuracy accuracy is __78%__.

## 1.3) Analyzing the Model
By analyzing the weight vector, we can say that higher the value of class (1st feature), we have less probability of survival.Similarly, regarding gender (2nd feature); females have more probability of survival as compare to males. 3rd learned feature shows that younger people have more chance of survival as compare to poor people. If there are a smaller number of siblings/spouses and parents/children abroad, we have more chances of survival. Lastly, people who paid high fares have more chances of survival as compare to people who paid less fares.

Hence, we can deduce that  1st class younger women passengers having no relatives abroad and paid higher amount of money for the fare are most likely to survive. 3rd class older
men passengers with relatives abroad and paid lesser amount of money for the fare are most like to die.
The most influencial features of this problem are Class and Gender, which can be seen below in graphical form:-



![alt text](https://github.com/mohsin123321/machine-learning/blob/master/projects/logistic%20regression/images/distribution.jpg?raw=true)


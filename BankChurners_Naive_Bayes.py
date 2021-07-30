import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

bank_churn = pd.read_csv(('BankChurners.csv'), usecols=["Customer_Age", "Months_on_book", "Months_Inactive_12_mon", "Credit_Limit" ])

bank_churn


X_churn = bank_churn.iloc[:, 0:4].values
X_churn
X_churn[0]

Y_churn = bank_churn.iloc[:, 4].values
Y_churn

#Label Encoder

label_encoder_labelch = LabelEncoder()

X_churn[:,1]

labelch = label_encoder_labelch.fit_transform(X_churn[:,1])


X_churn[0]

label_encoder_Customer_Age = LabelEncoder()
label_encoder_Months_on_book = LabelEncoder()
label_encoder_Months_Inactive_12_mon = LabelEncoder()
label_encoder_Credit_Limit = LabelEncoder()


X_churn[:,3] = label_encoder_Customer_Age.fit_transform(X_churn[:,3])
X_churn[:,10] = label_encoder_Months_on_book.fit_transform(X_churn[:,10])
X_churn[:,9] = label_encoder_Months_Inactive_12_mon .fit_transform(X_churn[:,12])  
X_churn[:,14] = label_encoder_Credit_Limit.fit_transform(X_churn[:,14])


X_churn[0]

X_churn

#Standard Scaler

scaler_churn = StandardScaler()
X_churn = scaler_churn.fit_transform(X_churn)

X_churn[0]

#Training and test

X_churn_training, X_churn_test, Y_churn_training, Y_churn_test = train_test_split(X_churn, Y_churn, test_size = 0.25, random_state = 0)

X_churn_training.shape, Y_churn_training.shape

X_churn_test.shape, Y_churn_test.shape

#Naive Bayes

naive_bank_churners = GaussianNB()
naive_bank_churners.fit(X_churn, Y_churn)

churn_nb = naive_bank_churners.predict(X_churn_test)

churn_nb 

naive_bank_churners.classes
naive_bank_churners.class_count_
naive_bank_churners.class_prior_  
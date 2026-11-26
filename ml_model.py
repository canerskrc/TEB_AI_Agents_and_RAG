import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X = np.array([
    [30000,5000,0],
[45000,8000,0],
[60000,1000,0],
[80000,1500,1],
[50000,12000,0],
[40000,10000,3],
[25000,25000,5],
[30000,5000,0],
])
# 0 : dusuk risk 1: yuksek risk
y = np.array([0,0,0,0,1,1,1,1])

model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))

model.fit(X, y)

def predict_risk(
        income: float,
        debt:float,
        late_payments:int
):
    print(income,debt,late_payments)
    if income <= 0 or debt < 0 or late_payments < 0:
        return {
            'status':'error',
            'message':' Gecersiz finansal veri.'
        }

    customer = np.array([[income, debt, late_payments]])
    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]
    return {
        'risk_class':(
            'Yuksek risk' if prediction == 1 else 'Dusuk risk'
        ),
        'risk_score':round(float(probability),4),
        'education_demo':True
    }

if __name__ == '__main__':
    result = predict_risk(
        income=45000,
        debt=10000,
        late_payments=3
    )
    print(result)

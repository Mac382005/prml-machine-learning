How to reuse the model:
-
import joblib
pipe = joblib.load('final_model.joblib')
y_pred = pipe.predict(X_new)
y_prob = pipe.predict_proba(X_new)[:,1]

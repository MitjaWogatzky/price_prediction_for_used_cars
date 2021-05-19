import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def benchmark(model, dataset, n_runs=100):
    X = dataset.data
    y = dataset.target
    
    train_scores = []
    test_scores = []
    
    for run in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        train_scores.append(train_score)
        test_scores.append(test_score)
        
    return np.mean(train_scores), np.mean(test_scores)
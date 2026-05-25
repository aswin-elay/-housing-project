from sklearn.ensemble import RandomForestRegressor 
 
def train_model(X_train, y_train): 
    model = RandomForestRegressor(n_estimators=200, max_depth=10)  
    model.fit(X_train, y_train) 
    return model 

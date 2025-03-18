# Compute RMSE for the white-box model
def white_box_model(x, R=0.1, K=0.05):
    theta_a = x[:, 3]  # ambient temperature
    theta_or = x[:, 4]  # delta top oil
    theta_hr = x[:, -2]  # heat run test y (assumption)
    x_param = x[:, -3]  # heat run test x (assumption)
    y_param = x[:, -1]  # heat run test gradient (assumption)

    white_box_pred = ((1 + R * K**2) / (1 + R)) ** x_param * (theta_or - theta_a) + K**y_param * (theta_hr - theta_or)
    return white_box_pred


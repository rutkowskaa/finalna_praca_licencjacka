def MSE_cross_val(preds, prog):
    actual = data[prog:]
    mse = (1 / len(preds)) * sum((actual - preds) ** 2)
    return mse
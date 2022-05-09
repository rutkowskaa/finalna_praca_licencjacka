module Loss

function MSE(preds, prog, Ys)
    actual = Ys[prog:length(Ys)]
    mse = (1 / length(preds)) * sum((actual - preds).^ 2)
    return float(mse)
end

function MSE_inv(preds, prog, Ys)
    actual = Ys[prog:length(Ys)]
    inv = actual .* preds
    deleteat!(inv, findall(x->x>0, inv))
    mse = (1 / length(preds)) * sum((actual - preds).^ 2) - sum(inv)
    return mse
end

function SMAPE(preds, prog, Ys)
    actual = Ys[prog:length(Ys)]
    return 1/length(actual) * sum(2 * abs.(preds-actual) / (abs.(actual) + abs.(preds))*100)
end

function RMSE(preds, prog, Ys)
    return sqrt(MSE_cross_val(preds, prog, Ys))
end

function MSLE(preds, prog, Ys)
    actual = Ys[prog:length(Ys)]
    mse = (1 / length(preds)) * sum((log.(actual) - log.(preds)).^ 2)
    return mse
end

function RMSLE(preds, prog, Ys)
    return sqrt(MSLE(preds, prog, Ys))
end

function SSE(preds, prog, Ys)
    actual = Ys[prog:length(Ys)]
    return (actual - preds) .^ 2
end

#Y = [1, 2, -3, 4, -5, 6]
#pred = [1, -2, 3, 4, 5, -6]

#print(MSE_cross_val(pred, 1, Y))
#print(MSE_inv_cross_val(pred, 1, Y))

end
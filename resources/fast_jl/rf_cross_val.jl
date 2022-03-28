
#using Pkg
#
#Pkg.add("DecisionTree")
#using DecisionTree

function rf_cross_val(dict)

    dlugosc_okna = dict["dlugosc_okna"]
    prog = dict["prog"]
    Ys = dict["data"]
    X = dict["X"]
    println("SHAPE", size(X))
    X = reshape(X, 88, 2)
    params = dict["params"]
    leng = length(Ys)
    print(params)

    function MSE_cross_val(preds, prog)
        actual = Ys[prog:length(Ys)]
        mse = (1 / length(preds)) * sum((actual - preds).^ 2)
        return mse
    end

    all_preds = []

        for depth = 2: params["max_depth"]
            for n_estimator = 1: params["max_n_estimators"]
                for sample = 2: params["min_samples_split"]
                    for leaf = 2: params["min_samples_leaf"]

                        pred = []
                        println("$depth, $n_estimator, $sample, $leaf")
                        for i = prog: leng
                            dolny = i - prog + 1
                            gorny = i

                            train_y = Ys[dolny : gorny]
                            train_x = X[dolny : gorny, :]

                            train_x = reshape(train_x, 44, 2)

                            model = DecisionTree.build_forest(train_y,
                                                              train_x,
                                                              n_trees=n_estimator,
                                                              max_depth=depth,
                                                              min_samples_split=sample,
                                                              min_samples_leaf=leaf

                            )


                            prediction = apply_forest(model, train_x[1, :])

                            push!(pred, prediction)

                        end
                        all_preds = append!(all_preds, [depth, n_estimator, sample, leaf, MSE_cross_val(pred, prog)])

                    end
                end
            end
        end
        bledy = reshape(all_preds, (length(params) + 1), Int(length(all_preds)/(length(params) + 1)))
        print("ALL ", all_preds)
        mm = findmin(bledy[(length(params) + 1),:])
        minn = mm[2]
        result = bledy[:, minn]

        to_ret = Dict(
            "depth" => result[1],
            "n_estimators"=> result[2],
            "min_samples_split"=> result[3],
            "min_samples_leaf"=> result[4]
        )

        return to_ret
end
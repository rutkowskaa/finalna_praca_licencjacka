include("Loss.jl")

function vector_rf_cross_val(dict)

    dlugosc_okna = dict["dlugosc_okna"]
    prog = dict["prog"]
    Ys = dict["data"]
    X = dict["X"]

    wymiary = size(X)
    X = reshape(X, wymiary[1], wymiary[2])
    params = dict["params"]
    leng = length(Ys)

    all_preds = []

        for depth = 2: params["max_depth"]
            for n_estimator = 2: params["n_estimators"]
                for sample = 2: params["min_samples_split"]
                    for leaf = 2: params["min_samples_leaf"]

                        pred = []
                        println("$depth, $n_estimator, $sample, $leaf")
                        for i = prog: leng
                            dolny = i - prog + 1
                            gorny = i - 1

                            train_y = Ys[dolny : gorny]
                            train_x = X[dolny : gorny, :]

                            train_x = reshape(train_x, gorny - dolny + 1, wymiary[2])
                            #print(train_x)
                            model = DecisionTree.build_forest(train_y,
                                                              train_x,
                                                              -1,
                                                              n_estimator,
                                                              1,
                                                              depth,
                                                              leaf,
                                                              sample,
                                                              0.0;
                                                              rng = 3
                            )


                            prediction = apply_forest(model, X[gorny + 1, :])

                            push!(pred, prediction)

                        end
                        all_preds = append!(all_preds, [depth, n_estimator, sample, leaf, Loss.MSE(pred, prog, Ys)])

                    end
                end
            end
        end

        bledy = reshape(all_preds, length(params)+1, :)
        tylko_bledy = bledy[length(params)+1, :]

        indeks_najmniejszego_bledu = findmin(tylko_bledy)[2]
        result = bledy[:, indeks_najmniejszego_bledu]

        to_ret = Dict(
            "max_depth" => Integer(result[1]),
            "n_estimators"=> Integer(result[2]),
            "min_samples_split"=> Integer(result[3]),
            "min_samples_leaf"=> Integer(result[4])
        )

        return to_ret
end

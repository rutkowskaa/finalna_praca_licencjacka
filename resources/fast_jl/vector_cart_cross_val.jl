include("Loss.jl")

function vector_cart_cross_val(dict)

    dlugosc_okna = dict["dlugosc_okna"]
    prog = dict["prog"]
    Ys = dict["data"]
    X = dict["X"]
    wymiary = size(X)

    X = reshape(X, wymiary[1], wymiary[2])

    print(size(X))
    params = dict["params"]
    leng = length(Ys)


    function MSE_cross_val(preds, prog)
        actual = Ys[prog:length(Ys)]
        mse = (1 / length(preds)) * sum((actual - preds).^ 2)
        return mse
    end

    all_preds = []

        for depth = 2: params["max_depth"]
                for sample = 2: params["min_samples_split"]
                    for leaf = 2: params["min_samples_leaf"]

                        pred = []
                        println("$depth, $sample, $leaf")
                        for i = prog: leng
                            dolny = i - prog + 1
                            gorny = i - 1

                            train_y = Ys[dolny : gorny]
                            train_x = X[dolny : gorny, :]

                            train_x = reshape(train_x, gorny - dolny + 1, wymiary[2])

                            model = DecisionTree.build_tree(train_y,
                                                            train_x,
                                                            0,
                                                            depth,
                                                            leaf,
                                                            sample,
                                                            0.0,
                                                            rng=1)
                            #println(apply_tree(model, X[gorny + 1, :]))

                            prediction = apply_tree(model, X[gorny + 1, :])
                            push!(pred, prediction)
                            #print(pred)

                        end
                        all_preds = append!(all_preds, [depth, sample, leaf, float(Loss.MSE(pred, prog, Ys))])

                    end
                end

        end
        bledy = reshape(all_preds, (length(params) + 1), Int(length(all_preds)/(length(params) + 1)))
        #print(bledy)
        mm = findmin(bledy[(length(params) + 1),:])
        minn = mm[2]
        result = bledy[:, minn]

        to_ret = Dict(
            "max_depth" => Integer(result[1]),
            "min_samples_split"=> Integer(result[2]),
            "min_samples_leaf"=> Integer(result[3])
        )

        return to_ret
end

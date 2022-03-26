
using Pkg

Pkg.add("DecisionTree")
using DecisionTree

function rf_cross_val(dict)
    names(Main)
    X = Vector([1.1,2.2,3.3])
    Y = Vector([1.1,2.2,3.3])
    X = reshape(X, size(X))

    X = Float32.(X)
    Y = Float32.(Y)
    print(typeof(X))
    print(typeof(Y))
    model = DecisionTree.build_forest(Y, X')

    dlugosc_okna = dict["dlugosc_okna"]
    prog = dict["prog"]
    Ys = dict["data"].values
    X = dict["X"].values
    params = dict["params"]

    Ys = reshape(Ys, (length(Ys), 1))
    leng = length(Ys)
    println(size(X))
    println(size(Ys))
        function MSE_cross_val(preds, prog)
            actual = data[prog!]
            mse = (1 / length(preds)) * sum((actual - preds) ^ 2)
            return mse
        end

    all_preds = Matrix
    pure_errors = Matrix


        for depth = 1: params["max_depth"]
            for n_estimator = 1: params["max_n_estimators"]
                for sample = 2: params["min_sample_split"]
                    for leaf = 2: params["min_samples_leaf"]

                        pred = Matrix

                        for i = prog: leng
                            dolny = i - prog + 1
                            gorny = i

                            println(dolny)
                            println(gorny)

                            train_y = Ys[dolny : gorny]
                            train_x = X[dolny : gorny]



                            println("Done")
                        end


                        #all_preds = np.append(all_preds, [depth, n_estimator, pred])
                        #pure_errors = np.append(pure_errors, [Int(depth), Int(n_estimator), Int(sample), Int(leaf), MSE_cross_val(preds=pred, prog=self.prog)])
                        #pure_errors = pure_errors.reshape(-1, nrow(params) + 1)
                    end
                end
            end
        end
        #bledy = np.array(pure_errors[!, nrow(params)])
#
        #min_errors = min(bledy)
        #opt_depth = np.where(bledy==min_errors)[0]
        #result = pure_errors[opt_depth][0][0:nrow(params)]
        #to_ret = Dict{
        #    "depth" => result[0],
        #    "n_estimators"=> result[1],
        #    "min_sample_split"=> result[2],
        #    "min_samples_leaf"=> result[3]
        #}

        return 2#to_ret
end


#rf_cross_val(Dict("dlugosc_okna"=> 1/2,
#                     "prog"=> 10,
#                     "data"=> [1,2,3,4,5],
#                     "X"=> [1,2,3,4,5],
#                     "params"=> Dict(
#                            "max_depth"=> 5,
#                            "max_n_estimators"=> 8,
#                            "min_sample_split"=> 5,
#                            "min_samples_leaf"=> 5
#                     )))
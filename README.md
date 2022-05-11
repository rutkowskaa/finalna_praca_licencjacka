# finalna_praca_licencjacka

Generalnie najważniejsze pliki to:

resources/single_data - zawiera klasy z modelami jednowektorowymi
resources/vectorised_data/MISO/ARX_repr - zawiera klasy z modelami wielowektorowymi
resources/fast_jl - zawiera metody do walidacji (julia)

Nazwy najważniejszych zmiennych starałem się trzymać takie same. Opisałem je w CART_AR.py. Ogólny przykład wykorzystania dostępny jest w plikach ar.py oraz arx.py w folderze głównym
Żeby projekt zadziałał poprawnie, musi być zainstalowane środowisko julia:
https://julialang.org/downloads/

Do policzenia CART i RF w julii wykorzystałem paczkę:
https://github.com/JuliaAI/DecisionTree.jl

KNN wymaga na tyle mało mocy obliczeniowej, że nie było potrzeby tłumaczenia go.

Nie pamiętam kiedy rozmawialiśmy ostatnio, czy miała Pani Doktor okazję do korzystania z julii. Na wypadek jeśli nie to najmniej intuicyjne (moim zdaniem) różnice z pythonem zamieszczam poniżej:
# Julia zaczyna indeksowanie od 1 nie od 0 jak w pythonie - pilnowałem tego, ale jest to duże pole do błędów.
# Metoda reshape w julii zmienia typ obiektu.
# Konwersja python dataframe do julia dataframe jest nietrywialna - konieczne jest korzystanie z array.
# Arraye w julii są szybsze od np.array, a konwersja jest automatyczna.

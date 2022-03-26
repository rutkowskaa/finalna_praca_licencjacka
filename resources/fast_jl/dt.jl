using Pkg

Pkg.add("PyCall")
using PyCall

@pyimport yfinance


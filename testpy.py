import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'num':[1,2,3,4,5,6,7,8,9],
    'age':[23,78,22,19,45,33,20,21,33],
    'gender':[3,8,2,9,5,3,10,11,23],
})

testplt = df.plot.line(x='num')
testplt.figure.savefig('testplt.png')


import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'num':[1,2,3,4,5,6,7,8,9],
    'age':[23,78,22,19,45,33,20,21,33],
    'bb':[21,72,42,19,25,33,40,11,53]
})

df1 = pd.DataFrame({
    'num':[1,2,3,4,5,6,7,8,9],
    'age':[3,8,2,9,5,3,10,11,23],
    'cc':[213,8,24,23,45,12,30,51,63]
})

bx = df.plot.line(x='num',y ='age')
p1 = df1.plot.line(x='num',y='age',ax=bx)
p1.legend(["age1", "age2"]);
p1.figure.savefig('p1.png')




cx = df.plot.line(x='num', y='bb')
p2 = df1.plot.line(x='num',y = 'cc', ax=cx)
p2.figure.savefig('p2.png')


'''
https://erdogant.github.io/bnlearn/pages/html/Examples.html#dataframes
'''

import bnlearn as bn
# Import dataset
df = bn.import_example()
# Structure learning
model = bn.structure_learning.fit(df)
# Plot
G = bn.plot(model)
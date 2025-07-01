import templateOperation as tmpOP
import numpy as np
import preProcessing as pp

for avggap in range(1,50):
    
    print( avggap , " = ",pp.requiredNumberOfIterations(averageGap=avggap))




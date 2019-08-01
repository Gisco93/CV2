
if __name__ == '__main__':
    allgood = True
    try:
        import numpy as np
        A = np.ones((5,5))
        print('`import numpy` ok!')
    except:
        allgood = False
        print('`import numpy` failed')

    try:
        import torch
        A = torch.randn(5,5)
        print('`import torch` ok!')
    except:
        allgood = False
        print('`import torch` failed')

    try:
        import matplotlib.pyplot as plt
        h = plt.figure()
        plt.close(h)
        print('`import matplotlib.pyplot` ok!')
    except:
        allgood = False
        print('`import matplotlib.pyplot` failed')

    try:
        import maxflow
        print('`import maxflow` ok!')
    except:
        allgood = False
        print('`import maxflow` failed')

    if allgood:
        import gco
        print('\n\nRunning graphcuts example..')
        gco.test_graphcut()
        print("\n\nEverything is ok!")

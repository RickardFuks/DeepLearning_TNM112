import numpy as np
import matplotlib.pyplot as plt

class DataGenerator:
    def __init__(self):
        self.x = []
        self.y = []

    def generate(self, K, N, sigma):
        x = np.zeros((K*N,2))
        y = np.zeros((K*N))
        
        for k in range(K):
            x[k*N:(k+1)*N,:] = np.random.normal((k+1)/(K+1),sigma,(N,2))
            y[k*N:(k+1)*N] = k

        self.x = x
        self.y = y
        self.K = K

    def plot_data(self, nmn=None):
        K = len(np.unique(self.y))
        
        plt.figure(figsize=(4,4))
        for i in range(K):
            plt.scatter(self.x[self.y==i,0], self.x[self.y==i,1], s=20, alpha=0.8)
        plt.axis('equal')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.legend(['Class %d'%i for i in range(K)])
        plt.tight_layout()
        if nmn:
            plt.savefig(nmn)
        plt.show()

    def rotate(self, ang):
        W = np.array([[np.cos(ang),-np.sin(ang)],
                      [np.sin(ang), np.cos(ang)]])
    
        xt = np.zeros(self.x.shape)
        for i in range(self.x.shape[0]):
            xi = self.x[i,:]
            xi = xi[:,np.newaxis]
            xt[i,:] = np.matmul(W,xi)[:,0]
        
        self.x = xt

    def export_data(self, outdir):
        np.savez(outdir, self.x, self.y)
        #f = open("y.txt", "w")
        #for row in y:
        #    f.write("%0.2f\n"%row)
        #f.close()

    def import_data(self, outdir):
        tmp = np.load(outdir)
        self.x = tmp['arr_0']
        self.y = tmp['arr_1']
        self.K = len(np.unique(self.y))
        
        #with open("y.txt", "r") as f:
        #    txt = f.read()
        #y = np.array([float(i) for i in txt.split()])
        
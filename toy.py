import numpy as np
import numpy.matlib as nm
import matplotlib.pyplot as plt

from src.svgd.svgd import SVGD

class MVN:
    def __init__(self, mu, A):
        self.mu = mu
        self.A = A
    
    def calculate_lnprob(self, theta):
        return -1 * np.matmul(theta - nm.repmat(self.mu, theta.shape[0], 1), self.A)

def plot_distribution(theta, mu, iteration):
    plt.figure(figsize=(8, 6))
    plt.scatter(theta[:, 0], theta[:, 1], alpha=0.5, label='Sample Points')
    plt.scatter(mu[0], mu[1], color='red', marker='x', s=100, label='True Mean')
    plt.title(f'SVGD Step {iteration}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    A = np.array([[0.2260, 0.1652], [0.1652, 0.6779]])
    mu = np.array([-0.6871, 0.8010])
    
    model = MVN(mu, A)
    
    x0 = np.random.normal(0, 1, [100, 2])
    
    svgd = SVGD()
    
    # 초기 분포 시각화
    plot_distribution(x0, mu, 'Initial')
    
    # SVGD 업데이트 및 시각화
    for iteration in range(1000):
        theta = svgd.update(x0, model.calculate_lnprob, iteration=1, stepsize=0.01)
        
        # 각 100 스텝마다 시각화
        if iteration % 100 == 0:
            plot_distribution(theta, mu, iteration)
        
        x0 = theta  # 다음 업데이트를 위해 현재 theta를 x0로 설정

    print("ground truth: ", mu)
    print("svgd: ", np.mean(theta, axis=0))

# import numpy as np
# import numpy.matlib as nm
# import matplotlib.pyplot as plt

# from src.svgd.svgd import SVGD

# class MVN:
#     def __init__(self, mu, A):
#         self.mu = mu
#         self.A = A
    
#     def calculate_lnprob(self, theta):
#         return -1 * np.matmul(theta - nm.repmat(self.mu, theta.shape[0], 1), self.A)

# def plot_distribution(theta, mu, iteration):
#     plt.figure(figsize=(8, 6))
#     plt.scatter(theta[:, 0], theta[:, 1], alpha=0.5, label='Sample Points')
#     plt.scatter(mu[0], mu[1], color='red', marker='x', s=100, label='True Mean')
#     plt.title(f'SVGD Step {iteration}')
#     plt.xlabel('X-axis')
#     plt.ylabel('Y-axis')
#     plt.xlim(-3, 3)
#     plt.ylim(-3, 3)
#     plt.legend()
#     plt.grid()
#     # plt.show()

#     # 파일로 저장
#     plt.savefig(f"outputImgs/svgd_step_{iteration}.png")
#     plt.close()  # 메모리 관리를 위해 닫기

# if __name__ == '__main__':
#     A = np.array([[0.2260, 0.1652], [0.1652, 0.6779]])
#     mu = np.array([-0.6871, 0.8010])
    
#     model = MVN(mu, A)
    
#     x0 = np.random.normal(2, 2, [100, 2])
    
#     svgd = SVGD()
    
#     # 초기 분포 시각화
#     plot_distribution(x0, mu, 'Initial')
    
#     # SVGD 업데이트 및 시각화
#     for iteration in range(1000):
#         theta = svgd.update(x0, model.calculate_lnprob, iteration=1, stepsize=0.01)
        
#         # 각 100 스텝마다 시각화
#         if iteration % 1 == 0:
#             plot_distribution(theta, mu, iteration)
        
#         x0 = theta  # 다음 업데이트를 위해 현재 theta를 x0로 설정

#     print("ground truth: ", mu)
#     print("svgd: ", np.mean(theta, axis=0))

import numpy as np
import numpy.matlib as nm
import matplotlib.pyplot as plt
import imageio
import os

from src.svgd.svgd import SVGD

class MVN:
    def __init__(self, mu, A):
        self.mu = mu
        self.A = A
    
    def calculate_lnprob(self, theta):
        return -1 * np.matmul(theta - nm.repmat(self.mu, theta.shape[0], 1), self.A)

def plot_distribution(theta, mu, iteration, save_dir='outputImgs'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(theta[:, 0], theta[:, 1], alpha=0.5, label='Sample Points')
    
    # 첫 번째 평균
    plt.scatter(mu[0], mu[1], color='red', marker='x', s=100, label='True Mean 1')
    
    # 두 번째 평균 (예를 들어, mu[0] + 1, mu[1] + 1을 사용하여 다른 위치에 표시)
    plt.scatter(mu[0] + 1, mu[1] + 1, color='blue', marker='o', s=100, label='True Mean 2')
    
    plt.title(f'SVGD Step {iteration}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.legend()
    plt.grid()
    
    # 파일로 저장
    plt.savefig(f"{save_dir}/svgd_step_{iteration}.png")
    plt.close()  # 메모리 관리를 위해 닫기


def create_gif(image_folder='outputImgs', gif_name='svgd_evolution.gif'):
    images = []
    for filename in sorted(os.listdir(image_folder), key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else float('inf')):
        if filename.endswith('.png') and filename != 'svgd_step_Initial.png':
            images.append(imageio.imread(f"{image_folder}/{filename}"))
    imageio.mimsave(gif_name, images, fps=10)  # fps 값 조정 가능


if __name__ == '__main__':
    A = np.array([[0.2260, 0.1652], [0.1652, 0.6779]])
    mu = np.array([-4, 3])
    
    model = MVN(mu, A)
    
    x0 = np.random.normal(7, 7, [100, 2])
    
    svgd = SVGD()
    
    # 초기 분포 시각화
    plot_distribution(x0, mu, 'Initial')
    
    # SVGD 업데이트 및 시각화
    for iteration in range(700):
        theta = svgd.update(x0, model.calculate_lnprob, iteration=1, stepsize=0.01)
        
        # 각 스텝마다 시각화 및 저장
        plot_distribution(theta, mu, iteration)
        
        x0 = theta  # 다음 업데이트를 위해 현재 theta를 x0로 설정

    print("ground truth: ", mu)
    print("svgd: ", np.mean(theta, axis=0))
    
    # GIF 생성
    create_gif()

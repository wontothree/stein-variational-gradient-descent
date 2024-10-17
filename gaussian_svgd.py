import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.svgd.svgd import SVGD  # SVGD 모듈 임포트

np.random.seed(19)
torch.manual_seed(19)

if not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda')


class GaussianDistribution(object):
    def __init__(self, mu, cov, device='cuda'):
        super(GaussianDistribution, self).__init__()

        self.mu = mu
        self.cov = cov
        self.precision = torch.inverse(cov)

        self.R = torch.linalg.cholesky(self.cov)
        self.normal = torch.distributions.normal.Normal(
            torch.zeros_like(mu), torch.ones_like(mu))

    def nl_pdf(self, x):
        # 로그 확률 밀도 함수 수정
        return -0.5 * ((x - self.mu).matmul(self.precision)).matmul(x - self.mu)

    def sample(self, num_samples=1):
        return (self.R @ self.normal.sample((num_samples,)).T).T + self.mu


class MVN:
    def __init__(self, mu, A):
        self.mu = mu
        self.A = A
    
    def calculate_lnprob(self, theta):
        return -1 * np.matmul(theta - np.tile(self.mu, (theta.shape[0], 1)), self.A)


if __name__ == '__main__':
    # SVGD 초기화
    svgd = SVGD()

    # 가우시안 분포 설정
    dim = 2
    mu = torch.tensor([1.2, .6], device=device)
    cov = (
        0.9 *
        (torch.ones([2, 2], device=device) - torch.eye(2, device=device)).T +
        torch.eye(2, device=device) * 1.3)
    gaussian_dist = GaussianDistribution(mu, cov, device=device)

    # MVN 모델 정의
    A = torch.inverse(cov).cpu().numpy()  # A는 공분산의 역행렬
    model = MVN(mu.cpu().numpy(), A)  # MVN 객체 생성

    # SVGD를 위한 초기 샘플
    num_svgd_samples = 1000
    x_svgd = gaussian_dist.sample(num_svgd_samples).cpu().numpy()  # 가우시안 분포에서 초기 샘플링
    # print(x_svgd)

    # SVGD 업데이트
    loss_log = []
    for iteration in tqdm(range(2000)):
        # SVGD 업데이트
        x_svgd = svgd.update(x_svgd, model.calculate_lnprob, iteration=1, stepsize=0.01)

        # 로그 확률을 기반으로 손실 계산
        loss_log.append(-model.calculate_lnprob(x_svgd).mean())  # 평균 로그 확률 계산

    # 진짜 샘플 생성
    num_samples = 1000
    true_samples = np.zeros([num_samples, 2])
    for j in range(num_samples):
        true_samples[j, :] = gaussian_dist.sample(1).cpu().numpy()

    # 시각화
    fig = plt.figure("training logs - net", dpi=150, figsize=(7, 2.5))
    plt.plot(loss_log)
    plt.title("Unnormalized PDF")
    plt.xlabel("Iterations")
    plt.ylabel(r"$- \log \ \mathrm{N}(\mathbf{x} | \mu, \Sigma) + const.$")
    plt.grid()

    fig = plt.figure(dpi=150, figsize=(9, 4))
    plt.subplot(121)
    plt.scatter(x_svgd[:, 0], x_svgd[:, 1], s=.5, color="#db76bf")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.xlim([-3, 6])
    plt.ylim([-4, 5])
    plt.title("SVGD Results")
    
    plt.subplot(122)
    plt.scatter(true_samples[:, 0],
                true_samples[:, 1],
                s=.5,
                color="#5e838f")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.xlim([-3, 6])
    plt.ylim([-4, 5])
    plt.title(r"$\mathbf{x} \sim \mathrm{N}(\mu, \Sigma)$")
    
    plt.tight_layout()
    plt.show()

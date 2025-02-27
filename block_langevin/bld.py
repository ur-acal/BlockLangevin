import torch
from tqdm import tqdm
import torch.distributions as D
import numpy as np
import pandas as pd
from typing import Union, Iterable

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
def matrix_sqrt(matrix: torch.Tensor):
    vals, vecs = torch.linalg.eigh(matrix)
    matrix_sqrt = torch.matmul(vecs, torch.matmul(torch.diag(vals.real.sqrt()), torch.inverse(vecs)))
    return matrix_sqrt

def gaussian_kl(gauss: D.MultivariateNormal, gauss2: D.MultivariateNormal):
    return D.kl_divergence(gauss2, gauss).item()
       

def gaussian_wasserstein(gauss: D.MultivariateNormal, gauss2: D.MultivariateNormal):
    cov = gauss.covariance_matrix
    mu = gauss.mean
    mu_emp = gauss2.mean
    cov_emp = gauss2.covariance_matrix
    sqrtcov = matrix_sqrt(cov)
    wasserstein = (mu - mu_emp).norm() + (
        cov + cov_emp - 2 * matrix_sqrt(sqrtcov @ cov_emp @ sqrtcov)
    ).trace()
        # wasserstein_vals.append((i, wasserstein))
    return np.sqrt(wasserstein.item())

class LDSampler:
    def __init__(self, lr: float, thinning: int, beta: float, metric: str = "W2", sample_count=1):
        self.metric = metric
        self.thinning = thinning
        self.lr = lr
        self.beta = beta
        self.coeff = np.sqrt(2 * lr / beta)

    def _log_sample(self, step: int):
        if step % self.thinning == 0:
            cov_emp = torch.cov(self.state)
            mu_emp = torch.mean(self.state, dim=1)
            gauss_emp = D.MultivariateNormal(loc=mu_emp, 
                                             covariance_matrix=cov_emp)
            self.samples.append((step, D.kl_divergence(gauss_emp, self.target).item(), 
                                 gaussian_wasserstein(
                                     gauss_emp,
                                     self.target
                                 )))

    def _do_step(self, 
                 gradient_oracle: callable):
        self.state: torch.Tensor
        gradient = gradient_oracle(self.state)
        # perform the Liemkuler-Matthews update
        self.state += -self.lr * gradient + self.coeff * torch.randn_like(self.state)

    def sample(self, 
               gradient_oracle: callable, 
               nsteps: int,
               x0: torch.Tensor,
               target):
        self.target = target
        self.samples = []
        self.state = x0.clone().detach()
        for i in tqdm(range(nsteps)):
            if i % self.thinning == 0:
                self._log_sample(i)  
            self._do_step(gradient_oracle)
       

class BLDSampler:
    def __init__(self, 
                 lr: float, 
                 thinning: int, 
                 beta: float, 
                 blocks: Union[int, Iterable],
                 metric: str = "W2", 
                 selection: str = 'cyclic',
                 sample_count=1):
        self.metric = metric
        self.thinning = thinning
        self.lr = lr
        self.beta = beta
        self.selection = selection
        self.block_config = blocks
        self.coeff = np.sqrt(2 * lr / beta)
        self.current_block = None
        self.sample_count = sample_count

    def _log_sample(self, step: int):
        if step % self.thinning == 0:
            cov_emp = torch.cov(self.state)
            mu_emp = torch.mean(self.state, dim=1)
            gauss_emp = D.MultivariateNormal(loc=mu_emp, 
                                             covariance_matrix=cov_emp)
            self.samples.append((step, D.kl_divergence(gauss_emp, self.target).item(), 
                                 gaussian_wasserstein(
                                     gauss_emp,
                                     self.target
                                 )))



    def _do_step(self, 
                 gradient_oracle: callable):
        self.state: torch.Tensor
        gradient = gradient_oracle(self.state)
        # perform the Liemkuler-Matthews update
        # breakpoint()
        self.state += self.Ui.matmul(-self.lr * gradient + self.coeff * torch.randn_like(self.state))

    def change_block(self):
        if self.selection == 'random':
            self.current_block = np.random.randint(low=0, high=len(self.blocks))
        else:
            if self.current_block is None:
                self.current_block = 0
            else:
                self.current_block = (self.current_block + 1) % len(self.blocks)
        self.Ui.fill_(0)
        self.Ui[self.blocks[self.current_block], self.blocks[self.current_block]] = 1
 

    def sample(self, 
               gradient_oracle: callable, 
               cycles: int,
               block_steps: int,
               x0: torch.Tensor,
               target):
        self.target = target
        if isinstance(self.block_config, int):
            self.blocks = np.array_split(np.arange(x0.shape[0]), self.block_config)
        else:
            self.blocks = self.block_config
        self.samples = []
        self.state = x0.clone().detach()
        self.Ui = torch.zeros((self.state.shape[0], self.state.shape[0]), device=device)
        lr = self.lr
        # self.lr /= block_steps
        # breakpoint()
        for i in tqdm(range(cycles * blocks)):
            if i % block_steps == 0:
                self.change_block()
                # if i % (len(self.blocks) * block_steps) == 0:
            if i % self.thinning == 0:
                self._log_sample(i) 
            self._do_step(gradient_oracle)
        self.lr = lr
    

def make_df(samples, name, seed):
    df = pd.DataFrame(data=samples, columns=['step', 'KL', 'W2'])
    df['seed'] = seed
    df.to_csv(f'data/{name}')


if __name__ == '__main__':
    N = 50
    seed = 58888319060236145
    torch.manual_seed(seed)
    covmat = 5 * (torch.rand((N, N), device=device) * 2 - 1)
    covmat = (covmat.T + covmat) / 2
    # covmat = torch.eye(N, device=device)
    mineig = torch.linalg.eigvalsh(covmat).real.min()
    if mineig < 0:
        covmat += torch.eye(covmat.shape[0], device=device) * 1.2 * abs(mineig)
    simmat = covmat.inverse()
    def grad(x):
        return simmat.matmul(x)
    x0 = 0.1*torch.randn((N, 10000), device=device)
    target = D.MultivariateNormal(loc=torch.zeros(N, device=device),
                                                   covariance_matrix=covmat)
    totalsteps = 150000
    lr = 1e-3
    sampler = LDSampler(lr=lr, thinning=30, beta=1)
    sampler.sample(grad, totalsteps, x0, target=target)
    make_df(sampler.samples, name='full_gradient_gaussian.csv', seed=seed)
    for blocks in [2, 5, 10]:
        for epochs in [10]:
            for selection in ['random', 'cyclic']:
                sampler = BLDSampler(lr=lr, thinning=30, beta=1, blocks=blocks, sample_count=500, selection=selection)
                sampler.sample(grad, totalsteps, epochs, x0, target=target)
                make_df(sampler.samples, name=f'{selection}_block_b{blocks}_e{epochs}_gaussian.csv',
                         seed=seed)
    for blocks in [2, 5, 10]:
        for epochs in [10]:
            # sampler = LDSampler(lr=1e-3, thinning=1, beta=1)
            sampler = BLDSampler(lr=lr, thinning=30, beta=1, blocks=blocks, sample_count=500)
            # sampler.sample(grad, 10000, x0, target=target)
            sampler.sample(grad, totalsteps, epochs, x0, target=target)
            make_df(sampler.samples, name=f'block_b{blocks}_e{epochs}_gaussian.csv', seed=seed)
    for blocks in [5]:
        for epochs in [10, 25, 50]:
            # sampler = LDSampler(lr=1e-3, thinning=1, beta=1)
            sampler = BLDSampler(lr=lr, thinning=30, beta=1, blocks=blocks, sample_count=500)
            # sampler.sample(grad, 10000, x0, target=target)
            sampler.sample(grad, totalsteps, epochs, x0, target=target)
            make_df(sampler.samples, name=f'block_b{blocks}_e{epochs}_gaussian.csv', seed=seed)
    for blocks in [5]:
        epochs = 10
        for perurbation in [0.6, .8, 1.0, 1.2]:
            perturb = (1 + perurbation * torch.randn(N, N, device=device)) * simmat
            def grad(x):
                return perturb.matmul(x)
            # sampler = LDSampler(lr=1e-3, thinning=1, beta=1)
            sampler = BLDSampler(lr=lr, thinning=30, beta=1, blocks=blocks, sample_count=500)
            # sampler.sample(grad, 10000, x0, target=target)
            sampler.sample(grad, totalsteps, epochs, x0, target=target)
            make_df(sampler.samples, name=f'block_b{blocks}_e{epochs}_gaussian_p_{perurbation}.csv', seed=seed)


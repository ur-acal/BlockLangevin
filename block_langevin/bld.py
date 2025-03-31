import torch
from tqdm import tqdm
import torch.distributions as D
import numpy as np
import pandas as pd
from typing import Union, Iterable

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def matrix_sqrt(matrix: torch.Tensor):
    # Given a PSD matrix A, compute B such that A=BB
    vals, vecs = torch.linalg.eigh(matrix)
    matrix_sqrt = torch.matmul(vecs, torch.matmul(torch.diag(vals.real.sqrt()), torch.inverse(vecs)))
    return matrix_sqrt

def gaussian_kl(gauss: D.MultivariateNormal, gauss2: D.MultivariateNormal):
    # Compute the KL-divergence between two Gaussians
    return D.kl_divergence(gauss2, gauss).item()
       

def gaussian_wasserstein(gauss: D.MultivariateNormal, gauss2: D.MultivariateNormal):
    # Compute the 2-Wasserstein distance between two Gaussians
    cov = gauss.covariance_matrix
    mu = gauss.mean
    mu_emp = gauss2.mean
    cov_emp = gauss2.covariance_matrix
    sqrtcov = matrix_sqrt(cov)
    wasserstein = (mu - mu_emp).norm() + (
        cov + cov_emp - 2 * matrix_sqrt(sqrtcov @ cov_emp @ sqrtcov)
    ).trace()
    return np.sqrt(wasserstein.item())

class LDSampler:
    """Implementation of the unadjusted Langevin algorithm using Euler-Maruyama discretization
    """
    def __init__(self, lr: float, thinning: int, beta: float, metric: str = "W2"):
        self.metric = metric
        self.thinning = thinning
        self.lr = lr
        self.beta = beta
        self.coeff = np.sqrt(2 * lr / beta)

    def _log_sample(self, step: int):
        # Log a single sample of the W2/KL-divergence, using the covariance/mean of the current state
        if step % self.thinning == 0:
            # Compute the empirical mean/covariance across samples
            cov_emp = torch.cov(self.state)
            mu_emp = torch.mean(self.state, dim=1)
            # Make a torch Distributions object
            gauss_emp = D.MultivariateNormal(loc=mu_emp, 
                                             covariance_matrix=cov_emp)
            # Log the samples
            self.samples.append((step, D.kl_divergence(gauss_emp, self.target).item(), 
                                 gaussian_wasserstein(
                                     gauss_emp,
                                     self.target
                                 )))

    def _do_step(self, 
                 gradient_oracle: callable):
        # Perform a single time step
        self.state: torch.Tensor
        gradient = gradient_oracle(self.state)
        # perform the Euler-Maruyama update
        self.state += -self.lr * gradient + self.coeff * torch.randn_like(self.state)

    def sample(self, 
               gradient_oracle: callable, 
               nsteps: int,
               x0: torch.Tensor,
               target):
        # Main sampling loop
        self.target = target # Target potential f(x)
        self.samples = [] # sample storage
        self.state = x0.clone().detach() # current state
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
                 selection: str = 'cyclic'):
        """Implementation of Block Langevin Dynamics

        NOTE: Using an Euler-Maruyama discretization, higher-accuracy results should use a higher order method (e.g. SRK)

        Args:
            lr (float): Learning rate
            thinning (int): thinning parameter (store a sample every `thinning` iterations)
            beta (float): Inverse temperature
            blocks (Union[int, Iterable]): Number of blocks
            metric (str, optional): Metric to compute. Defaults to "W2".
            selection (str, optional): Selection strategy, either "random" or "cyclic". Defaults to 'cyclic'.
        """
        self.metric = metric
        self.thinning = thinning
        self.lr = lr
        self.beta = beta
        self.selection = selection
        self.block_config = blocks
        self.coeff = np.sqrt(2 * lr / beta)
        self.current_block = None

    def _log_sample(self, step: int):
        """Log a single sample of W2/KL vs. target distribution,
          computed using the empirical mean/covariance

        Args:
            step (int): Current time step
        """
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
        """Perform a time step single step

        Args:
            gradient_oracle (callable): Oracle for the gradient of the target potential $$f(x)$$
        """
        self.state: torch.Tensor
        gradient = gradient_oracle(self.state)
        # perform the Euler-Maruyama update
        self.state += self.Ui.matmul(-self.lr * gradient + self.coeff * torch.randn_like(self.state))

    def change_block(self):
        """Choose the next block to evolve
        """
        if self.selection == 'random':
            self.current_block = np.random.randint(low=0, high=len(self.blocks))
        else:
            if self.current_block is None:
                self.current_block = 0
            else:
                self.current_block = (self.current_block + 1) % len(self.blocks)
        # Compute the partial diagonal block matrix (U_ii=1 if i in current block, 0 otherwise)
        self.Ui.fill_(0)
        self.Ui[self.blocks[self.current_block], self.blocks[self.current_block]] = 1
 

    def sample(self, 
               gradient_oracle: callable, 
               cycles: int,
               block_steps: int,
               x0: torch.Tensor,
               target: D.Distribution):
        """Main BLD sampling loop

        Args:
            gradient_oracle (callable): Oracle for potential gradient $$\nabla f(x)$$
            cycles (int): Number of (expected) whole-problem passes
            block_steps (int): Number of steps per block
            x0 (torch.Tensor): Starting state (also determines number of replicas).
                Expected shape is (d, R), where R is the number of replicas used for 
                statistical estimation
            target (torch.distributions.Distribution): Target distribution
        """
        self.target = target
        if isinstance(self.block_config, int):
            self.blocks = np.array_split(np.arange(x0.shape[0]), self.block_config)
        else:
            self.blocks = self.block_config
        self.samples = []
        self.state = x0.clone().detach()
        self.Ui = torch.zeros((self.state.shape[0], self.state.shape[0]), device=device)
        lr = self.lr

        for i in tqdm(range(cycles * blocks)):
            # Change blocks
            if i % block_steps == 0:
                self.change_block()

            # Log a new sample of W2/KL
            if i % self.thinning == 0:
                self._log_sample(i) 
            # Perform a time step
            self._do_step(gradient_oracle)
        self.lr = lr
    

def make_df(samples:Iterable[tuple], name: str, seed: int):
    """Output a CSV based on samples

    Args:
        samples (Iterable[tuple]): List of records, with each record storing 
            a time step, KL-divergence value, and a W2 value
        name (str): Output file name (including .csv extension)
        seed (int): Random seed used to collect data
    """
    df = pd.DataFrame(data=samples, columns=['step', 'KL', 'W2'])
    df['seed'] = seed
    df.to_csv(f'data/{name}')


if __name__ == '__main__':
    N = 50 # Sample a 50 dimensional Gaussian
    seed = 58888319060236145 # Arbitrary seed generated by torch, just here for future reproducibility
    torch.manual_seed(seed)
    
    # Make the target potential

    covmat = 5 * (torch.rand((N, N), device=device) * 2 - 1)
    covmat = (covmat.T + covmat) / 2
    
    # Ensure the covariance matrix is Symmetric + PD
    mineig = torch.linalg.eigvalsh(covmat).real.min()
    if mineig < 0:
        covmat += torch.eye(covmat.shape[0], device=device) * 1.2 * abs(mineig)
    simmat = covmat.inverse()
    
    # Make the potential function (assuming mean is 0)
    def grad(x):
        return simmat.matmul(x)
    x0 = 0.1*torch.randn((N, 10000), device=device)

    # Make the target distribution object
    target = D.MultivariateNormal(loc=torch.zeros(N, device=device),
                                                   covariance_matrix=covmat)
    
    # Run for 150K steps with LR of 1e-3 (not magic numbers, but tractable to run and see differences)
    totalsteps = 150000
    lr = 1e-3

    # Collect Full LD results (b=1)
    sampler = LDSampler(lr=lr, thinning=30, beta=1)
    sampler.sample(grad, totalsteps, x0, target=target)
    make_df(sampler.samples, name='full_gradient_gaussian.csv', seed=seed)


    # Test the effect of blocks and block seletion on convergence
    for blocks in [2, 5, 10]:
        for epochs in [10]:
            for selection in ['random', 'cyclic']:
                sampler = BLDSampler(lr=lr, thinning=30, beta=1, blocks=blocks, selection=selection)
                sampler.sample(grad, totalsteps, epochs, x0, target=target)
                make_df(sampler.samples, name=f'{selection}_block_b{blocks}_e{epochs}_gaussian.csv',
                         seed=seed)
                
    # Test the effect of epoch duration on convergence (e.g. annealing blocks for longer before swapping)
    for blocks in [5]:
        for epochs in [10, 25, 50]:
            sampler = BLDSampler(lr=lr, thinning=30, beta=1, blocks=blocks)
            sampler.sample(grad, totalsteps, epochs, x0, target=target)
            make_df(sampler.samples, name=f'block_b{blocks}_e{epochs}_gaussian.csv', seed=seed)
    
    # Test the effect of fixed perturbations on convergence
    for blocks in [5]:
        epochs = 10
        for perurbation in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            perturb = (1 + perurbation * torch.randn(N, N, device=device)) * simmat
            def grad(x):
                return perturb.matmul(x)
            sampler = BLDSampler(lr=lr, thinning=30, beta=1, blocks=blocks)
            sampler.sample(grad, totalsteps, epochs, x0, target=target)
            make_df(sampler.samples, name=f'block_b{blocks}_e{epochs}_gaussian_p_{perurbation}.csv', seed=seed)


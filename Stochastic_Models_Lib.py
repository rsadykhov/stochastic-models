import numpy as np
import pandas as pd
import plotly.express as px

##########################################################################################################################################################################################################################

# Arithmetic Brownian Motion

class Arithmetic_Brownian_Motion:
    """dS = mu*dt + sigma*dW
    Requires numpy, pandas and plotly.express to run"""
    def __init__(self, mu, sigma, n_paths, n_steps, t, T, S_0):
        self.mu = mu
        self.sigma = sigma
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.t = t
        self.T = T
        self.S_0 = S_0
        
    def get_paths(self):
        """Returns the paths, S, for the Arithmetic Brownian Motion using the Euler-Maruyama method"""
        dt = self.T/self.n_steps
        dW = np.sqrt(dt)*np.random.randn(self.n_paths, self.n_steps)
        dS = self.mu*dt + self.sigma*dW
        
        dS = np.insert(dS, 0, self.S_0, axis=1)
        S = np.cumsum(dS, axis=1)
        
        return S
    
    def get_expectation(self):
        """Returns the expectation, E[S], for the Arithmetci Brownian Motion"""
        ES = self.mu*self.t+self.S_0
        return ES
    
    def get_variance(self):
        """Returns the variance, Var[S], for the Arithmetic Brownian Motion"""
        VarS = self.t*self.sigma**2
        return VarS
    
    def get_auto_cov(self, t1, t2):
        """Returns the auto-covariance for the Arithmetic Brownian Motion"""
        AC = (self.sigma**2)*min(self.t[t1], self.t[t2])
        return AC
    
    def simulate(self, plot_expected=False):
        """Returns the plot of the random paths taken by the Arithmatic Brownian Motion"""
        plotting_df = pd.DataFrame(self.get_paths().transpose())
        if plot_expected==True:
            plotting_df["Expected Path"]=self.get_expectation()
        fig = px.line(plotting_df, labels={"value":"Value of S", "variable":"Paths"})
        return fig.show()




##########################################################################################################################################################################################################################

#Geometric Brownian Motion

class Geometric_Brownian_Motion:
    """dS = mu*S*dt + sigma*S*dW
    Model describing the evolution of stock prices
    Requires numpy, pandas and plotly.express to run"""
    def __init__(self, mu, sigma, n_paths, n_steps, t, T, S_0):
        self.mu = mu
        self.sigma = sigma
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.t = t
        self.T =T
        self.S_0 = S_0
    
    def get_paths(self):
        """Returns the paths, S, for the Geometric Brownian Motion using Euler-Maruyama method"""
        dt = self.T/self.n_steps
        dW = np.sqrt(dt)*np.random.randn(self.n_paths, self.n_steps)
        dS = self.mu*dt + self.sigma*dW
        
        dS = np.insert(dS, 0, 0, axis=1)
        S = np.cumsum(dS, axis=1)
        
        S = self.S_0*np.exp(S)
        return S
    
    def get_expection(self):
        """Returns the expectation, E[S], of the Geometric Brownian Motion"""
        ES = self.S_0*np.exp(self.mu*self.t)
        return ES
    
    def get_variance(self):
        """Returns the variance, Var[S], of the Geometric Brownian Motion"""
        VarS = (self.S_0**2)*np.exp(2*self.mu*self.t)*(np.exp(self.t*self.sigma**2)-1)
        return VarS
    
    def simulate(self, plot_expected=False):
        """Returns the plot of the random paths taken by the Geometric Brownian Motion"""
        plotting_df = pd.DataFrame(self.get_paths().transpose())
        if plot_expected==True:
            plotting_df["Expected Path"]=self.get_expection()
        fig = px.line(plotting_df, labels={"value":"Value of S", "variable":"Paths"})
        return fig.show()




##########################################################################################################################################################################################################################

#Ornstein-Uhlenbeck Process

class Ornstein_Uhlenbeck_Process:
    """dS = alpha*(mu-S)*dt + sigma*dW
    Model describes the evolution of interest rates
    Requires numpy, pandas and plotly.express"""
    def __init__(self, alpha, mu, sigma, n_paths, n_steps, t, T, S_0):
        self.alpha =alpha
        self.mu = mu
        self.sigma = sigma
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.t = t
        self.T =T
        self.S_0 = S_0
    
    def get_paths(self, analytic_EM=False):
        """Returns the paths, S, for the Ornsteain_uhlenbeck Process using Euler-Maruyama method.
        Intakes an argument analytic_EM with bool values. If True, then returns the simulation with the analytic 
        moments for Euler-Maruyama; if False, then returns plain Euler-Maruyama simulation"""
        dt = self.T/self.n_steps
        N = np.random.randn(self.n_steps, self.n_paths)
        S = np.concatenate((self.S_0*np.ones((1, self.n_paths)), np.zeros((self.n_steps, self.n_paths))), axis=0)
            
            
        if analytic_EM==True:
            sdev = self.sigma*np.sqrt((1-np.exp(-2*self.alpha*dt))/(2*self.alpha))
            for i in range(0, self.n_steps):
                S[i+1,:] = self.mu + (S[i,:]-self.mu)*np.exp(-self.alpha*dt) + sdev*N[i,:]
        else:
            sdev = self.sigma*np.sqrt(dt)
            for i in range(0, self.n_steps):
                S[i+1,:] = S[i,:] + self.alpha*(self.mu-S[i,:])*dt + sdev*N[i,:]
        
        return S
    
    def get_expectation(self):
        """Returns the expectation, E[S], for the Ornstein-Uhlenbeck Process"""
        ES = self.mu + (self.S_0-self.mu)*np.exp(-self.alpha*t)
        return ES
    
    def get_variance(self):
        """Returns the variance, Var[S], for the Ornstein-Uhlenbeck Process"""
        VarS = (1-np.exp(-2*self.alpha*t))*(self.sigma**2)/(2*self.alpha)
        return VarS
    
    def simulate(self, analytic_EM=False, plot_expected=False):
        """Returns the plot of the random paths taken by the Ornstain_uhlenbeck Process"""
        plotting_df = pd.DataFrame(self.get_paths(analytic_EM))
        if plot_expected==True:
            plotting_df["Expected Path"]=self.get_expectation()
        fig = px.line(plotting_df, labels={"value":"Value of S", "variable":"Paths"})
        return fig.show()




##########################################################################################################################################################################################################################

#Brownian Bridge

class Brownian_Bridge:
    """dS = ((b-X)/(T-t))*dt + sigma*dW
    Model can support useful variance reduction techniques for pricing derivative contracts using Monte-Carlo simulation, 
    such as sampling. Also used in scenario generation.
    Requires numpy, pandas and plotly.express"""
    def __init__(self, alpha, beta, sigma, n_paths, n_steps, t, T):
        self.alpha =alpha
        self.beta = beta
        self.sigma = sigma
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.t = t
        self.T =T
    
    def get_paths(self):
        """Returns the paths, S, for the Brownian Bridge using the Euler-Maruyama method"""
        dt = self.T/self.n_steps
        dW = np.sqrt(dt)*np.random.randn(self.n_steps, self.n_paths)
        S = np.concatenate((self.alpha*np.ones((1, self.n_paths)),
                             np.zeros((self.n_steps-1, self.n_paths)), self.beta*np.ones((1, self.n_paths))), axis=0)
        
        for i in range(0, self.n_steps-1):
            S[i+1,:] = S[i,:] + (self.beta-S[i,:])/(self.n_steps-i+1) +self.sigma*dW[i,:]
        
        return S
    
    def get_expectation(self):
        """Returns the expectation, E[S], for the Brownian Bridge"""
        ES = self.alpha + (self.beta-self.alpha)/T*t
        return ES
    
    def get_variance(self):
        """Returns the variance, Var[S], for the Brownian Bridge"""
        VarS = self.t*(self.T-self.t)/self.T
        return VarS
    
    def simulate(self, plot_expected=False):
        """Returns the plot of the random paths taken by the Brownian Bridge"""
        plotting_df = pd.DataFrame(self.get_paths())
        if plot_expected==True:
            plotting_df["Expected Path"]=self.get_expectation()
        fig = px.line(plotting_df, labels={"value":"Value of S", "variable":"Paths"})
        return fig.show()




##########################################################################################################################################################################################################################

#Feller Square-Root Process

class Feller_Square_Root_Process:
    """dS = alpha*(mu-S)*dt + sigma*sqrt(S)*dW
    Model describes the evolution of interest rates
    Requires numpy, pandas and plotly.express"""
    def __init__(self, alpha, mu, sigma, n_paths, n_steps, t, T, S_0):
        self.alpha =alpha
        self.mu = mu
        self.sigma = sigma
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.t = t
        self.T =T
        self.S_0 = S_0
    
    def get_paths(self, sim_type="EM", analytic_EM=False):
        """Returns the paths, S, for the Feller Square-Root Process using either Euler-Maruyama method or the exact method.
        For Euler-Maruyama, set sim_type to "EM"; for exact, set it to "E". Intakes an argument analytic_EM with bool values. 
        If True, then returns the simulation with the analytic moments for Euler-Maruyama; if False, then returns plain 
        Euler-Maruyama simulation"""
        dt = self.T/self.n_steps
        N = np.random.randn(self.n_steps, self.n_paths)
        S = np.concatenate((self.S_0*np.ones((1, self.n_paths)), np.zeros((self.n_steps, self.n_paths))), axis=0)   
        
        if sim_type=="EM":
            if analytic_EM==True:
                a = (self.sigma**2)/self.alpha*(np.exp(-self.alpha*dt)-np.exp(-2*self.alpha*dt))
                b = self.mu*(self.sigma**2)/(2*self.alpha)*(1-np.exp(-self.alpha*dt))**2
                for i in range(0, self.n_steps):
                    S[i+1,:] = self.mu + (S[i,:]-self.mu)*np.exp(-self.alpha*dt) + np.sqrt(a*S[i,:]+b)*N[i,:]
                    S[i+1,:] = np.maximum(S[i+1,:], np.zeros((1, self.n_paths)))
            else:
                for i in range(0, self.n_steps):
                    S[i+1,:] = S[i,:] + self.alpha*(self.mu-S[i,:])*dt + self.sigma*np.sqrt(S[i,:]*dt)*N[i,:]
                    S[i+1,:] = np.maximum(S[i+1,:], np.zeros((1, self.n_paths)))
        elif sim_type=="E":
            d = 4*self.alpha*self.mu/(self.sigma**2)
            k = (self.sigma**2)*(1-np.exp(-self.alpha*dt))/(4*self.alpha)
            for i in range(0, self.n_steps):
                delta = 4*self.alpha*S[i,:]/((self.sigma**2)*(np.exp(self.alpha*dt)-1))
                S[i+1,:] = np.random.noncentral_chisquare(d, delta, (1, self.n_paths))*k
        else:
            raise TypeError("sim_type can only take values in [EM, E]")
        
        return S
    
    def get_expectation(self):
        """Returns the expectation, E[S], for the Feller Square-Root Process"""
        ES = self.mu + (self.S_0-self.mu)*np.exp(-self.alpha*t)
        return ES
    
    def get_variance(self):
        """Returns the variance, Var[S], for the Feller Square-Root Process"""
        VarS = ((self.sigma**2)*(np.exp(-self.alpha*self.t)-np.exp(-self.alpha*2*self.t))*self.S_0/self.alpha +
               (self.sigma**2)*np.exp(-self.alpha*2*self.t)*(np.exp(self.alpha*self.t)-1)**2*self.mu/(2*self.alpha))
        return VarS
    
    def simulate(self, sim_type="EM", analytic_EM=False, plot_expected=False):
        """Returns the plot of the random paths taken by the Feller Square-Root Process"""
        plotting_df = pd.DataFrame(self.get_paths(sim_type, analytic_EM))
        if plot_expected==True:
            plotting_df["Expected Path"]=self.get_expectation()
        fig = px.line(plotting_df, labels={"value":"Value of S", "variable":"Paths"})
        return fig.show()




#Constant Elasticity of Variance

class Constant_Elasticity_Of_Variance:
    """dS = mu*S*dt + sigma*S^(beta+1)*dW
    Model used to reproduce the volatility smile effect
    Requires numpy, pandas and plotly.express"""
    def __init__(self, mu, sigma, n_paths, n_steps, t, T, S_0, beta):
        self.mu = mu
        self.sigma = sigma
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.t = t
        self.T =T
        self.S_0 = S_0
        self.beta = beta
    
    def get_paths(self):
        """Returns the paths, S, for the Constant Elasticity of Variance Process"""
        dt = self.T/self.n_steps
        dW = np.sqrt(dt)*np.random.randn(self.n_steps, self.n_paths)
        S = np.concatenate((self.S_0*np.ones((1, self.n_paths)), np.zeros((self.n_steps, self.n_paths))), axis=0) 
        
        for i in range(0, self.n_steps):
            S[i+1,:] = S[i,:] + self.mu*S[i,:]*dt + self.sigma*(S[i,:]**(self.beta+1))*dW[i,:]
            S[i+1,:] = np.maximum(S[i+1,:], np.zeros((1, self.n_paths)))
        
        return S
    
    def get_expectation(self):
        """Returns the expectation, E[S], for the Constant Elasticity of Variance Process"""
        ES = self.S_0*np.exp(self.mu*self.t)
        return ES
    
    def simulate(self, plot_expected=False):
        """Returns the plot of the random paths taken by the Constant Elasticity of Variance Process"""
        plotting_df = pd.DataFrame(self.get_paths())
        if plot_expected==True:
            plotting_df["Expected Path"]=self.get_expectation()
        fig = px.line(plotting_df, labels={"value":"Value of S", "variable":"Paths"})
        return fig.show()




##########################################################################################################################################################################################################################

#Heston Stochastic Volatility Process

class Heston_Stochastic_Volatility:
    
    def __init__(self, mu, k, theta, epsilon, n_paths, n_steps, t, T, S_0, v_0, rho):
        self.mu = mu
        self.k = k
        self.theta = theta
        self.epsilon = epsilon
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.t = t
        self.T =T
        self.S_0 = S_0
        self.v_0 = v_0
        self.rho = rho
    
    def get_paths(self):
        """Returns the paths, S, for the Heston Stochastic Volatility Process"""
        dt = self.T/self.n_steps
        Nv = np.random.randn(self.n_steps, self.n_paths)
        N = np.random.randn(self.n_steps, self.n_paths)
        NS = self.rho*Nv + np.sqrt(1-self.rho**2)*N
        
        v = np.concatenate((self.v_0*np.ones((1, self.n_paths)), np.zeros((self.n_steps, self.n_paths))), axis=0)
        S = np.concatenate((self.S_0*np.ones((1, self.n_paths)), np.zeros((self.n_steps, self.n_paths))), axis=0)
        
        a = (self.epsilon**2)/self.k*(np.exp(-self.k*dt)-np.exp(-2*self.k*dt))
        b = self.theta*(self.epsilon**2)/(2*self.k)*(1-np.exp(-self.k*dt))**2
        for i in range(0, self.n_steps):
            v[i+1,:] = self.theta + (v[i,:]-self.theta)*np.exp(-self.k*dt) + np.sqrt(a*v[i,:]+b)*Nv[i,:]
            v[i+1,:] = np.maximum(v[i+1,:], np.zeros((1, self.n_paths)))
        
        for j in range(0, self.n_steps):
            S[j+1,:] = S[j,:] + (self.mu-0.5*v[j,:])*dt + self.epsilon*np.sqrt(v[j,:]*dt)*NS[j,:]
            S[j+1,:] = np.maximum(S[j+1,:], np.zeros((1, self.n_paths)))
        
        return S, v
    
    def get_expectation(self):
        """Returns the expectation, E[S], for the Heston Stochastic Volatility Process"""
        ES = (self.S_0 + (self.mu-0.5*self.theta)*self.t
                         + (self.theta-self.v_0)*(1-np.exp(-self.k*self.t))/(2*self.k))
        return ES
    
    def simulate(self, plot_expected=False):
        """Returns the plot of the random paths taken by the Heston Stochastic Volatility Process"""
        plotting_df = pd.DataFrame(self.get_paths()[0])
        if plot_expected==True:
            plotting_df["Expected Path"]=self.get_expectation()
        fig = px.line(plotting_df, labels={"value":"Value of S", "variable":"Paths"})
        return fig.show()




##########################################################################################################################################################################################################################

#Variance Gamma Process

class Variance_Gamma_Process:
    """dS = mu*dG(t) + sigma*dW(dG(t))
    Model used in option pricing
    Requires numpy, pandas and plotly.express"""
    def __init__(self, mu, sigma, n_paths, n_steps, t, T, S_0, rate):
        self.mu = mu
        self.sigma = sigma
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.t = t
        self.T =T
        self.S_0 = S_0
        self.rate = rate
    
    def get_paths(self):
        """Returns the paths, S, for the Variance Gamma Process"""
        dt = self.T/self.n_steps
        kappa = 1/self.rate
        dG = np.random.gamma(dt/kappa, kappa, (self.n_steps, self.n_paths))
        
        dS = self.mu*dG+self.sigma*np.random.randn(self.n_steps, self.n_paths)*np.sqrt(dG)
        
        dS = np.insert(dS, 0, self.S_0, axis=0)
        S = np.cumsum(dS, axis=0)
        
        return S
    
    def get_expectation(self):
        """Returns the expectation, E[S], for the Variance Gamma Process"""
        ES = self.mu*self.t+self.S_0 
        return ES
    
    def get_variance(self):
        """Returns the variance, Var[S], for the Variance Gamma Process"""
        VarS = (self.sigma**2 + (self.mu**2)/self.rate)*t
        return VarS
    
    def simulate(self, plot_expected=False):
        """Returns the plot of the random paths taken by the Variance Gamma Process"""
        plotting_df = pd.DataFrame(self.get_paths())
        if plot_expected==True:
            plotting_df["Expected Path"]=self.get_expectation()
        fig = px.line(plotting_df, labels={"value":"Value of S", "variable":"Paths"})
        return fig.show()




##########################################################################################################################################################################################################################

#Merton Jump-Diffusion Process

class Merton_Jump_Diffusion_Process:
    """S = (mu-0.5*sigma^2)*t + sigma*W(t) + sum_{i=1}^{N(t)} Z_i
    Model describes stock price with continuous movement that have rare large jumps
    Requires numpy, pandas and plotly.express"""
    def __init__(self, muS, sigmaS, muJ, sigmaJ, lambdaJ, n_paths, n_steps, t, T, S_0):
        self.muS = muS
        self.sigmaS = sigmaS
        self.muJ = muJ
        self.sigmaJ = sigmaJ
        self.lambdaJ = lambdaJ
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.t = t
        self.T =T
        self.S_0 = S_0
    
    def get_paths(self):
        """Returns the paths, S, for the Merton Jump-Diffusion Process"""
        dt = self.T/self.n_steps
        dX = (self.muS-0.5*self.sigmaS**2)*dt + self.sigmaS*np.sqrt(dt)*np.random.randn(self.n_steps, self.n_paths)
        dP = np.random.poisson(self.lambdaJ*dt, (self.n_steps, self.n_paths))
        dJ = self.muJ*dP + self.sigmaJ*np.sqrt(dP)*np.random.randn(self.n_steps, self.n_paths)
        
        dS = dX + dJ
        
        dS = np.insert(dS, 0, self.S_0, axis=0)
        S = np.cumsum(dS, axis=0)
        
        return S
    
    def get_expectation(self):
        """Returns the expectation, E[S], for the Merton Jump-Diffusion Process"""
        ES = (self.muS+self.lambdaJ*self.muJ)*t+self.S_0
        return ES
    
    def get_variance(self):
        """Returns the variance, Var[S], for the Merton Jump-Diffusion Process"""
        VarS = (self.muS**2+self.lambdaJ*(self.muJ**2+self.sigmaJ**2))*t
        return VarS
    
    def simulate(self, plot_expected=False):
        """Returns the plot of the random paths taken by the Merton Jump-Diffusion Process"""
        plotting_df = pd.DataFrame(self.get_paths())
        if plot_expected==True:
            plotting_df["Expected Path"]=self.get_expectation()
        fig = px.line(plotting_df, labels={"value":"Value of S", "variable":"Paths"})
        return fig.show()




##########################################################################################################################################################################################################################

#Kou Jump-Diffusion Model

class Kou_Jump_Diffusion_Process:
    """S = mu*t +sigma*W(t) + sum_{i=1}^{N(t)} Z_i
    Model describes stock price with continuous movement that have rare large jumps, with the jump sizes following a double 
    exponential distribution
    Requires numpy, pandas and plotly.express"""
    def __init__(self, mu, sigma, lambdaN, eta1, eta2, p, n_paths, n_steps, t, T, S_0):
        self.mu = mu
        self.sigma = sigma
        self.lambdaN = lambdaN
        self.eta1 = eta1
        self.eta2 = eta2
        self.p = p
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.t = t
        self.T =T
        self.S_0 = S_0
    
    def get_paths(self):
        """Returns the paths, S, for the Kou Jump-Diffusion Process"""
        dt = self.T/self.n_steps
        dX = (self.mu-0.5*self.sigma**2)*dt + self.sigma*np.sqrt(dt)*np.random.randn(self.n_steps, self.n_paths)
        dP = np.random.poisson(self.lambdaN*dt, (self.n_steps, self.n_paths))
        
        #Bilateral Exponential R.V.       
        U = np.random.uniform(0,1, (self.n_steps, self.n_paths))
        Z = np.zeros((self.n_steps, self.n_paths))
        for i in range(0, len(U[0])):
            for j in range(0, len(U)):
                if U[j,i]>=self.p:
                    Z[j,i]=(-1/self.eta1)*np.log((1-U[j,i])/self.p)
                elif U[j,i]<self.p:
                    Z[j,i]=(1/self.eta2)*np.log(U[j,i]/(1-self.p))
        
        
        dJ = (np.exp(Z)-1)*dP
        dS = dX + dJ
        
        dS = np.insert(dS, 0, self.S_0, axis=0)
        S = np.cumsum(dS, axis=0)
        
        return S
    
    def get_expectation(self):
        """Returns the expectation, E[S], for the Kou Jump-Diffusion Process"""
        ES = (self.mu+self.lambdaN*(self.p/self.eta1-(1-self.p)/self.eta2))*t+self.S_0
        return ES
    
    def get_variance(self):
        """Returns the variance, Var[S], for the Kou Jump-Diffusion Process"""
        VarS = (self.sigma**2+2*self.lambdaN*(self.p/(self.eta1**2)+(1-self.p)/(self.eta2**2)))*t
        return VarS
    
    def simulate(self, plot_expected=False):
        """Returns the plot of the random paths taken by the Kou Jump-Diffusion Process"""
        plotting_df = pd.DataFrame(self.get_paths())
        if plot_expected==True:
            plotting_df["Expected Path"]=self.get_expectation()
        fig = px.line(plotting_df, labels={"value":"Value of S", "variable":"Paths"})
        return fig.show()




##########################################################################################################################################################################################################################

if __name__ == '__main__':

    #Useful Variables for Tests
    npaths = 50
    nsteps = 200
    T = 1
    dt = T/nsteps
    t = np.arange(0, T+dt, dt)

    Arithmetic_Brownian_Motion(0.05, 0.4, npaths, nsteps, t, T, 200).simulate(plot_expected=True)
    #Geometric_Brownian_Motion(0.2, 0.4, npaths, nsteps, t, T, 500)
    #Ornstein_Uhlenbeck_Process(10, 0.07, 0.1, npaths, nsteps, t, T, 0.05)
    #Brownian_Bridge(1, 2, 0.5, npaths, nsteps, t, T)
    #Feller_Square_Root_Process(5, 0.07, 0.265, npaths, nsteps, t, T, 0.03)
    #Constant_Elasticity_Of_Variance(0.2, 0.4, npaths, nsteps, t, T, 500, 0)
    #Heston_Stochastic_Volatility(0.1, 5, 0.07, 0.2, npaths, nsteps, t, T, 100, 0.03, 0)
    #Variance_Gamma_Process(0.2, 0.3, npaths, nsteps, t, T, 0, 1/0.05)
    #Merton_Jump_Diffusion_Process(0.2, 0.3, -0.1, 0.15, 0.5, npaths, nsteps, t, T, 0)
    #Kou_Jump_Diffusion_Process(0.2, 0.3, 0.5, 9, 5, 0.5, npaths, nsteps, t, T, 0)
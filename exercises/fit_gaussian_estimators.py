from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    sigma = 1
    samples = np.random.normal(mu, sigma, 1000)
    uni_gauss = UnivariateGaussian()
    uni_gauss.fit(samples)
    print(uni_gauss.mu_, uni_gauss.var_)

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100).astype(int)
    estimated_mus_diff = []
    for m in ms:
        uni_gauss.fit(samples[:m])
        estimated_mus_diff.append(np.abs(uni_gauss.mu_ - mu))

    go.Figure([go.Scatter(x=ms, y=estimated_mus_diff, mode='lines', name="r$|\hat\mu - \mu|$"),
           go.Scatter(x=ms, y=[0]*len(ms), mode='lines', name=r'0')],
          layout=go.Layout(title=r"$\text{Diff of Estimated Expectation and Actual Expectation by Number Of Samples}$", 
                  xaxis_title="$m\\text{ - number of samples}$", 
                  yaxis_title="r$|\hat\mu - \mu|$",
                  height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    sorted_samples = np.sort(samples)
    pdfs = uni_gauss.pdf(sorted_samples)
    go.Figure([go.Scatter(x=sorted_samples, y=pdfs, mode='lines', name=r'$pdf(x)$')],
          layout=go.Layout(title=r"$\text{ PDF function of randomly sampled X from normal distribution}$",
                  xaxis_title="$\\text{ random samples from N(10,1)}$",
                  yaxis_title="r$pdf(x)$",
                  height=500)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0,0,4,0])
    sigma = np.array([
            [1,0.2,0,0.5],
            [0.2,2,0,0],
            [0,0,1,0],
            [0.5,0,0,1]
        ])
    
    samples = np.random.multivariate_normal(mu, sigma, 1000)
    multi_gauss = MultivariateGaussian()
    multi_gauss.fit(samples)
    print(multi_gauss.mu_)
    print(multi_gauss.cov_)

    # Question 5 - Likelihood evaluation
    ms = np.linspace(-10, 10, 200).astype(np.float64)
    log_MLEs = []
    for f1 in ms:
        for f3 in ms:
            mu_ = np.array([f1, 0, f3, 0])
            log_likelihood = MultivariateGaussian.log_likelihood(mu_, sigma, samples)
            log_MLEs.append(log_likelihood)

    log_MLEs = np.array(log_MLEs).reshape((ms.size,ms.size))
    
    fig = go.Figure([go.Histogram2dContour(x=log_MLEs[:, 0], y=log_MLEs[:, 1], 
            colorscale = 'Blues', reversescale = True, xaxis = 'x', yaxis = 'y')],
          layout=go.Layout(title=r"$\text{ Log Likelihood heatmap }$",
                  xaxis_title="r$ \\tilde{\mu}=[f_{1},0,f_{3}^{j},0]:f_{3}^{j}\\in\\left[np.linspace(-10,10,200)\\right] $",
                  yaxis_title="r$ \\tilde{\mu}=[f_{1}^{i},0,f_{3},0]:f_{1}^{i}\\in\\left[np.linspace(-10,10,200)\\right] $",
                  height=500))
    fig.show()
    

    # Question 6 - Maximum likelihood
    max_value = np.max(log_MLEs)
    locations = zip(*np.where(log_MLEs == max_value))
    print("max value",max_value)
    for loc in locations:
        print("f1 is", ms[loc[0]], "f3 is", ms[loc[1]])

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

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
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

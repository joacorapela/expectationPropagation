import numpy as np

def sample(theta, a, w, n_samples):
    uniform_samples = np.random.uniform(size=n_samples)
    answer = np.empty(shape=n_samples)
    for i in range(n_samples):
        if uniform_samples[i] < w:
            answer[i] = np.random.normal(loc=0, scale=np.sqrt(a))
        else:
            answer[i] = np.random.normal(loc=theta, scale=1.0)
    return answer


# agents/gan_augmentor.py
import numpy as np

class LightweightGAN:
    """
    Evolutionary GAN using only numpy.
    Generates synthetic OHLC candles (open, high, low, close).
    """
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=4):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Generator weights
        self.G_W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.G_b1 = np.zeros(hidden_dim)
        self.G_W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.G_b2 = np.zeros(output_dim)
        # Discriminator weights
        self.D_W1 = np.random.randn(output_dim, hidden_dim) * 0.1
        self.D_b1 = np.zeros(hidden_dim)
        self.D_W2 = np.random.randn(hidden_dim, 1) * 0.1
        self.D_b2 = 0.0

    def _relu(self, x): return np.maximum(0, x)
    def _sigmoid(self, x): return 1 / (1 + np.exp(-x))

    def generate(self, noise):
        h = self._relu(noise @ self.G_W1 + self.G_b1)
        return h @ self.G_W2 + self.G_b2

    def discriminate(self, x):
        h = self._relu(x @ self.D_W1 + self.D_b1)
        return self._sigmoid(h @ self.D_W2 + self.D_b2)

    def _mutate(self, arr, rate=0.1):
        mask = np.random.rand(*arr.shape) < rate
        arr += mask * np.random.randn(*arr.shape) * 0.05
        return arr

    def train(self, real_data, epochs=500, batch_size=32):
        """
        real_data: numpy array of shape (n_samples, 4) – normalized OHLC.
        """
        n = real_data.shape[0]
        for _ in range(epochs):
            # Train discriminator
            idx = np.random.choice(n, batch_size, replace=True)
            real = real_data[idx]
            noise = np.random.randn(batch_size, self.input_dim)
            fake = self.generate(noise)

            d_real = self.discriminate(real).mean()
            d_fake = self.discriminate(fake).mean()

            best_D = (self.D_W1.copy(), self.D_b1.copy(), self.D_W2.copy(), self.D_b2.copy())
            best_score = d_real - d_fake
            for _ in range(5):
                self._mutate(self.D_W1); self._mutate(self.D_b1); self._mutate(self.D_W2)
                self.D_b2 += np.random.randn() * 0.01
                d_real_new = self.discriminate(real).mean()
                d_fake_new = self.discriminate(fake).mean()
                score = d_real_new - d_fake_new
                if score > best_score:
                    best_score = score
                    best_D = (self.D_W1.copy(), self.D_b1.copy(), self.D_W2.copy(), self.D_b2.copy())
                self.D_W1, self.D_b1, self.D_W2, self.D_b2 = best_D

            # Train generator
            noise = np.random.randn(batch_size, self.input_dim)
            fake = self.generate(noise)
            d_fake = self.discriminate(fake).mean()

            best_G = (self.G_W1.copy(), self.G_b1.copy(), self.G_W2.copy(), self.G_b2.copy())
            best_score = d_fake
            for _ in range(5):
                self._mutate(self.G_W1); self._mutate(self.G_b1); self._mutate(self.G_W2)
                self.G_b2 += np.random.randn() * 0.01
                fake_new = self.generate(noise)
                score = self.discriminate(fake_new).mean()
                if score > best_score:
                    best_score = score
                    best_G = (self.G_W1.copy(), self.G_b1.copy(), self.G_W2.copy(), self.G_b2.copy())
                self.G_W1, self.G_b1, self.G_W2, self.G_b2 = best_G

        print(f"GAN training complete. D(real)={d_real:.2f}, D(fake)={d_fake:.2f}")

    def generate_samples(self, n=100):
        noise = np.random.randn(n, self.input_dim)
        return self.generate(noise)

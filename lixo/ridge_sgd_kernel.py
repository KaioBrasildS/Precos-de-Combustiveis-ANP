import torch
import numpy as np

# Função kernel (RBF ajustado com a equação 21.39)
def Ridgkernel(x1, x2, sigma_0=1.0, l=1.0):
    diff = np.array(x1, dtype=float) - np.array(x2, dtype=float)
    return sigma_0**2 * np.exp(-np.dot(diff, diff) / (2 * l**2))

# Implementação da Regressão Ridge com SGD para Séries Temporais Online usando PyTorch
class RidgeSGDKernelTorch:
    def __init__(self, eta=0.01, c=1.0, sigma=1.0):
        self.eta = eta  # taxa de aprendizado
        self.c = c  # parâmetro de regularização
        self.sigma = sigma  # parâmetro do kernel
        self.alpha = None  # pesos (serão inicializados depois)
        self.X_train_tensor = None  # histórico de amostras de treino

    def _compute_kernels(self, x_new):
        """Calcula os valores do kernel entre os dados de treino e a nova amostra."""
        return torch.exp(
            -torch.norm(self.X_train_tensor - x_new, dim=1) ** 2 / (2 * self.sigma ** 2)
        )

    def partial_fit(self, x_new, y_new):
        x_new_tensor = torch.tensor(x_new, dtype=torch.float32)

        # Inicializar se for a primeira amostra
        if self.X_train_tensor is None:
            self.X_train_tensor = x_new_tensor.unsqueeze(0)  # Primeira entrada
            self.alpha = torch.zeros(1, dtype=torch.float32)  # Inicializar peso
            pred_n = 0.0
        else:
            self.X_train_tensor = torch.vstack([self.X_train_tensor, x_new_tensor])  # Adiciona nova amostra
            self.alpha = torch.cat([self.alpha, torch.zeros(1)])  # Adiciona peso para a nova amostra

            # Calcula os valores do kernel
            kernels = self._compute_kernels(x_new_tensor)
            pred_n = torch.dot(self.alpha, kernels).item()  # Previsão

        # Calcular o erro
        error = y_new - pred_n

        # Atualizar os pesos
        self.alpha *= (1 - self.eta * self.c)  # Regularização dos pesos antigos
        self.alpha[-1] = self.eta * error  # Atualização do peso atual

        return pred_n

    def predict(self, x_new):
        x_new_tensor = torch.tensor(x_new, dtype=torch.float32)
        kernels = self._compute_kernels(x_new_tensor)
        return torch.dot(self.alpha, kernels).item()

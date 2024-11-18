import torch
import numpy as np

# Função kernel (RBF ajustado com a equação 21.39)
def Ridgkernel(x1, x2, sigma_0=1.0, l=1.0):
    diff = np.array(x1, dtype=float) - np.array(x2, dtype=float)
    return sigma_0**2 * np.exp(-np.dot(diff, diff) / (2 * l**2))

# Implementação da Regressão Ridge com SGD para Séries Temporais Online usando PyTorch
class RidgeSGDKernelTorchDict:
    def __init__(self, eta=0.01, c=1.0, sigma=1.0):
        self.eta = eta  # taxa de aprendizado
        self.c = c  # parâmetro de regularização
        self.sigma = sigma  # parâmetro do kernel
        self.alpha_dict = {}  # dicionário para armazenar os pesos associados às chaves
        self.X_train_dict = {}  # dicionário para armazenar os dados de treino

    def partial_fit(self, x_new, y_new, key):
        """
        Adiciona uma nova entrada ao modelo e atualiza os pesos.
        
        Args:
        - x_new: nova amostra (array ou lista de características).
        - y_new: valor alvo associado à amostra.
        - key: identificador único para a amostra.
        """
        x_new_tensor = torch.tensor(x_new, dtype=torch.float32)
        self.X_train_dict[key] = x_new_tensor  # Adiciona nova entrada ao conjunto de treino
        if key not in self.alpha_dict:
            self.alpha_dict[key] = torch.tensor(0.0, dtype=torch.float32)  # Inicializa o peso correspondente

        # Calcular a previsão
        pred_n = 0
        for k, x_train in self.X_train_dict.items():
            if k != key:  # Considera apenas os pontos anteriores
                kernel_value = torch.exp(-torch.norm(x_train - x_new_tensor) ** 2 / (2 * self.sigma ** 2))
                pred_n += self.alpha_dict[k] * kernel_value

        # Calcular o erro
        error = y_new - pred_n

        # Atualizar os pesos
        for k in list(self.alpha_dict.keys()):
            if k != key:
                self.alpha_dict[k] *= (1 - self.eta * self.c)  # Atualização regularizada
        self.alpha_dict[key] = self.eta * error  # Atualiza o peso para a nova amostra

    def predict(self, x_new):
        """
        Realiza uma previsão para uma nova entrada.
        
        Args:
        - x_new: nova entrada (array ou lista de características).
        
        Returns:
        - Previsão como float.
        """
        x_new_tensor = torch.tensor(x_new, dtype=torch.float32)
        prediction = 0
        for k, x_train in self.X_train_dict.items():
            kernel_value = torch.exp(-torch.norm(x_train - x_new_tensor) ** 2 / (2 * self.sigma ** 2))
            prediction += self.alpha_dict[k] * kernel_value
        return prediction.item()

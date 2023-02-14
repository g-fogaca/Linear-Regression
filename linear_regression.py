import numpy as np
class LinearRegression:
    
    def fit(self, X_train, y_train):
        '''

        Parameters
        ----------
        X_train : matriz de observações das variáveis independentes
        y_train : vetor de observações da variável dependente
            DESCRIPTION.

        Returns
        -------
        theta : vetor com os parâmetros da regressão linear

        '''
        # inserir coluna "ones"
        ones = np.ones(X_train.shape[0])
        X_train = np.insert(X_train, 0, ones, axis = 1)
        
        # Calcula os estimadores de mínimos quadrados        
        # theta = (X^T.X)^(-1).X^T.y
        inversa = np.linalg.inv(np.dot(X_train.T, X_train))
        theta = np.dot(np.dot(inversa, X_train.T), y_train)
        
        return theta
    
    def predict(self, X_test):
        '''
        
        Parameters
        ----------
        X_test : matriz com features dos valores a serem previstos

        Returns
        -------
        y_pred : predições de y

        '''
        # inserir coluna "ones"
        ones = np.ones(X_test.shape[0])
        X_test = np.insert(X_test, 0, ones, axis = 1)
        
        # Faz as predições para os valores de X_test
        y_pred = np.dot(X_test, self.theta)
        
        return y_pred

import pickle
from abc import abstractmethod
from gurobipy import *
from metrics import PairsExplained

import numpy as np


class BaseModel(object):
    """
    Base class for models, to be used as coding pattern skeleton.
    Can be used for a model on a single cluster or on multiple clusters"""

    def __init__(self):
        """Initialization of your model and its hyper-parameters"""
        pass

    @abstractmethod
    def fit(self, X, Y):
        """Fit function to find the parameters according to (X, Y) data.
        (X, Y) formatting must be so that X[i] is preferred to Y[i] for all i.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # Customize what happens in the fit function
        return

    @abstractmethod
    def predict_utility(self, X):
        """Method to call the decision function of your model

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        # Customize what happens in the predict utility function
        return

    def predict_preference(self, X, Y):
        """Method to predict which pair is preferred between X[i] and Y[i] for all i.
        Returns a preference for each cluster.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of preferences for each cluster. 1 if X is preferred to Y, 0 otherwise
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return (X_u - Y_u > 0).astype(int)

    def predict_cluster(self, X, Y):
        """Predict which cluster prefers X over Y THE MOST, meaning that if several cluster prefer X over Y, it will
        be assigned to the cluster showing the highest utility difference). The reversal is True if none of the clusters
        prefer X over Y.
        Compared to predict_preference, it indicates a cluster index.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, ) index of cluster with highest preference difference between X and Y.
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return np.argmax(X_u - Y_u, axis=1)

    def save_model(self, path):
        """Save the model in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the file in which the model will be saved
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(clf, path):
        """Load a model saved in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the path to the file to load
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model


class RandomExampleModel(BaseModel):
    """Example of a model on two clusters, drawing random coefficients.
    You can use it to understand how to write your own model and the data format that we are waiting for.
    This model does not work well but you should have the same data formatting with TwoClustersMIP.
    """

    def __init__(self):
        self.seed = 444
        self.weights = self.instantiate()

    def instantiate(self):
        """No particular instantiation"""
        return []

    def fit(self, X, Y):
        """fit function, sets random weights for each cluster. Totally independant from X & Y.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        np.random.seed(self.seed)
        indexes = np.random.randint(0, 2, (len(X)))
        num_features = X.shape[1]
        weights_1 = np.random.rand(num_features)
        weights_2 = np.random.rand(num_features)

        weights_1 = weights_1 / np.sum(weights_1)
        weights_2 = weights_2 / np.sum(weights_2)
        self.weights = [weights_1, weights_2]
        return self

    def predict_utility(self, X):
        """Simple utility function from random weights.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        

        return np.stack([np.dot(X, self.weights[0]), np.dot(X, self.weights[1])], axis=1)
        


class TwoClustersMIP(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_pieces, n_clusters):
        """Initialization of the MIP Variables

        Parameters
        ----------
        n_pieces: int
            Number of pieces for the utility function of each feature.
        n°clusters: int
            Number of clusters to implement in the MIP.
        """
        self.seed = 123
        self.K = n_clusters
        self.L = n_pieces
        self.N_CRITERIA = 4
        self.epsilon = 0.0001
        self.model = self.instantiate()
       

    def instantiate(self):
        """Instantiation of the MIP Variables - To be completed."""

        # Instanciation du modèle
        m = Model("MIP model")
        return m

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """

        #Ordonnées des points d'inflexion en variables
        ordonnees_infl = [[self.model.addVar(name=f"u_{i}_{l}") for l in range(self.L)] for i in range(self.N_CRITERIA)]

        P = len(X)

        sigma_plus_x = [self.model.addVar(name=f"sigma+_x_{i}") for i in range(P)]
        sigma_minus_x = [self.model.addVar(name=f"sigma-_x_{i}") for i in range(P)]

        sigma_plus_y = [self.model.addVar(name=f"sigma+_y_{i}") for i in range(P)]
        sigma_minus_y = [self.model.addVar(name=f"sigma-_y_{i}") for i in range(P)]

        #On met à jour le modèle avec les nouvelles variables
        self.model.update()

        # On détermine les valeurs miniamles et maximales pour chaque critère d'évaluation
        A = np.concatenate((X,Y), axis = 0)
        mins = A.min(axis=0)
        maxs = A.max(axis=0)

        def xl(i, l):
            return mins[i] + l*(maxs[i] - mins[i])/self.L
        
        #Fonction qui permet de récupérer la somme des poids
        def weight_sum(ordo):
            somme = 0
            for i in range(self.N_CRITERIA):
                somme += ordo[i][-1]
            return somme 
    
        def u_i(i,j,X):
            x = X[j][i]

            if x == maxs[i]:
                return ordonnees_infl[i][self.L - 1]

            else : 
                #On récupère la première des 5 abscisses de points d'inflexion inférieure à la valeur d'intérêt 
                for l in range(self.L):
                    if x >= xl(i, l):
                        li = l
                        break

                

                #On retourne la valeur de la fonction en la valeur d'intérêt 
                res =  (ordonnees_infl[i][li+1] - ordonnees_infl[i][li]) / (xl(i,li+1) - xl(i,li)) * (x - xl(i,li)) + ordonnees_infl[i][li]
                return res
        
        def u(j, X):
            return sum(u_i(i,j, X) for i in range(self.N_CRITERIA))
        
        #Fonction qui retourne la valeur de la fonction d'utilité pour un élément avec les erreurs sigma
        def tot_X(j,X):
            return u(j, X) - sigma_plus_x[j] + sigma_minus_x[j]
        
        def tot_Y(j,Y):
            return u(j, Y) - sigma_plus_y[j] + sigma_minus_y[j]
        
        #Contrainte pour garantir les premières ordonnées nulles pour chaque critère 
        for i in range(self.N_CRITERIA):
            self.model.addConstr(ordonnees_infl[i][0] == 0)

        #Contrainte pour garantir la croissance des fonctions d'utilité
        for l in range(self.N_CRITERIA):
            for i in range(self.L - 1):
                self.model.addConstr(ordonnees_infl[i][l+1] - ordonnees_infl[i][l] >= self.epsilon)


        #Contrainte pour garantir que la somme des poids vaut 1
        for k in range(self.K):
            self.model.addConstr(weight_sum(ordonnees_infl) == 1)
        
        
        for i1 in range(P):
                self.model.addConstr(tot_X(i1, X) - tot_Y(i1, Y) >= 2*self.epsilon)
                self.model.addConstr(sigma_minus_x[i1] >= 0)
                self.model.addConstr(sigma_minus_y[i1] >= 0)
                self.model.addConstr(sigma_plus_x[i1] >= 0)
                self.model.addConstr(sigma_plus_y[i1] >= 0)
        #Mise à jour du modèle après ajout des contraintes
        self.model.update()

        self.model.setObjective(sum(sigma_minus_x) + sum(sigma_minus_y) + sum(sigma_plus_x) + sum(sigma_plus_y), GRB.MINIMIZE) 
        
        # Résolution du PL
        self.model.optimize()


        return self.model

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        # To be completed
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.
        return


class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self):
        """Initialization of the Heuristic Model.
        """
        self.seed = 123
        self.models = self.instantiate()

    def instantiate(self):
        """Instantiation of the MIP Variables"""
        # To be completed
        return

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # To be completed
        return

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        # To be completed
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.
        return

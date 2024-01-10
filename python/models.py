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
        ordonnees_infl = [[[self.model.addVar(name=f"u_{k}_{i}_{l}") for l in range(self.L)] for i in range(self.N_CRITERIA)] for k in range(self.K)]
        

        #On ajoute une variable binaire pour modéliser le fait qu'un paire est effectivement expliquée par le modèle 
        b = [[self.model.addVar(vtype=GRB.BINARY, name=f"b_{j}") for j in range(2000)] for k in range(self.K)]

        #On met à jour le modèle avec les nouvelles variables
        self.model.update()

        #On détermine les valeurs maximales et minimales pour chaque critère d'évaluation 
        A = np.concatenate((X,Y), axis = 0)
        mins = A.min(axis=0)
        maxs = A.max(axis=0)


        #Fonction pour récupèrer les abscisses des points d'inflexion
        def xl(i, l):
            return mins[i] + l*(maxs[i] - mins[i])/self.L
        
        #Fonction qui permet de récupérer la somme des poids pour chaque critère (pour la contrainte de la somme des poids valant 1)
        def weight_sum(k, ordo):
            somme = 0
            for i in range(self.N_CRITERIA):
                somme += ordo[k][i][-1]
            return somme 

        



        #Fonction qui renvoie la valeur de la fonction d'utilité pour un critère i en n'importe quel point X[j] 
        def u_k_i(k,i,j,X):
            x = X[j][i]

            if x == maxs[i]:
                return ordonnees_infl[k][i][self.L - 1]

            else : 
                #On récupère la première des 5 abscisses de points d'inflexion inférieure à la valeur d'intérêt 
                for l in range(self.L):
                    if x >= xl(i, l):
                        li = l
                        break

                

                #On retourne la valeur de la fonction en la valeur d'intérêt 
                res =  (ordonnees_infl[k][i][li+1] - ordonnees_infl[k][i][li]) / (xl(i,li+1) - xl(i,li)) * (x - xl(i,li)) + ordonnees_infl[k][i][li]

                return res

        #Fonction qui renvoie la valeur de la somme des utilités pour un produit
        def u_k(k,j, X):
            return sum(u_k_i(k,i,j, X) for i in range(self.N_CRITERIA))
        
        #Fontion pour retourner la valeur de la fonction indicatrice pour l'explication des paires
        def ind(b0, b1):
            i = []
            for j in range(len(b1)):
                if b0[j] == 1 or b1[j] == 1 :
                    i.append(1)
            return i
        



        #Définition des différentes contraintes

        #Contrainte pour garantir les premières ordonnées nulles pour chaque critère 
        for k in range(self.K):
            for i in range(self.N_CRITERIA):
                self.model.addConstr(ordonnees_infl[k][i][0] == 0)

        #Contrainte pour garantir la croissance des fonctions d'utilité
        for k in range(self.K):
            for l in range(self.N_CRITERIA):
                for i in range(self.L - 1):
                    self.model.addConstr(ordonnees_infl[k][i][l+1] - ordonnees_infl[k][i][l] >= self.epsilon)


        #Contrainte pour garantir que la somme des poids vaut 1
        for k in range(self.K):
            self.model.addConstr(weight_sum(k, ordonnees_infl) == 1)


        # # Constants
        # # M is chosen to be as small as possible given the bounds on x and y. On choisit une valeur proche de 1
        # M = 1 + self.epsilon

        # # If x > y, then b = 1, otherwise b = 0
    
        # for j in range(len(X)):
        #     #On ajoute les contraintes qui permettent de vérifier si une paire est expliquée ou non. Si le premier cluster n'explique pas la paire (b[j] ==0)
        #     #Alors on vérifie que le deuxième l'explique. Sinon on ne se préoccupe pas du deuxième cluster.  
        #     self.model.addConstr(u_k(0,j,X) >= u_k(0,j,Y) + 2*self.epsilon - M * (1 - b[0][j]), name="bigM_constr1")
        #     self.model.addConstr(u_k(0,j,X) <= u_k(0,j,Y) + self.epsilon + M * b[0][j], name="bigM_constr2")
        #     self.model.addConstr((b[0][j]== 0) >> (u_k(1,j,X) >= u_k(1,j,Y) + 2*self.epsilon - M * (1 - b[1][j])))
        #     self.model.addConstr((b[0][j]== 0) >> (u_k(1,j,X) >= u_k(1,j,Y) + self.epsilon + M * b[1][j]))
        #     self.model.addConstr((b[0][j]== 1) >> (b[1][j] == 0))
            

        #Fonction qui permet de retourner un liste contenant les utilités en chaque produit en fonction du cluster k    
        def list_utilities_per_dataset(X):
            Ux = np.array([[],[]])
            
            for j in range(len(X)):
                Ux[0] = np.append(Ux[0], u_k(0,j,X))
                Ux[1] = np.append(Ux[1], u_k(1,j,X))
            return Ux 

    

        

        

        #Mise à jour du modèle après ajout des contraintes
        self.model.update()
        
        # Fonction Objectif
        self.model.setObjective( np.sum(np.sum(list_utilities_per_dataset(X) - list_utilities_per_dataset(Y) > 0, axis=1) > 0) , GRB.MAXIMIZE)
                
          
        
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

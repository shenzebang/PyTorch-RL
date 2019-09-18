from fbpca import pca, diffsnorm
from numpy import linalg as LA


class LONuclearNormBall:
    def __init__(self,
                 radius=1
                 ):
        self.radius = radius

    def lo_oracle(self, target_matrix):
        """
        Y = argmax_{||Y||_*<=self.radius} X'*Y
        :param target_matrix: ndarray
        :return:
        Y: ndarray
        """
        (U, s, Va) = pca(target_matrix, 1, True)
        # err = diffsnorm(target_matrix, U, s, Va)
        # print("in lO")
        # print(target_matrix.shape)
        # print(err/LA.norm(target_matrix))
        # print(LA.norm(target_matrix)/s)
        return self.radius * U * Va


class LODebug:
    def __init__(self,
                 radius=1
                 ):
        self.radius = radius

    def lo_oracle(self, target_matrix):
        return target_matrix

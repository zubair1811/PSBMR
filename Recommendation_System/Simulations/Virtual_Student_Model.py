import numpy as np
from PSBMR_Framework.M_Matrix import R_table


class Student():
    """
    Model of P and Q students (depending on lambdas parameters)
    """

    def __init__(self, R_table_model, initKC, learning_rates, alpha, beta, lambdas=None):
        """
        Params :
            R_table_model :
                KC requirement for each activity (R_table object)

            initKC :
                Initial KC values
            learning_rates :
                Rate of learning for each KC
            alpha :
                KC update additive parameter
            beta :
                KC update multiplicative parameter
            lambdas :
                lambda vector (constant to 1 if Q_student)
        """
        self.R_table_model = R_table_model
        self.initKC = initKC
        self.KC = initKC
        self.learning_rates = learning_rates
        self.alpha = alpha
        self.beta = beta
        self.n_a_list = R_table_model.n_a  # tell the total no. of activities Ex: [6,3, 2, 2]  tabls rows=[1st,2nd,3rd, 4th]
        self.n_c = R_table_model.n_c
        self.n_p = R_table_model.n_p

        self.lambdas = lambdas
        self.possible_activities = R_table_model.enumerate_activities()  # Gives all acitives excercise generated  by R-table function

    def exercize(self, activity):

        success_prob, q = self.prob_success(activity)
        # print(q)
        # print(success_prob)
        # print(np.random.uniform())
        success = np.random.uniform() < success_prob

        if success:  # update KC using formula KC = KC +learning rate *(q - KC)
            self.KC = self.KC + self.learning_rates * \
                      np.maximum(q - self.KC, 0)

        return success

    def prob_success(self,
                     activity):  # it gets the competence real value from the R table and againt competence mentioned success prop. appr. 0.9983
        """
        Activity must be a numpy array of shape (n_p,)
        returns success_probs and expected reward
        """
        assert (activity.shape == (self.n_p,))
        q = self.R_table_model.get_KCVector(
            activity)  # return the p of the get_KCVector function real competence of the R table [c1,c2,c3,c4,...]
        # print(q)
        success_probs = self.get_lambdas(activity) / (
                    1 + np.exp(-self.beta * (self.KC - q) - self.alpha))  # Student modelisation
        success_prob = np.prod(success_probs) ** (1. / success_probs.shape[0])

        return success_prob, q

    def get_lambdas(self, activity):  # check the acivity equal to 1 or not if 1 return 0 otherwise 1

        res = 1
        if (self.lambdas is None):
            res = 1
        else:  # work if lamdas=1
            for act in self.lambdas:
                if np.all(np.equal(act,
                                   activity)):  # its check the activity if all the activity [1 1 1 1] return 0 otherwise 1

                    return 0

        return res

    def get_best_activity(self):
        """
        Returns the best activity in terms of expected reward and its expected reward
        """
        current_max = -float("inf")
        print(current_max)

        for activity in self.possible_activities:  # [6,3,2,2] total 72 activites
            success_prob, q = self.prob_success(activity)
            expected_reward = success_prob * np.sum(np.maximum(q - self.KC, 0)) + \
                              (1 - success_prob) * np.sum((np.minimum(q - self.KC, 0)))
            if expected_reward > current_max:
                current_max = expected_reward
                best_activity = activity

        return best_activity, current_max

    def reset(self):
        self.KC = self.initKC
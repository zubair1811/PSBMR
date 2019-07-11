

import numpy as np


class R_table():
    """
    R_table model, KC requirement for each activity
        - Factorized : when each activity is parametrized by n_p parameters
        - Not factorized : the value can directly be read in a table n_a*n_c
        # n_c Knowledge competence KC
        # n_p pulling simultaneosly MAB
    """

    def __init__(self, list_tables):

        self.list_tables = list_tables
        self.n_p = len(list_tables)
        self.n_c = list_tables[0].shape[1]
        self.n_a = [table.shape[0] for table in
                    list_tables]
        self.enumerated = self.enum_recursion(self.n_a).astype(
            'int')

    def get_KCVector(self, activity):
        assert (activity.shape == (self.n_p,))
        p = np.ones(self.n_c)
        # print(p)
        # print("start")
        for i in range(self.n_p):
            # print("rount {}".format(i))
            # print(self.list_tables[i])
            # print("activity")
            # print(activity[i])
            p = p * self.list_tables[i][activity[i],
                    :]
            # print("p value ")
            # print(p)
        # print("last value ")
        return (p)  # return q(a)

    def enumerate_activities(self):
        return (self.enumerated)

    def enum_recursion(self, list_n_a):
        """
        utils
        """

        if len(list_n_a) == 1:
            return np.arange(list_n_a[0])[:, None]


        else:
            last_recurs = self.enum_recursion(list_n_a[:-1])
            dim = last_recurs.shape
            res = np.zeros((dim[0] * list_n_a[-1], dim[1] + 1))
            for j in range(list_n_a[-1]):
                res[j * dim[0]:(j + 1) * dim[0], -1] = j
                res[j * dim[0]:(j + 1) * dim[0], :-1] = last_recurs
            # print(res)
            return (res)

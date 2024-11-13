from copy import copy
import numpy as np
from pyfunc_rust import *
import dimod
from time import time

class Problem:

    def __init__(self):
        self.obj = Formula()
        self.cond = []
        self.qubo_cond = []
        self.weights = []

    def add_objective(self, f):
        if type(f) == Variable or type(f) == Formula:
            if f.comp != "":
                raise ValueError("Error: The input must be a function, not an equation or an inequation.")
            else:
                self.obj += f
        else:
            raise TypeError("The type of input must be Variable or Formula.")
    
    def add_constraint(self, f):

        global variables, variable_range

        if type(f) == Variable or type(f) == Formula:
            if f.comp != ">=" and f.comp != "==":
                raise ValueError("The input must be an equation or an inequation, not a function.")
            elif f.order != 1:
                raise ValueError("The input must be linear condition.")
            elif f.comp == ">=":
                M = 0
                m = 0

                for n in range(len(f.lin.index_list)):
                    var_index = f.lin.index_list[n]
                    if f.lin.coef_list[n] > 0:
                        M += f.lin.coef_list[n] * variable_range[var_index][1]
                        m += f.lin.coef_list[n] * variable_range[var_index][0]
                    elif f.lin.coef_list[n] < 0:
                        M += f.lin.coef_list[n] * variable_range[var_index][0]
                        m += f.lin.coef_list[n] * variable_range[var_index][1]

                if M < - f.const:
                    raise ValueError("This condition cannot be satisfied.")
                elif m >= - f.const:
                    raise ValueError("This condition is always satisfied.")
                elif M == - f.const:
                    f.comp = "=="
                    self.cond.append(f)
                else:
                    index_aux = len(variables)
                    variables.append("")
                    variable_range.append([0, M + f.const])
                    variable_A.append(A(M + f.const))
                    for i in range(len(variable_A[-1])):
                        binindex_index.append([index_aux, i])

                    f.comp = ">="
                    f.lin.append(index_aux, -1.0)
                    self.cond.append(f)
            else:
                M = 0
                m = 0

                for n in range(len(f.lin.index_list)):
                    var_index = f.lin.index_list[n]
                    if f.lin.coef_list[n] > 0:
                        M += f.lin.coef_list[n] * variable_range[var_index][1]
                        m += f.lin.coef_list[n] * variable_range[var_index][0]
                    elif f.lin.coef_list[n] < 0:
                        M += f.lin.coef_list[n] * variable_range[var_index][0]
                        m += f.lin.coef_list[n] * variable_range[var_index][1]
                
                if M < - f.const or m > - f.const:
                    raise ValueError("This condition cannot be satisfied.")
                
                self.cond.append(f)

        else:
            TypeError("The type of input f must be Variable or Formula.")
    
    def set_weights(self, weights):
        self.weights = weights
    
    def compile(self):

        if len(self.weights) != len(self.cond):
            raise TypeError("The number of elements in weights must match the number of constraints.")
        try:
            self.weights = np.array(self.weights, dtype=np.float32)
        except:
            raise TypeError("The type of all elements in weights must be numeric.")

        self.qubo = obj_bin(self.obj.quad, self.obj.lin, index_binindexptr, variable_A, variable_range)


        for n in range(len(self.qubo_cond), len(self.cond)):
            cond_bin_lin = Lin([], [])
            cond_bin_const = self.cond[n].const
            for i in range(len(self.cond[n].lin.index_list)):
                index = self.cond[n].lin.index_list[i]
                cond_bin_const += self.cond[n].lin.coef_list[i] * variable_range[index][0]
                for j in range(len(variable_A[index])):
                    cond_bin_lin.append(index_binindexptr[index] + j, self.cond[n].lin.coef_list[i] * variable_A[index][j])

            self.qubo_cond.append(pow_cond_bin(cond_bin_lin, cond_bin_const))

        for n in range(0, len(self.cond)):
            self.qubo = add_quad(self.qubo, self.qubo_cond[n], 1.0, self.weights[n])

        return self.qubo


    def solution(self, result, solver):

        if solver == "neal":
            solution = dict()
            for (index, value) in result.items():
                if variables[int(index[0])] != "":
                    if int(index[1]) == 0:
                        add_dict(solution, variables[index[0]], variable_range[index[0]][0] + int(value) * variable_A[index[0]][index[1]])
                    else:
                        add_dict(solution, variables[index[0]], int(value) * variable_A[index[0]][index[1]])
        
        elif solver == "dimod":
            solution = dict()
            for index in range(len(binindex_index)):
                value = result[index]

                index0 = binindex_index[index][0]
                index1 = binindex_index[index][1]
                if variables[index0] != "":
                    if index1 == 0:
                        add_dict(solution, variables[index0], variable_range[index0][0] + int(value) * variable_A[index0][index1])
                    else:
                        add_dict(solution, variables[index0], int(value) * variable_A[index0][index1])

        else:
            raise ValueError("The input \'solver\' must be 'neal' or 'dimod'.")
        # ckeck whether the solution satisfies constraint conditions

        bool_solution = []

        for i in range(len(self.cond)):
            if self.cond[i].comp == "==":
                S = self.cond[i].const
                for j in range(len(self.cond[i].lin.index_list)):
                    S += self.cond[i].lin.coef_list[j] * solution[variables[self.cond[i].lin.index_list[j]]]
                bool_solution.append(S == 0)
            else:
                S = self.cond[i].const
                for j in range(len(self.cond[i].lin.index_list)):
                    if variables[self.cond[i].lin.index_list[j]] != "":
                        S += self.cond[i].lin.coef_list[j] * solution[variables[self.cond[i].lin.index_list[j]]]
                bool_solution.append(S >= 0)
                
        return solution, bool_solution

variables = []
variable_range = []
variable_A = []
index_binindexptr = [0]
binindex_index = []


class Formula:
    def __init__(self):
        self.lin = Lin([], [])
        self.quad = Quad([], [], [])
        self.const = 0.0
        self.order = 0
        self.comp = ""

    def __add__(self, other):
        return self._add(self, other, [+1, +1])
    
    def __radd__(self, other):
        return self._add(self, other, [+1, +1])

    def __sub__(self, other):
        return self._add(self, other, [+1, -1])
    
    def __rsub__(self, other):
        return self._add(self, other, [-1, +1])
    
    def _add(self, f, g, sign):

        if f.comp != "":
            raise SyntaxError("Operation is not defined for ineqation.")
        
        F = Formula()

        try:
            num = float(g)

            F.const = sign[0] * f.const + sign[1] * num
            F.order = f.order
            F.lin.index_list = f.lin.index_list
            F.quad.index_list = f.quad.index_list
            F.quad.index_list_list = f.quad.index_list_list

            if sign[0] == 1:
                F.lin.coef_list = f.lin.coef_list
                F.quad.coef_list_list = f.quad.coef_list_list

            else:
                F.lin.coef_list = times_coef_list(- 1, F.lin.coef_list)
                F.quad.coef_list_list = times_coef_list_list(- 1, F.quad.coef_list_list)
            
            return F

        except:
            pass

        if type(g) == Variable or type(g) == Formula:
            if g.comp != "":
                raise SyntaxError("Operation is not defined for ineqation.")

            F.order = max(f.order, g.order)
            F.const = sign[0] * f.const + sign[1] * g.const
            F.lin= add_lin(f.lin, g.lin, sign[0], sign[1])
            F.quad = add_quad(f.quad, g.quad, sign[0], sign[1])

            return F

        else:
            raise TypeError("Attempting to add by a value other than a formula or a numeric.")

    def __mul__(self, other):
        return self._mul(self, other)

    def __rmul__(self, other):
        return self._mul(self, other)
    
    def __pow__(self, n):
        if self.comp != "":
            raise SyntaxError("Square operation is not defined for ineqation.")
        try:
            n_ = int (n)
            if n_ != n or n < 0:
                raise ValueError("The exponent of power must be a non-negative integer.")

        except:
            raise TypeError("The exponent of power must be a non-negative integer.")

        if n == 0:
            return 1
        
        elif n == 1:
            return self

        if self.order == 0:
            F = 1
            for _ in range(n):
                F = self * F
        
        elif self.order >= 2 or n >= 3:
            raise ValueError("Terms of order three or higher cannot be handled.")

        else:
            F = Formula()
            F.order = 2
            if self.const != 0:
                F.const = self.const**2
                F.lin = times_lin(2 * self.const, self.lin)
            F.quad = pow_lin(self.lin)            
        
        return F

    def __truediv__(self, other):
        try:
            num = float(other)
            return self._mul(self, 1 / num)
        except:
            raise ValueError("Attempting to devide by a value other than a numeric value.")
    
    def _mul(self, f, g):

        if f.comp != "":
            raise SyntaxError("Operator is not defined for ineqation.")
        
        global variables

        try:
            num = float(g)

            if num != 0:

                F = Formula()

                F.order = f.order
                F.const = num * f.const

                F.lin.index_list = f.lin.index_list
                F.lin.coef_list = times_coef_list(num, f.lin.coef_list)

                F.quad.index_list = f.quad.index_list
                F.quad.index_list_list = f.quad.index_list_list
                F.quad.coef_list_list = times_coef_list_list(num, f.quad.coef_list_list)

                return F
            else:
                return Formula()
        except:
            pass
            
        if type(g) == Variable or type(g) == Formula:

            if g.comp != "":
                raise SyntaxError("Operation is not defined for ineqation.")
            
            F = Formula()            
            F.order = f.order + g.order

            if F.order >= 3:
                raise ValueError("The QUBO form must have only terms of order two or lower.")

            F.const = f.const * g.const

            F.lin = add_lin(f.lin, g.lin, g.const, f.const)
            F.quad = add_quad(f.quad, g.quad, g.const, f.const)

            return F


        else:
            raise TypeError("Attempting to multiply by a value other than a numeric value.")

    def __pos__(self):
        return self
    
    def __neg__(self):

        if self.comp != "":
            raise SyntaxError(" Operation is not defined for ineqation.")
        
        F = self
        F.const = - self.const
        F.lin = times_lin(-1, self.lin)
        F.quad = times_quad(-1, self.quad)

        return F

    def __lt__(self, other):
        # <
        return self._add_eneq(other - self - 1)

    def __le__(self, other):
        # <=
        return self._add_eneq(other - self)
    
    def __gt__(self, other):
        # >
        return self._add_eneq(self - other - 1)
    
    def __ge__(self, other):
        # >= 
        return self._add_eneq(self - other)
    
    def _add_eneq(self, F):

        if F.order >= 2:
            raise ValueError("The enequation must be linear.")
        else:
            for coef in F.lin.coef_list:
                if int(coef) != coef:
                    raise ValueError("Coefficients of Constraints must be integer.")
            
            F.comp = ">="
            return F
    
    def __eq__(self, f):
        # ==

        try: 
            F = self - f
        except:
            raise SyntaxError("Comparison operators are defined only between functions.")
        
        if F.comp != "":
            raise SyntaxError("Comparison operators are defined only between functions.")

        
        if F.order >= 2:
            raise ValueError("The equation must be linear.")
        else:
            for coef in F.lin.coef_list:
                if int(coef) != coef:
                    raise ValueError("Coefficients of Constraints must be integer.")
            
            F.comp = "=="
            return F

class Variable(Formula):
    def __init__(self, string, var_min, var_max):
        self.lin = Lin([], [])
        self.quad = Quad([], [], [])
        self.const = 0.0
        self.order = 1
        self.comp = ""

        global variables

        try:
            self.var_min = int(np.floor(var_min))
            self.var_max = int(np.ceil(var_max))

            if self.var_max <= self.var_min:
                raise ValueError("The range of variables must not be empty.")
                return
        except:
            raise ValueError("The minimum or maximum value of variables must be integer.")

        if type(string) != str:
            message = "The type of variable name {} is not str.".format(string)
            raise TypeError(message)
        elif string in variables:
            message = "There is already a variable with the same name \'{}\'.".format(string)
            raise ValueError(message)
        elif string[0] == "%":
            raise SyntaxError("The charactor \'%\' cannot be used in the first of the variable name.")
        else:
            variable_A.append(A(var_max - var_min))
            index_binindexptr.append(index_binindexptr[-1] + len(variable_A[-1]))

            index = len(variables)
            for i in range(len(variable_A[-1])):
                binindex_index.append([index, i])
            variables.append(string)
            variable_range.append([var_min, var_max])
            self.lin.coef_list = [1]
            self.lin.index_list = [index]

def todict(self):
    return quad_todict(self, binindex_index)

def toBinaryQuadraticModel(self):
    return dimod.BinaryQuadraticModel.from_qubo(self.todense())

Quad.todict = todict
Quad.toBinaryQuadraticModel = toBinaryQuadraticModel


def A(n):
    A = []
    while True:
        A.append(2**(int(np.log2(n + 1)) - 1))
        n -= A[-1]
        if n == 0:
            break
    return A

def append_lin(lin, coef, index):
    where = where_list(lin.index_list, index)

    if where[0]:
        lin.coef_list[where[1]] += coef
    else:
        lin.index_list.insert(where[1], index)
        lin.coef_list.insert(where[1], coef)

def append_quad(quad, coef, index):
    where_row = where_list(quad.index_list, index[0])

    if where_row[0]:
        where_col = where_list(quad.index_list_list[where_row[1]], index[1])

        if where_col[0]:
            quad.coef_list_list[where_row[1], where_col[1]] += coef

        else:
            quad.index_list_list[where_col[1]].insert(where_col[1], index[1])
            quad.coef_list_list[where_col[1]].insert(where_col[1], index[1])
    
    else:
        quad.index_list.insert(where_row[1], index[0])
        quad.index_list_list.insert(where_row[1], [index[1]])
        quad.coef_list_list.insert(where_row[1], [coef])

    return

def add_dict(dict, var, num):
    if var in dict:
        dict[var] += num
    else:
        dict[var] = num
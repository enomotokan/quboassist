from copy import copy
import numpy as np

class Problem:

    def __init__(self):
        self.obj = Formula()
        self.cond = []
        self.first_cond_num = cond_num
        self.cond_aux_M = {}
        self.weight = []
        self.qubo = {}
        self.var_coef = {}
        self.variables = set()

    def add_objective(self, f):
        if type(f) == Variable or type(f) == Formula:
            if f.comp != "":
                print("Error: The input must be a function, not an equation or an inequation.")
            else:
                self.obj += f
        else:
            print("Error: The type of input must be Variable or Formula.")
            return
    
    def add_constraint(self, w, f):
        if type(f) == Variable or type(f) == Formula:
            if f.comp != ">=" and f.comp != "==":
                print("Error: The input f must be an equation or an inequation, not a function.")
                return
            else:
                try:
                    w = float(w)

                    if f.comp == ">=":
                        M = 0
                        m = 0

                        for variable in f.coef_lin:
                            if f.coef_lin[variable] > 0:
                                M += f.coef_lin[variable] * variable_range[variable][1]
                                m += f.coef_lin[variable] * variable_range[variable][0]
                            else:
                                M += f.coef_lin[variable] * variable_range[variable][0]
                                m += f.coef_lin[variable] * variable_range[variable][1]
                        
                        if M < - f.const:
                            print("Error: This condition cannot be satisfied.")
                            return
                        elif m >= - f.const:
                            print("This condition is always satisfied.")
                            return
                        elif M == - f.const:
                            f.comp = "=="
                            self.cond.append(f)
                            self.weight.append(w)
                        else:
                            self.cond_aux_M[len(self.cond)] = M + f.const
                            self.cond.append(f)
                            self.weight.append(w)
                    else:
                        self.cond.append(f)
                        self.weight.append(w)


                except:
                    print("Error: The type of input w must be a number.")
                    return
        else:
            print("Error: The type of input f must be Variable or Formula.")
            return
    
    def compile(self):
        self.variables |= set(self.obj.coef_quad.keys())
        
        for key_col in self.obj.coef_quad:
            self.variables.add(key_col)
            self.variables |= set(self.obj.coef_quad[key_col].keys())
        
        self.variables |=self.obj.coef_lin.keys()

        for i in range(len(self.cond)):
            self.variables |= set(self.cond[i].coef_lin.keys())

        for variable in self.variables:
            variable_coef[variable] = A(variable_range[variable][1] - variable_range[variable][0])

        # expand quadratic monomials

        for key_col in self.obj.coef_quad:
            for key_row in self.obj.coef_quad[key_col]:
                if key_col == key_row:
                    add_dict(self.obj.coef_lin, key_col, 2 * self.obj.coef_quad[key_col][key_row] * variable_range[key_col][0])
                else:
                    add_dict(self.obj.coef_lin, key_col, self.obj.coef_quad[key_col][key_row] * variable_range[key_row][0])
                    add_dict(self.obj.coef_lin, key_row, self.obj.coef_quad[key_col][key_row] * variable_range[key_col][0])
        
        # expand the two power of the left hand side of conditions
        
        for i in range(len(self.cond)):
            const = self.cond[i].const
            for key in self.cond[i].coef_lin:
                const += self.cond[i].coef_lin[key] * variable_range[key][0]

            if self.cond[i].comp == "==":
                key_list = list(self.cond[i].coef_lin.keys())
                for key_index in range(len(key_list)):
                    add_dict(self.obj.coef_lin, key_list[key_index], self.weight[i] * 2 * const * self.cond[i].coef_lin[key_list[key_index]])
                    add_LIL(self.obj.coef_quad, key_list[key_index], key_list[key_index], self.cond[i].coef_lin[key_list[key_index]]**2)
                    
                    for key_index_ in range(key_index + 1, len(key_list)):
                        if variables[key_list[key_index]] >= variables[key_list[key_index_]]:
                            key_index, key_index_ = key_index_, key_index
                        add_LIL(self.obj.coef_quad, key_list[key_index], key_list[key_index_], self.weight[i] * 2 * self.cond[i].coef_lin[key_list[key_index]] * self.cond[i].coef_lin[key_list[key_index_]])                      

            else:
                variable_coef["aux{}%".format(self.first_cond_num + i)] = A(self.cond_aux_M[i])
                add_dict(self.obj.coef_lin, "aux{}%".format(self.first_cond_num + i), - 2 * self.weight[i] * const * 2)
                add_LIL(self.obj.coef_quad, "aux{}%".format(self.first_cond_num + i), "aux{}%".format(self.first_cond_num + i), self.weight[i])

                key_list = list(self.cond[i].coef_lin.keys())
                
                for key_index in range(len(key_list)):
                    add_dict(self.obj.coef_lin, key_list[key_index], self.weight[i] * 2 * const * self.cond[i].coef_lin[key_list[key_index]])
                    add_LIL(self.obj.coef_quad, key_list[key_index], key_list[key_index], self.weight[i] * self.cond[i].coef_lin[key_list[key_index]]**2)
                    
                    for key_index_ in range(key_index + 1, len(key_list)):
                        if variables[key_list[key_index]] >= variables[key_list[key_index_]]:
                            key_index, key_index_ = key_index_, key_index
                        add_LIL(self.obj.coef_quad, key_list[key_index], key_list[key_index_], self.weight[i] * 2 * self.cond[i].coef_lin[key_list[key_index]] * self.cond[i].coef_lin[key_list[key_index_]])                      

                    add_LIL(self.obj.coef_quad, key_list[key_index], "aux{}%".format(self.first_cond_num + i), - self.weight[i] * 2 * self.cond[i].coef_lin[key_list[key_index]])
        obj_bin_coef_LIL = {}

        for key_col in self.obj.coef_quad:
            for key_row in self.obj.coef_quad[key_col]:
                if key_col == key_row:
                    for i in range(len(variable_coef[key_col])):
                        add_LIL(obj_bin_coef_LIL, key_col + "%" + str(i), key_row + "%" + str(i), self.obj.coef_quad[key_col][key_row] * variable_coef[key_col][i] * variable_coef[key_col][i])
                        for j in range(i + 1, len(variable_coef[key_row])):
                            add_LIL(obj_bin_coef_LIL, key_col + "%" + str(i), key_row + "%" + str(j), self.obj.coef_quad[key_col][key_row] * 2 * variable_coef[key_col][i] * variable_coef[key_col][j])
                else:
                    for i in range(len(variable_coef[key_col])):
                        for j in range(len(variable_coef[key_row])):
                            add_LIL(obj_bin_coef_LIL, key_col + "%" + str(i), key_row + "%" + str(j), self.obj.coef_quad[key_col][key_row] * variable_coef[key_col][i] * variable_coef[key_row][j])

        for key in self.obj.coef_lin:
            for i in range(len(variable_coef[key])):
                add_LIL(obj_bin_coef_LIL, key + "%" + str(i), key + "%" + str(i), self.obj.coef_lin[key] * variable_coef[key][i])
        
        for key_col in obj_bin_coef_LIL:
            for key_row in obj_bin_coef_LIL[key_col]:
                self.qubo[(key_col, key_row)] = obj_bin_coef_LIL[key_col][key_row]

    def solution(self, result):
        solution = {}
        for key in self.variables:
            if key[:3] != "aux":
                for i in range(len(variable_coef[key])):
                    add_dict(solution, key, variable_coef[key][i] * int(result[key + "%" + str(i)]) + variable_range[key][0])

        # ckeck whether the solution satisfies constraint conditions

        bool_solution = []

        for i in range(len(self.cond)):
            if self.cond[i].comp == "==":
                S = self.cond[i].const
                for key in self.cond[i].coef_lin:
                    S += self.cond[i].coef_lin[key] * solution[key]
                bool_solution.append(S == 0)
            else:
                S = self.cond[i].const
                for key in self.cond[i].coef_lin:
                    S += self.cond[i].coef_lin[key] * solution[key]
                bool_solution.append(S >= 0)
                
        return solution, bool_solution

variables = {}
variable_range = {}
variable_coef = {}
cond_num = 0

class Formula:
    def __init__(self):
        self.coef_lin = {}
        self.coef_quad = {}
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
        
        F = Formula()

        try:
            num = int(g)
            if float(g) != num:
                print("Error: Coefficients of Constraints must be integer.")
                return

            F.order = f.order
            F.coef_lin = copy(f.coef_lin)
            F.coef_quad = copy(f.coef_quad)
            F.const = copy(f.const)

            if sign[0] == -1:
                for key in F.coef_lin:
                    F.coef_lin[key] *= -1
                for key in F.coef_quad:
                    F.coef_quad[key] *= -1
            F.const += sign[1] * num
            return F

        except:
            pass

        if type(g) == Variable or type(g) == Formula:
            F.order = max(f.order, g.order)
            F.const = sign[0] * f.const + sign[1] * g.const

            F.coef_lin = copy(f.coef_lin)
            F.coef_quad = copy(f.coef_quad)

            if sign[0] == -1:
                for key in F.coef_lin.keys():
                    F.coef_lin[key] *= -1
                for key_col in F.coef_quad.keys():
                    for key_row in F.coef_quad[key_col].keys():
                        F.coef_quad[key_col][key_row] *= -1

            for key in g.coef_lin.keys():
                add_dict(F.coef_lin, key, sign[1] * g.coef_lin[key])
            
            for key_col in g.coef_quad.keys():
                for key_row in g.coef_quad[key_col].keys():
                    add_LIL(F.coef_quad, key_col, key_row, sign[1] * g.coef_quad[key_col][key_row])

            return F

        else:
            print("Error: Attempting to add by a value other than a formula or a number.")
            pass

    def __mul__(self, other):
        return self._mul(self, other)

    def __rmul__(self, other):
        return self._mul(self, other)
    
    def __pow__(self, n):
        try:
            n = int (n)
        except:
            print("Error: The exponent of power must be a non-negative integer")
            pass

        F = 1
        for i in range(n):
            F = self * F
        
        return F

    def __truediv__(self, other):
        try:
            num = int(other)
            return self._mul(self, 1 / num)
        except:
            print("Error: Attempting to devide by a value other than a number.")
            pass
    
    def _mul(self, f, g):

        F = Formula()
        global variables

        try:
            num = int(g)
            if float(g) != num:
                print("Error: Coefficients of Constraints must be integer.")

            if num != 0:
                F.order = f.order
                for key in f.coef_lin.keys():
                    F.coef_lin[key] = f.coef_lin[key] * num
                return F
            else:
                return F
        except:
            pass
            
        if type(f) == Variable or type(f) == Formula:
            F.order = f.order + g.order
            F.const = f.const * g.const

            if F.order >= 3:
                print("Error: The QUBO form must have only terms of order two or lower.")
                return

            # coeffficients of linear terms

            if g.const != 0:
                for key in f.coef_lin.keys():
                    F.coef_lin[key] = g.const * f.coef_lin[key]
            
            if f.const != 0:
                for key in g.coef_lin.keys():
                    add_dict(F.coef_lin, key, f.const * g.coef_lin[key])

            # coefficients of quadratic terms

            if g.const != 0:
                for key_col in f.coef_quad.keys():
                    for key_row in f.coef_quad[key_col].keys():
                        add_LIL(F.coef_quad, key_col, key_row, g.const * f.coef_quad[key_col][key_row])

            if f.const != 0:
                for key_col in g.coef_quad.keys():
                    for key_row in g.coef_quad[key_col].keys():
                        add_LIL(F.coef_quad, key_col, key_row, f.const * g.coef_quad[key_col][key_row])

            for key1 in f.coef_lin.keys():
                for key2 in g.coef_lin.keys():    
                    if variables[key1] <= variables[key2]:
                        add_LIL(F.coef_quad, key1, key2, f.coef_lin[key1] * g.coef_lin[key2])
                    else:
                        add_LIL(F.coef_quad, key2, key1, f.coef_lin[key1] * g.coef_lin[key2])
            return F


        else:
            print("Error: Attempting to multiply by a value other than a number.")
            return

    def __pos__(self):
        return self
    
    def __neg__(self):
        F = Formula()
        F.order = self.order

        for key in self.coef_lin.keys():
            F.coef_lin[key] = - self.coef_lin[key]
        
        for key_col in self.coef_quad.keys():
            for key_row in self.coef_quad[key_col].keys():
                add_LIL(F.coef_quad, key_col, key_row, - self.coef_quad[key_col][key_row])

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
            print("Error: The enequation must be linear.")
            return
        else:
            F.comp = ">="
            return F
    
    def __eq__(self, f):
        # ==

        F = self - f
        if F.order >= 2:
            print("Error: The equation must be linear.")
            return
        else:
            F.comp = "=="
            return F

class Variable(Formula):
    def __init__(self, string, var_min, var_max):
        self.coef_lin = {}
        self.coef_quad = {}
        self.const = 0.0
        self.order = 1
        self.comp = ""

        global variables

        try:
            self.var_min = int(np.floor(var_min))
            self.var_max = int(np.ceil(var_max))

            if self.var_max <= self.var_min:
                print("Error: The range of variables must not be empty.")
                return
            else:
                variable_range[string] = [var_min, var_max]
        except:
            print("Error: The minimum or maximum value of variables must be integer.")
            return
            


        if type(string) != str:
            print("Error: The type of variable name {} is not str.".format(string))
            return
        elif string in variables.keys():
            print("Error: There is already a variable with the same name \'{}\'.".format(string))
            return
        elif string[:3] == "aux":
            print("Error: The first three characters variable name cannot be \'aux\'.")
        else:
            self.name = string
            variables[string] = len(variables)
            self.coef_lin[string] = 1
            variable_range[string] =[var_min, var_max]
    
    def change_min(self, var_min):
        try:
            if self.var_max <= var_min:
                print("Error: The range of variables must not be empty.")
                return
            self.var_min = int(np.floor(var_min))
            variable_range[self.name][0] = self.var_min
        except:
            print("Error: The minimum or maximum value of a variable must be integer.")
            return
    
    def change_max(self, var_max):
        try:
            if self.var_min >= var_max:
                print("Error: The range of variables must not be empty.")
                return
            self.var_max = int(np.floor(var_max))
            variable_range[self.name][0] = self.var_max
        except:
            print("Error: The minimum or maximum value of a variable must be integer.")
            return

def A(n):
    A = []
    while True:
        A.append(2**(int(np.log2(n + 1)) - 1))
        n -= A[-1]
        if n == 0:
            break
    return A

def add_LIL(LIL, col, row, num):
    if col in LIL:
        if row in LIL[col]:
            LIL[col][row] += num
        else:
            LIL[col][row] = num

    else:
        LIL[col] = {row: num}

    return

def add_dict(dict_coef, var, num):
    if var in dict_coef:
        dict_coef[var] += num
    else:
        dict_coef[var] = num

    return
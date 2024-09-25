from copy import copy
import numpy as np

class Problem:

    def __init__(self):
        self.obj = Formula()
        self.cond = []
        self.cond_aux_coef = {}
        self.weight = []
        self.qubo = {}
        self.var_coef = {}

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
            if f.comp != (">=" or "=="):
                print("Error: The input f must be a function, not an equation or an inequation.")
                return
            else:
                try:
                    w = float(w)

                    if f.comp == ">=":
                        M = f.const
                        m = f.const

                        for variable in f.coef_lin:
                            if f.coef_lin[variable] > 0:
                                M += f.coef_lin[variable] * variable_range[variable][1]
                                m += f.coef_lin[variable] * variable_range[variable][0]
                            else:
                                M += f.coef_lin[variable] * variable_range[variable][0]
                                m += f.coef_lin[variable] * variable_range[variable][1]
                        
                        if M <= -1:
                            print("Error: This condition cannot be satisfied.")
                            return
                        elif m >= 0:
                            print("This condition is always satisfied.")
                            return
                        elif M == 0:
                            f.comp = "=="
                            self.cond.append(f)
                            self.weight.append(w)
                        else:
                            self.cond.append(f)
                            self.weight.append(w)
                            self.cond_aux_coef[len(self.cond)] = A(M)
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
        for variable in variables.keys():
            variable_coef[variable] = [variable_range[variable][1], A(variable_range[variable][1] - variable_range[variable][0])]
        
        

        


        


variables = {}
variable_range = {}
variable_coef = {}

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
                if key in F.coef_lin:
                    F.coef_lin[key] += sign[1] * g.coef_lin[key]
                else:
                    F.coef_lin[key] = sign[1] * g.coef_lin[key]
            
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
                    if key in F.coef_lin:
                        F.coef_lin[key] += f.const * g.coef_lin[key]
                    else:
                        F.coef_lin[key] = f.const * g.coef_lin[key]

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

        for key in self.coefficients.keys():
            F.coefficients[key] = - self.coefficients[key]
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
    
    def __eq__(self, F):
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
        elif string[-1] == "%":
            print("Error: The last character of a variable name must not be %.")
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

    return LIL
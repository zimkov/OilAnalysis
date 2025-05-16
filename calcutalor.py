import polynoms as pn
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

class Calculator:
    def __init__(self, startValues, minValues, maxValues, polynomCoefs, disturbancesCoefs, normValues):
        self.values = np.array(startValues)
        self.minValues = np.array(minValues)
        self.maxValues = np.array(maxValues)
        self.functions = self.initFunctions(polynomCoefs)
        self.qfunctions = self.initFunctions(disturbancesCoefs)
        self.normValues = normValues
        self.optimized_coefs = None  # Для хранения оптимизированных коэффициентов

    def initFunctions(self, coefs):
        functions = []
        for coef_set in coefs:
            num_coefs = len(coef_set)

            if num_coefs == 2:
                polynomial = pn.LinearPolynomial(*coef_set)
            elif num_coefs == 3:
                polynomial = pn.QuadraticPolynomial(*coef_set)
            elif num_coefs == 4:
                polynomial = pn.CubicPolynomial(*coef_set)
            else:
                raise ValueError(f"Неподдерживаемое количество коэффициентов: {num_coefs}")

            functions.append(polynomial)

        return functions
    
    def apply_constraints(self, solution):
        """Применение ограничений для каждого параметра"""
        constrained = np.zeros_like(solution)
        for i in range(solution.shape[1]):
            constrained[:, i] = np.clip(
                solution[:, i],
                self.minValues[i],
                self.maxValues[i]
            )
        return constrained

    def normalize(self, value):
        """Нормализация для каждого параметра индивидуально"""
        return (value - self.minValues) / (self.maxValues - self.minValues)

    def calculate(self, timeIntervals):
        """Основной расчет с ограничениями"""
        # Решение системы дифференциальных уравнений
        solution = odeint(self.calcFunctions, 
                          self.values, 
                          timeIntervals)

        # Применение ограничений и нормализации
        constrained_solution = self.apply_constraints(solution)

        return constrained_solution
    
    def adjust_coefficients(self, target):
        """Оптимизация коэффициентов"""
        initial_guess = np.concatenate([
            func.coefficients 
            for func in self.functions + self.qfunctions
        ])
        
        def loss(coefs):
            ptr = 0
            for func in self.functions + self.qfunctions:
                n = len(func.coefficients)
                func.coefficients = coefs[ptr:ptr+n]
                ptr += n

            prediction = self.calculate(np.linspace(0, 1, 11))
            
            # Штрафы за нарушения границ
            lower_penalty = np.sum(
                np.where(prediction < self.minValues, 
                        1000 * (self.minValues - prediction), 0)
            )
            
            upper_penalty = np.sum(
                np.where(prediction > self.maxValues, 
                        1000 * (prediction - self.maxValues), 0)
            )
            
            # Штраф за отклонение от целевых значений
            target_penalty = np.mean((prediction - target)**2)
            
            return target_penalty + lower_penalty + upper_penalty

        # Ограничения для коэффициентов (пример для положительных коэффициентов)
        bounds = [(0.01, None) for _ in initial_guess]
        
        result = minimize(loss, initial_guess, method='L-BFGS-B', bounds=bounds)
        self.optimized_coefs = result.x
        return result

    def calcFunctions(self, u, t):
        """Модифицированная система уравнений с ограничениями"""
        [L1_t, L2_t, L3_t, L4_t, L5_t, L6_t, L7_t, 
         L8_t, L9_t, L10_t, L11_t, L12_t, L13_t, L14_t, L15_t] = u
        
        # Динамическое ограничение производных
        max_derivatives = (self.maxValues - u) * 0.1  # Максимальный рост 10% от текущего значения

        # Расчет возмущений
        q1 = np.clip(self.qfunctions[0].calc(t), 0, self.maxValues[0])
        q2 = np.clip(self.qfunctions[1].calc(t), 0, self.maxValues[1])
        q3 = np.clip(self.qfunctions[2].calc(t), 0, self.maxValues[2])
        q4 = np.clip(self.qfunctions[3].calc(t), 0, self.maxValues[3])

        # Расчет производных с ограничениями
        dL1_dx = -(self.functions[0].calc(L10_t) * self.functions[1].calc(L11_t) * self.functions[2].calc(L14_t))
        dL1_dx = np.clip(dL1_dx, 0, max_derivatives[0])

        dL2_dx = self.functions[3].calc(L3_t) * self.functions[4].calc(L7_t) * self.functions[5].calc(L8_t)* \
                 self.functions[6].calc(L9_t) * self.functions[7].calc(L13_t) - (self.functions[8].calc(L10_t) *
                 self.functions[9].calc(L11_t) * self.functions[10].calc(L14_t) * self.functions[11].calc(L15_t) * q1 + q2 + q3 + q4)
        dL2_dx = np.clip(dL2_dx, 0, max_derivatives[1])

        dL3_dx = self.functions[12].calc(L1_t) - (self.functions[13].calc(L15_t) * q1 + q3 + q4)

        dL4_dx = self.functions[14].calc(L1_t)

        dL5_dx = self.functions[15].calc(L1_t) * q2 - q1

        dL6_dx = q2 - (self.functions[16].calc(L4_t) * self.functions[17].calc(L11_t) * self.functions[18].calc(L12_t) *
                       self.functions[19].calc(L14_t) * q1)

        dL7_dx = self.functions[20].calc(L5_t) * self.functions[21].calc(L6_t) * self.functions[22].calc(L13_t) * \
                 self.functions[23].calc(L15_t) * q1 + q2 + q3

        dL8_dx = self.functions[24].calc(L5_t) * self.functions[25].calc(L6_t) * self.functions[26].calc(L11_t) * \
                 self.functions[27].calc(L13_t) * self.functions[28].calc(L14_t) * self.functions[29].calc(
            L15_t) * q1 + q2 + q3

        dL9_dx = self.functions[30].calc(L3_t) * self.functions[31].calc(L13_t) * q2 - (
                    self.functions[32].calc(L10_t) * self.functions[33].calc(L11_t) * self.functions[34].calc(
                L14_t) * q1)

        dL10_dx = self.functions[35].calc(L3_t) * self.functions[36].calc(L9_t) * self.functions[37].calc(
            L15_t) * q1 + q2 + q3 + q4

        dL11_dx = self.functions[38].calc(L3_t) * self.functions[39].calc(L13_t) * self.functions[40].calc(
            L14_t) * q1 + q3 - (self.functions[41].calc(L15_t) * q4)

        dL12_dx = self.functions[42].calc(L11_t) * self.functions[43].calc(L13_t) * self.functions[44].calc(
            L14_t) * q1 + q2 + q3 - (self.functions[45].calc(L15_t))

        dL13_dx = self.functions[46].calc(L2_t) * self.functions[47].calc(L3_t) * q2

        dL14_dx = self.functions[48].calc(L11_t) * self.functions[49].calc(L12_t) * self.functions[50].calc(
            L13_t) * q1 + q2

        dL15_dx = self.functions[51].calc(L2_t) * self.functions[52].calc(L3_t) * self.functions[53].calc(L13_t) * \
                  self.functions[54].calc(L14_t) * q1 + q2
        
        dL3_dx = np.clip(dL3_dx, 0, max_derivatives[2])
        dL4_dx = np.clip(dL4_dx, 0, max_derivatives[3])
        dL5_dx = np.clip(dL5_dx, 0, max_derivatives[4])
        dL6_dx = np.clip(dL6_dx, 0, max_derivatives[5])
        dL7_dx = np.clip(dL7_dx, 0, max_derivatives[6])
        dL8_dx = np.clip(dL8_dx, 0, max_derivatives[7])
        dL9_dx = np.clip(dL9_dx, 0, max_derivatives[8])
        dL10_dx = np.clip(dL10_dx, 0, max_derivatives[9])
        dL11_dx = np.clip(dL11_dx, 0, max_derivatives[10])
        dL12_dx = np.clip(dL12_dx, 0, max_derivatives[11])
        dL13_dx = np.clip(dL13_dx, 0, max_derivatives[12])
        dL14_dx = np.clip(dL14_dx, 0, max_derivatives[13])
        dL15_dx = np.clip(dL15_dx, 0, max_derivatives[14])

        return [dL1_dx, dL2_dx, dL3_dx, dL4_dx, dL5_dx, 
                dL6_dx, dL7_dx, dL8_dx, dL9_dx, dL10_dx, 
                dL11_dx, dL12_dx, dL13_dx, dL14_dx, dL15_dx]


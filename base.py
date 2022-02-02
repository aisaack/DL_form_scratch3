import numpy as np
class Variable:
    '''This class just holds input value, gradient'''
    def __init__(self, data:np.ndarray):
        self.data = data
        self.grad = None
        self.function = None

    def get_function(self, func):
        self.function = func

    def backward(self):
        func = self.function
        if func is not None:
            func_input = func.input
            func_input.grad = func.backward(self.grad)
            func_input.backward()


class Function:
    def __call__(self, inputs:Variable) -> Variable:
        x = inputs.data
        y = self.forward(x)
        out = Variable(y)
        out.get_function(self)
        self.input = inputs
        self.output = out
        return out
    
    def forward(self, x):
        '''Implement it locally.'''
        raise NotImplementedError()

    def backward(self, din):
        '''Chain rule
        Implement it locally.
        
        Calculating local gradient and multiplied by upstream gradient.

        Args:
            din     (np.ndarray): upstream gradient
        
        Return:
            upstream gradient (np.ndarray)
        '''
        raise NotImplementedError()


class Square(Function):
    def forward(self, x:Variable) -> Variable:
        return x ** 2

    def backward(self, din:Variable):
        '''Derivative function'''
        x = self.input.data
        return 2 * din * x


class Exp(Function):
    def forward(self, x:Variable) -> Variable:
        return np.exp(x)

    def backward(self, din:Variable):
        '''Derivative function'''
        x = self.input.data
        return np.exp(x) * din



if __name__ == '__main__':
    x = Variable(np.array(0.5))
    f1 = Square()
    f2 = Exp()
    f3 = Square()

    y1 = f1(x)
    y2 = f2(y1)
    y3 = f3(y2)

    y3.grad = np.array(1.0)
    y3.backward()
    print(x.grad)
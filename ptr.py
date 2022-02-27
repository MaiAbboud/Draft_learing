# a = [10,20,30]
# print(*a)

# def fun(*input1,**input2):
#     return sum(input)

# print(fun('50','10'))

# a = 10
# if a == 10:
    # raise ValueError('bal')


#
# isinsatnce to check type
# x = isinstance("Hello", tuple) 
# print(x)
import random
def Choice(*args):
    """Can be used to sample random parameter values."""
    return random.choice(args)

print(Choice(10,20,21,4,40))
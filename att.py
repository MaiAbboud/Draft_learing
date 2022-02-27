# class family:
#     parents = ['ali','sakina']
#     children = ['issa' , 'mary', 'mai']
#     age = [10,20,30]

#     def add_10_years(self):
#         return self.age+10
    
#     def show_family_info(self):
#         print(self.parents)




# abboud = family()
# abboud.show_family_info

# hasatt
# print('hasatt {}'.format(hasattr(abboud,'age')))
# print('hasatt {}'.format(hasattr(abboud,'age1')))
# print('getatt {}'.format(getattr(abboud,'age')))
# print('getatt {}'.format(getattr(abboud,'age2',20)))
# print('getatt {}'.format(getattr(abboud,'age2')))
# print('setatt {}'.format(getattr(abboud,'age',20)))
# print(abboud.age2)
# print('getatt {}'.format(getattr(abboud,'age2')))

class TrackerParams:
    """Class for tracker parameters."""
    # def set_default_values(self, default_vals: dict):
    #     for name, val in default_vals.items():
    #         if not hasattr(self, name):
    #             setattr(self, name, val)

    # def get(self, name: str, *default):
    #     """Get a parameter value with the given name. If it does not exists, it return the default value given as a
    #     second argument or returns an error if no default value is given."""
    #     if len(default) > 1:
    #         raise ValueError('Can only give one default value.')

    #     if not default:
    #         return getattr(self, name)

    #     return getattr(self, name, default[0])

    # def has(self, name: str):
    #     """Check if there exist a parameter with the given name."""
    #     return hasattr(self, name)


def parameters():
    params = TrackerParams()
    params.debug = 0
    params.visualization = False
    print(params.__dict__)

parameters()

class Coordinates:
	def __init__(self, x = 0, y = 0):
		self.x = x
		self.y = y

c = Coordinates(1, 2)

# prints {'x': 1, 'y': 2}
print(c.__dict__)
c.description = "My coordinates"
print(c.__dict__)
import torch 
a = torch.tensor([10,20,30])
print(a)
a = a.cuda
print(a)
# print(c.cuda)
class Config():

    static_members = {}

    def __init__(self, description=''):
        self.description = description

    def __setattr__(self, name, value):
        #inspired by http://code.activestate.com/recipes/65207-constants-in-python/
        if name in self.__dict__:
            raise Exception(f'Value of {name} is already set - change original value')
        self.__dict__[name] = value

    def set_static(self, name, func, args):
        if name in Config.static_members:
            return

        Config.static_members[name] = func(args)
        self.__dict__[name] = Config.static_members[name]

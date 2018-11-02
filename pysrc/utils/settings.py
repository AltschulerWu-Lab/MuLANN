class Settings(object):
    """
    Simple container to hold all settings from an external python file as own
    class attributes.
    Should be made singleton.
    Code from Dr Thomas Walter as available on https://github.com/ThomasWalter
    """

    def __init__(self, filename=None, dctGlobals=None):
        self.strFilename = filename
        if not filename is None:
            self.load(filename, dctGlobals)


    def load(self, filename, dctGlobals=None):
        self.strFilename = filename
        if dctGlobals is None:
            dctGlobals = globals()

        #Changing execfile to exec(open(fn).read() when going from 2.7 to 3.5
        exec(open(self.strFilename).read(), dctGlobals, self.__dict__)

    def update(self, dctNew, bExcludeNone=False):
        for strKey, oValue in dctNew.iteritems():
            if not bExcludeNone or not oValue is None:
                self.__dict__[strKey] = oValue

    def __getattr__(self, strName):
        if strName in self.__dict__:
            return self.__dict__[strName]
        else:
            raise SettingsError("Parameter '%s' not found in settings file '%s'." %
                                (strName, self.strFilename))

    def __call__(self, strName):
        return getattr(self, strName)

    def all(self, copy=True):
        return self.__dict__.copy()


class SettingsError(Exception):
    pass
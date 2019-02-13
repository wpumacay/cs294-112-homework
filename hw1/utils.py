

# hardcoded fix to load appropiate GLEW
def loadDynamicDeps():
    import ctypes

    _libGlewLibraryPath = ctypes.util.find_library( 'GLEW' )
    ctypes.CDLL( _libGlewLibraryPath, ctypes.RTLD_GLOBAL )





def loadDynamicDeps():
    """Hacky-fix to load the appropiate GLEW library

    Based on this fixes from controlsuite repo:
    https://github.com/deepmind/dm_control/blob/978230f1376de1826c430dd3dfc0e3c7f742f5fe/dm_control/mujoco/wrapper/util.py#L100
    """
    import ctypes

    _libGlewLibraryPath = ctypes.util.find_library( 'GLEW' )
    ctypes.CDLL( _libGlewLibraryPath, ctypes.RTLD_GLOBAL )


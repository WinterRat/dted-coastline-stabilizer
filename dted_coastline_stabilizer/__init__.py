def classFactory(iface):
    from .plugin import CoastPlugin

    return CoastPlugin(iface)

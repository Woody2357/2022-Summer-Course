"""
@author: Yong Zheng Ong

some utility functions for loading modules
"""
import os

def loadmodule(package, name, prefix='..'):
    """
    function to load a module given a string input
    """
    strCmd = "from " + prefix + package + " import " + name + " as module"
    exec(strCmd)
    return eval('module')

def loaddataset(name):
    datasetlist = os.listdir("datasets")
    if name not in datasetlist:
        raise ValueError("invalid dataset name given {}! input dataset list from list {}".format(name, datasetlist))
    return loadmodule("datasets.{}.dataloader".format(name), "DataEncapsulator", prefix='')

def loadtrainer(name):
    return loadmodule("libs.models.{}.trainer".format(name), "Trainer", prefix='')
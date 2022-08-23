import numpy as np
import function as func
unary = func.unary_functions
binary = func.binary_functions
unary_functions_str = func.unary_functions_str
binary_functions_str = func.binary_functions_str

class BinaryTree(object):
    def __init__(self,item,is_unary=True):
        self.key=item
        self.is_unary=is_unary
        self.leftChild=None
        self.rightChild=None
    def insertLeft(self,item, is_unary=True):
        if self.leftChild==None:
            self.leftChild=BinaryTree(item, is_unary)
        else:
            t=BinaryTree(item)
            t.leftChild=self.leftChild
            self.leftChild=t
    def insertRight(self,item, is_unary=True):
        if self.rightChild==None:
            self.rightChild=BinaryTree(item, is_unary)
        else:
            t=BinaryTree(item)
            t.rightChild=self.rightChild
            self.rightChild=t

def compute_by_tree(tree, x):
    ''' judge whether a emtpy tree, if yes, that means the leaves and call the unary operation '''
    if tree.leftChild == None and tree.rightChild == None:
        return tree.key(x)
    elif tree.leftChild == None and tree.rightChild is not None:
        return tree.key(compute_by_tree(tree.rightChild, x))
    elif tree.leftChild is not None and tree.rightChild == None:
        return tree.key(compute_by_tree(tree.leftChild, x))
    else:
        return tree.key(compute_by_tree(tree.leftChild, x), compute_by_tree(tree.rightChild, x))

def inorder(tree, actions):
    global count
    if tree:
        inorder(tree.leftChild, actions)
        action = actions[count].item()
        if tree.is_unary:
            action = action
            tree.key = unary[action]
            # print(count, action, func.unary_functions_str[action])
        else:
            action = action
            tree.key = binary[action]
            # print(count, action, func.binary_functions_str[action])
        count = count + 1
        inorder(tree.rightChild, actions)

# def inorder(tree):
#     if tree:
#         inorder(tree.leftChild)
#         print(tree.key)
#         inorder(tree.rightChild)

count = 0
def inorder_w_idx(tree):
    global count
    if tree:
        inorder_w_idx(tree.leftChild)
        print(tree.key, count)
        count = count + 1
        inorder_w_idx(tree.rightChild)

def basic_tree():
    tree = BinaryTree('', False)
    tree.insertLeft('', False)
    tree.leftChild.insertLeft('', True)
    tree.leftChild.insertRight('', True)
    tree.insertRight('', False)
    tree.rightChild.insertLeft('', True)
    tree.rightChild.insertRight('', True)
    return tree

def get_function(actions):
    global count
    count = 0
    computation_tree = basic_tree()
    inorder(computation_tree, actions)
    count = 0 # 置零
    return computation_tree

def inorder_test(tree, actions):
    global count
    if tree:
        inorder(tree.leftChild, actions)
        action = actions[count].item()
        print(action)
        if tree.is_unary:
            action = action
            tree.key = unary[action]
            # print(count, action, func.unary_functions_str[action])
        else:
            action = action
            tree.key = binary[action]
            # print(count, action, func.binary_functions_str[action])
        count = count + 1
        inorder(tree.rightChild, actions)

if __name__ =='__main__':
    # tree = BinaryTree(np.add)
    # tree.insertLeft(np.multiply)
    # tree.leftChild.insertLeft(np.cos)
    # tree.leftChild.insertRight(np.sin)
    # tree.insertRight(np.sin)
    # print(compute_by_tree(tree, 30)) # np.sin(30)*np.cos(30)+np.sin(30)
    # inorder(tree)
    # inorder_w_idx(tree)
    import torch
    bs_action = [torch.LongTensor([10]), torch.LongTensor([2]),torch.LongTensor([9]),torch.LongTensor([1]),torch.LongTensor([0]),torch.LongTensor([2]),torch.LongTensor([6])]

    function = lambda x: compute_by_tree(get_function(bs_action), x)
    x = torch.FloatTensor([[-1], [1]])

    count = 0
    tr = basic_tree()
    inorder_test(tr, bs_action)
    count = 0


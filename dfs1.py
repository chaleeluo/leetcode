# 问题一： 112. 路径总和 ###########################################################################
'''判断
给定如下二叉树，以及目标和 sum = 22，

              5
             / \
            4   8
           /   / \
          11  13  4
         /  \      \
        7    2      1
返回 true, 因为存在目标和为 22 的根节点到叶子节点的路径 5->4->11->2。
'''
def hasPathSum(self, root: TreeNode, sum: int) -> bool:
    if not root:
        return False
    if not root.left and not root.right:
        return sum - root.val == 0
    return self.hasPathSum(root.left,sum - root.val) or self.hasPathSum(root.right,sum - root.val)


# 问题二：113. 路径总和 II ###########################################################################
'''找到所有从根节点到叶子节点路径总和等于给定目标和的路径。
给定如下二叉树，以及目标和 sum = 22，
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \    / \
        7    2  5   1
返回:
[
   [5,4,11,2],
   [5,8,4,5]
]
'''
class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        res = []
        self.dfs(root, sum, [], res)
        return res
        
    def dfs(self, root, s, path, res):
        if not root:
            return []
        '''
        path记录路径，到达叶子节点检查路径和
        '''
        if not root.left and not root.right:
            if sum(path) + root.val == s:
                res.append(path + [root.val])
        self.dfs(root.left, s, path + [root.val], res)
        self.dfs(root.right, s, path + [root.val], res)


# 问题三：437. 路径总和 III ###########################################################################






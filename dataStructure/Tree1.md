**二叉树的层次遍历 II**

> 给定一个二叉树，返回其节点值自底向上的层次遍历。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）
>
> ![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/ruAMsa53pVQWN7FLK88i5jEGzHklTudgPFZkC3N4RoX5TCoXpIXitgexBIf7kp8EirO6ZTli47F2y.m3.NMP7xrm1Ljf4ekyKrp.U9dVtSQ!/b&bo=mgREAgAAAAADB*o!&rf=viewer_4)

```java
class Solution {
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        if(root==null) return new ArrayList();

        Queue<TreeNode> queue = new LinkedList<>(); //队列
        List<List<Integer>> result = new LinkedList<>(); //结果
        queue.add(root);//加入根结点
        while(!queue.isEmpty()){
            int size = queue.size();
            List<Integer> l = new ArrayList<>();
            while(size>0){
                TreeNode temp = queue.poll();
                l.add(temp.val);
                if(temp.left!=null) queue.add(temp.left);
                if(temp.right!=null) queue.add(temp.right);
                size--;
            }
            result.add(l);// 每次都往队头塞
        }
        Collections.reverse(result);
        return result;
    }
}
```



**二叉树的锯齿形层次遍历**

> 给定一个二叉树，返回其节点值的锯齿形层次遍历。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。
>
> 给定二叉树 `[3,9,20,null,null,15,7]`,
>
> [[3],[20,9],[15,7]]

```java
class Solution {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> re = new LinkedList<>();
        if(root==null) return re;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int i = 1;//单数层从左往右，双数层从右往左
        while(!queue.isEmpty()){
            int size = queue.size();
            List<Integer> l = new LinkedList<>();
            while(size>0){
                TreeNode temp = queue.poll();
                l.add(temp.val);
                if(temp.left!=null) queue.add(temp.left);
                if(temp.right!=null) queue.add(temp.right);   
                size--;
            }
            if(i%2==0) Collections.reverse(l);//单数层逆转一下
            i++;
            re.add(l);   
        }
        return re;
    }
}
```



**二叉树最小深度**

> 给定一个二叉树，找出其最小深度。
>
> 最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
>
> 有几种异常临界条件想好，当左右有一个null，则最短应该是有的

```java
class Solution {
    public int minDepth(TreeNode root) {
      if (root == null) return 0;
    
    	// null节点不参与比较
    	if (root.left == null && root.right != null) {
        return 1 + minDepth(root.right);
    	}
    	// null节点不参与比较
    	if (root.right == null && root.left != null) {
        return 1 + minDepth(root.left);
    	}
      return 1 + Math.min(minDepth(root.left), minDepth(root.right));
    }   
}
```



**相同的树**

> 给定两个二叉树，编写一个函数来检验它们是否相同。
>
> 如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。

```java
class Solution {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if(p==null&&q==null) return true;
        if(p==null||q==null) return false;
        if(p.val!=q.val) return false;
        return isSameTree(p.left,q.left)&isSameTree(p.right,q.right);
    }
}
```



**对称二叉树**

> 给定一个二叉树，检查它是否是镜像对称的。
>
> 例如，二叉树 `[1,2,2,3,4,4,3]` 是对称的。

```java
//递归，左右子树是否对称-->左子树左孩子是否与右子树右孩子对称，左子树右孩子是否和右子树左孩子对称
class Solution {
    public boolean isSymmetric(TreeNode root) {
        if(root==null) return true;
        return isSymmetric1(root.left,root.right);
    }
		//判断两颗子树是否对称
    public boolean isSymmetric1(TreeNode a,TreeNode b){
        if(a==null&&b==null) return true;
        if(a==null||b==null) return false;
        if(a.val!=b.val) return false;
        return isSymmetric1(a.left,b.right)&isSymmetric1(a.right,b.left);
    }
}
```



**路径总和**

> 给定一个二叉树和一个目标和，判断该树中是否存在根节点到叶子节点的路径，这条路径上所有节点值相加等于目标和。

```java
class Solution {
    public boolean hasPathSum(TreeNode root, int sum) {
        if(root==null) return false; //空结点false
        if(root.left==null&&root.right==null) return sum==root.val; //到叶子结点了
        return hasPathSum(root.left,sum-root.val)||hasPathSum(root.right,sum-root.val);
    }
}
```

**路径总和2**

> 给定一个二叉树和一个目标和，找到所有从根节点到叶子节点路径总和等于给定目标和的路径。

```java
class Solution {
    //写一个全局变量来存放暂时路径
    List<Integer> cur = new LinkedList();
    List<List<Integer>> re = new LinkedList<>();//结果

    public List<List<Integer>> pathSum(TreeNode root, int sum) {  
        path(root,sum);
        return re;
    }

    public void path(TreeNode root,int sum){
        if(root==null) return; //如果到叶子结点下的直接返回
        cur.add(root.val); //加入路径
        if(root.left==null&&root.right==null&&root.val==sum){
            //叶子结点已经满足则加入结果集合
            List<Integer> l = new LinkedList(cur);
            re.add(l);
        }
        //去左边路径看，如果已经是根结点则返回了
        path(root.left,sum-root.val);
        path(root.right,sum-root.val);
        cur.remove(cur.size()-1);//回退一个点继续
    } 
}
```



**根结点到叶子结点数字之和**

> 给定一个二叉树，它的每个结点都存放一个 0-9 的数字，每条从根到叶子节点的路径都代表一个数字。
>
> 例如，从根到叶子节点路径 1->2->3 代表数字 123。计算从根到叶子节点生成的所有数字之和。
>
> ![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/ruAMsa53pVQWN7FLK88i5mXSWQd.tLjqj41oOFJdyW3PkMYYSmgVsDzPp04DmLLfuInJi6B4Tpdfhi6W3M2S.c3RnbdFNJi0bmgD7S6lUmw!/b&bo=oAPwAQAAAAADB3A!&rf=viewer_4)

```java
class Solution {
    int sum=0;
    public int sumNumbers(TreeNode root) {
        sum(0,root);
        return sum;
    }
    public void sum(int val,TreeNode root){
        if(root==null) return;
        if(root.left==null&&root.right==null){
            //到了叶子结点
            sum+=val*10+root.val;
        }
        sum(val*10+root.val,root.left);//左边找
        sum(val*10+root.val,root.right);
        return;
    }
}
```



**翻转二叉树**

> 翻转一颗二叉树

```java
class Solution {
    public TreeNode invertTree(TreeNode root) {
        if (root==null) {
            return root;
        } else {
            TreeNode treeNode = root.left;
            root.left = root.right;
            root.right = treeNode;
            root.left = invertTree(root.left);
            root.right =invertTree(root.right);  
        }
        return root;
    }
}
```




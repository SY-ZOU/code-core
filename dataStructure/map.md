## 概述

> 图是由一组顶点和一组能够将两个顶点相连的边组成的
>
> 有向图和无向图

### 1. 术语

- 相邻顶点：当两个顶点通过一条边相连时，我们称这两个顶点是相邻的，并且称这条边依附于这两个顶点。
- 度：某个顶点的度就是依附于该顶点的边的个数。[有向图分为入度和出度]
- 边的数量=所有顶点的度数/2。
- 子图：是一幅图的所有边的子集(包含这些边依附的顶点)组成的图。
- 路径：是由边顺序连接的一系列的顶点组成 。
- 环：是一条至少含有一条边且终点和起点相同的路径。
- 连通图：如果图中任意一个顶点都存在一条路径到达另外一个顶点，那么这幅图就称之为连通图。
- 连通子图：一个非连通图由若干连通的部分组成，每一个连通的部分都可以称为该图的连通子图。
- 强连通图：有向图中i到j和j到i都有路径。
- 强连通分量：非强连通图中最大的连通子图。

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/ruAMsa53pVQWN7FLK88i5lB9h1oo7JsNlTlYpI1UpBTQwhzIozqUHg7ShtDzGTuWoCMOyzYrAAYHtSX49U2BcTKUKvPMOy00q*gbPQ0.2VM!/b&bo=fgS.AQAAAAADB.c!&rf=viewer_4)

### 2. 存储结构

#### （1）邻接矩阵

- 使用一个V*V的二维数组int[V][V] adj,把索引的值看做是顶点。
- 如果顶点v和顶点w相连，我们只需要将adj[v][w]和adj[w][v]的值设置为1,否则设置为0即可。

> 邻接矩阵的空间复杂度是V^2的，如果我们处理的问题规模比较大的话，内存空间极有可能 不够用。
>
> 邻接矩阵特别适合存储稠密图

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcaTA*eP9BLKLjGHLV6dU0SWRV3ln.yMKsW3lXazXOJtAcrpNkNpprJEsinrtGCR0281IPYrOHDskoaVhbiIxcE8!/b&bo=7AQKAwAAAAADN*M!&rf=viewer_4)

```c
//c语言实现
#define MAXV 20
typedef struct{
	int no;//顶点编号
	//其他信息等
}VertexType;

typedef struct{
	int edges[MAXV][MAXV];//边
	int v,e;
	VertexType vexs[MAXV];
}Graph;
```

```java
//java实现
//VertexType类： private int no;//编号  private String info;//信息
public class Graph1 {
    private ArrayList<VertexType> vertexList;//类表示顶点，编号有序
    private int[][] edges; //边的临接矩阵
    private int e;//边的数目

    public Graph1(int n){
        edges = new int[n][n];
        vertexList = new ArrayList<>(n);
        e = 0;
    }

    //添加边,v1,v2代表顶点下标
    public void insertEdge(VertexType v1, VertexType v2,int weight){
        edges[v1.getNo()][v2.getNo()] = weight;
        edges[v2.getNo()][v1.getNo()] = weight;
        e++;
    }
    //返回结点个数
    public int getNumOfVertex(){return vertexList.size();}
    //返回边个数
    public int getNumOfEdges(){return e;}
}
```

#### （2）邻接表

- 使用一个大小为V的数组 Queue[V] adj，把索引看做是顶点。
- 每个索引处adj[v]存储了一个队列，该队列中存储的是所有与该顶点相邻的其他顶点

> 邻接表的空间并不是是线性级别的，所以后面我们一直采用邻接表这种存储形式来表示图。
>
> 适合稀疏图。

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcaTA*eP9BLKLjGHLV6dU0SW4lhZr8aY93YISf*65KK0cz8XHjuZVvun.qi838ieXw0uOIU8h5ZC07B5Spt.peoI!/b&bo=4ARuAwAAAAADN5s!&rf=viewer_4)

```c
//c语言实现
#define MAXV 20
typedef struct Anode{
	VertexType node;//边顶点信息
	struct Anode next;//下一个边结点
  int weight;//权重
	//其他信息等
}ArcNode;

typedef struct Vnode{
	VertexType node;//头顶点信息
	ArcNode first;//第一个边结点
}VNode；

typedef struct{
	Vnode adjlist[MAXV];
	int v,e;
}Graph;
```

```java
//java实现
public class Graph2 {
    private ArrayList<VertexType> vertexList;//类表示顶点，编号有序
    private int e; //边的数目
    private Queue<ArcNodeType>[] adjList;//邻接表的队列，边结点单独一个类存权重等

    public Graph2(int v){ //初始化顶点数量
        vertexList = new ArrayList<>(v);
        e=0;//初始化边的数量
        adjList = new Queue[v];
        //初始化邻接表中的空队列
        for (int i = 0; i < adjList.length; i++) {
            adjList[i] = new LinkedList<ArcNodeType>();
        }
    }

    //获取顶点数目
    public int getV(){ return vertexList.size(); }
    //获取边的数目
    public int getE(){ return e; }
    //向图中添加一条边 v-w
    public void addEdge(VertexType v1, VertexType v2, int weight) {
        adjList[v1.getNo()].offer(new ArcNodeType(v2.getNo(),weight)); 
        adjList[v2.getNo()].offer(new ArcNodeType(v1.getNo(),weight)); 
        e++;
    }
    //获取和顶点v相邻的所有顶点
    public Queue<ArcNodeType> adj(VertexType v){ return adjList[v.getNo()]; }
}
```

## 图的搜索

### 1. 深度优先搜索DFS

> Depth first search

- 访问初始结点，并设置为访问。
- 查找结点v第一个临接结点w。
- 如果w不存在，回到1，访问v下一个结点。
- 如果w存在且未访问，对w进行递归，执行123。
- 查找v临接结点w的下一个结点，回到3。

#### （1）临接矩阵表示方法





- 临接表表示方法

```c
//c语言递归
int visited[MAXV] = {0};

void DFS(Graph* g , int v){
	visited[v] = 1 ;//访问该结点
	ArcNode p = g->adjlist[v].first; //指向此顶点第一个边结点
	if(p!=null){
		int no = v->no;
		if(visited[no]==0){
			DFS(g,no);
		}
		p = p - >next;
	}
}
```



### 2. 广度优先搜索BFS



```c
//c语言队列
void BFS(Graph* g , int v){
	int queue[MAXV], front = 0,rear = 0;//定义循环队列
	int visited[MAXV] = {0};
	visited[v] = 1;
	rear = (rear+1) % MAXV;
	queue[rear] = v;//该点进队列
	while(front!=rear){ //队列不空时
		front = (front+1)%MAXV;//出队列
		int w = queue[front] ;
		ArcNode p = g->adjlist[w].first;//指向此顶点第一个边结点
		while(p!=null){
			if(visited[p.no]==0){
				visited[p.no]=1;
			}
			rear = (rear+1) % MAXV;
			queue(rear) = p.no;//进队列
			p = p.next;
		}
	}
}
```






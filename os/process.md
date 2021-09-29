## 进程与线程

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/u0qUMsilGdkhXscKj.lHLoR*cwcLKr8FGo.XoRk99pKGOv9PuUmES2ZMDiPaM*CvkrpUYSINrgsYRPD9IcSrLQ!!/b&bo=HAIAAQAAAAADBz0!&rf=viewer_4)

#### 1. 进程

**进程是正在执行的一个程序或命令，每一个进程都是运行的实体，有自己的地址空间，占用一定的系统资源。**

**进程是资源分配的基本单位**，用来管理资源（例如：内存，文件，网络等资源）。

进程特点：`并发`，`异步`，`动态`，`独立`。

> 进程控制块 (Process Control Block, PCB) 描述进程的基本信息和运行状态，**所谓的创建进程和撤销进程，都是指对 PCB 的操作。（PCB是描述进程的数据结构）**。
>
> `程序段`、`相关的数据段`和`PCB`三部分便构成了进程实体。
>
> **PCB一定在内存，不在外存，要通过它挂起激活进程。是进程在系统存在的唯一标识，系统根据PCB感知进程**
>
> PCB组织方式：链接，索引，多级队列（就绪，阻塞都是队列）

#### 2. 线程

线程是独立调度的基本单位。一个进程中可以有多个线程，它们**共享进程资源**。

QQ 和浏览器是两个进程，浏览器进程里面有很多线程，例如 HTTP 请求线程、事件响应线程、渲染线程等等，线程的并发执行使得在浏览器中点击一个新链接从而发起 HTTP 请求时，浏览器还可以响应用户的其它事件。

#### 3. 区别

**拥有资源：进程是资源分配的基本单位，但是线程不拥有资源，线程可以访问隶属进程的资源。**

**调度：线程是独立调度的基本单位，在同一进程中，线程的切换不会引起进程切换，从一个进程内的线程切换到另一个进程中的线程时，会引起进程切换。**

**系统开销：由于创建或撤销进程时，系统都要为之分配或回收资源，如内存空间、I/O 设备等，所付出的开销远大于创建或撤销线程时的开销。类似地，在进行进程切换时，涉及当前执行进程 CPU 环境的保存及新调度进程 CPU 环境的设置，而线程切换时只需保存和设置少量寄存器内容，开销很小。**

**通信方面：进程间通信 (IPC) 需要进程同步和互斥手段的辅助，以保证数据的一致性。而线程间可以通过直接读/写同一进程中的数据段（如全局变量）来进行通信。**

## 进程生命周期

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/u0qUMsilGdkhXscKj.lHLu9nLLmXlOsNi795RKah0R838PHiGfdVqHBbR*nGwNvD8s90YqBNf7F4HoMspafc6A!!/b&bo=KgIJAQAAAAARBxA!&rf=viewer_4)

- 就绪（Ready）状态：当进程已分配到除CPU以外的所有必要资源后，只要再获得CPU，便可立即执行。时间片完也会就绪。存入就绪队列。
- 执行状态：进程已获得CPU，其程序正在执行。
- 阻塞状态：正在执行的进程由于发生某事件而暂时无法继续执行时，便放弃处理机而处于暂停状态，根据阻塞原因，系统中设置**多个阻塞队列**。（注意多个根据原因分配）

- 创建状态：如果进程所需资源尚不能得到满足，如无足够的内存空间，创建工作将无法完成，进程不能被调度，此时的进程状态为创建状态。但是该资源不包括 CPU 时间，缺少 CPU 时间会从运行态转换为就绪态。
- 终止状态：一个进程到达了自然结束点，或者出现了无法克服的错误，或是被操作系统所终结，或是被其他有终止权的进程所终结，它将进入终止状态。
- 挂起状态：使执行的进程暂停执行,静止下来,我们把这种静止状态称为挂起状态。（处理器速度快于IO，可能等待IO）他不能立即执行，必须要到活跃状态，**因为它的数据在外存，要先调到内存**。**自身，父进程，OS可以使其挂起。只有挂起它的进程能解除挂起**（负荷调节等用处）。优先挂起阻塞进程可以释放内存。PCB里面是状态信息。**交换（swap）技术：把不用的程序掉到外存，或掉进来。**

#### 1. 进程切换

- 保存上下文到PCB（寄存器，指针等）
- 更新PCB状态，将其状态改为就绪或阻塞状态
- 把PCB移到相应的队列（阻塞队列或就绪队列）
- 调度程序调度队列的下一个进程，选中的PCB状态也要修改（running）
- 修改内存空间，恢复被选中进程的现场，读取PCB上下文（寄存器，指针等）

#### 2. 进程创建

- 申请空白PCB：主进程表增加一个项，申请进程id 
- 为新进程分配资源：用户地址，用户栈空间，PCB空间
- 初始化进程控制块PCB：进程标识，处理机状态，进程状态

- 将新进程插入就绪队列，启动调度。

#### 3. 进程终止

- 根据被终止进程的PID找到它的PCB，从中读出该进程的状态。
- 若被终止进程正处于执行状态，应立即终止该进程的执行，重新进行调度。
- 若该进程还有子孙进程，立即将其所有子孙进程终止。
- 将被终止进程所拥有的全部资源，归还给其父进程，或者归还给系统。
- 将被终止进程的PCB从所在队列中移出。

#### 4. 阻塞/唤醒

- 正在执行的进程，由于无法继续执行，于是**进程便通过调用阻塞原语block把自己阻塞；（自己阻塞）**
- 把进程控制块中的现行状态由“执行”改为阻塞，并将PCB插入阻塞队列；
- 转调度程序进行重新调度，将处理机分配给另一就绪进程，并进行切换。

- 当被阻塞进程所期待的事件出现时，则由有关进程调用唤醒原语wakeup( )，将等待该事件的进程唤醒。
- 唤醒原语执行的过程是：首先把被阻塞的进程从等待该事件的阻塞队列中移出，将其PCB中的现行状态由阻塞改为就绪，然后再将该PCB插入到就绪队列中。

#### 5. 挂起/激活

- 首先检查被挂起进程的状态，若处于活动就绪状态，便将其改为静止就绪；对于活动阻塞状态的进程，则将之改为静止阻塞；进程从内存切换到外存。
- 然后将被挂起进程的PCB复制到指定的内存区域。

- 当发生激活进程的事件时，例如，父进程或用户进程请求激活指定进程，系统将利用激活原语active( )将指定进程激活 

- **从外存调回内存**（位置可能不一样）

## 进程调度

> 不同环境的调度算法目标不同，因此需要针对不同环境来讨论调度算法。

- 高级调度：把外存上处于后备队列（作业队列）中的那些作业调入内存，也就是说，它的调度对象是作业

- 低级调度：它所调度的对象是进程(或内核级线程)

- 中级调度：暂时不能运行的进程调至外存上去等待，把此时的进程状态称为就绪驻外存状态或挂起状态。

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/u0qUMsilGdkhXscKj.lHLsqOwFnal*Nkldse*jST8jUjbi7P0RUsV3sdgvxVRBby1Ky.c.MTS0G*eQg1N7kqFQ!!/b&bo=AgZoAwAAAAADB00!&rf=viewer_4)

#### 1. 批处理系统

> 批处理系统没有太多的用户操作，，调度算法目标是保证吞吐量和周转时间（从提交到终止的时间）。

##### 1.1 先来先服务（FCFS）

- **非抢占式的调度算法，按照请求的顺序进行调度。**
- **有利于长作业，但不利于短作业，**因为短作业必须一直等待前面的长作业执行完毕才能执行，而长作业又需要执行很长时间，造成了短作业等待时间过长。

| 作业号 | 提交时间 | 运行时间 | 开始时间 | 等待时间 | 完成时间 | 周转时间 | 带权周转时间 |
| ------ | -------- | -------- | -------- | -------- | -------- | -------- | ------------ |
| 1      | 8        | 2        | 8        | 0        | 10       | 2        | 1            |
| 2      | 8.4      | 1        | 10       | 1.6      | 11       | 2.6      | 2.6          |
| 3      | 8.8      | 0.5      | 11       | 2.2      | 11.5     | 2.7      | 5.4          |
| 4      | 9        | 0.2      | 11.5     | 2.5      | 11.7     | 2.7      | 13.5         |

```c
//对各进程按照到达时间进行排序，挑选最先到达的进程一次性执行完毕，判断是否所有进程都被调度，若是则结束，否则返回挑选最先到达的进程一次性执行完毕步骤，继续执行后续程序。

//排序: 按照进程的arrivetime（从小到大）对pcb数组中的N个进程进行排序 
void sort(pcb *p, int N)   
{     
	for(int i=0; i < N-1; i++)
	{
		for(int j=0; j<N-1-i; j++)  
		{
			if(p[j].arrivetime > p[j+1].arrivetime)   
			{    
				pcb temp;    
				temp=p[j];    
				p[j]=p[j+1];    
				p[j+1]=temp;    
			}
		}
	}
} 
//运行
void run(pcb *p, int N)     
{
	int k;    
	for(k=0; k<N; k++)    
	{    
		if(k==0) //第1个进程   
		{     
			p[k].starttime = p[k].arrivetime; //第1个进程到达之后即可执行   
			p[k].finishtime = p[k].starttime + p[k].servicetime; 
		}    
		else 
		{    
			p[k].starttime = (p[k-1].finishtime >= p[k].arrivetime)? p[k-1].finishtime: p[k].arrivetime;    
			p[k].finishtime = p[k].starttime + p[k].servicetime;
		}    
	}    
	for(k=0; k<N; k++)    
	{    
		p[k].zztime = p[k].finishtime - p[k].arrivetime;    
		p[k].dqzztime = p[k].zztime / p[k].servicetime;    
     }    
} 
```

##### 1.2 短作业优先（SJF）

- 非抢占式的调度算法，按估计运行时间最短的顺序进行调度。
- **长作业有可能会饿死，**处于一直等待短作业执行完毕的状态。因为如果一直有短作业到来，那么长作业永远得不到调度。

| 作业号 | 提交时间 | 运行时间 | 开始时间 | 等待时间 | 完成时间 | 周转时间 | 带权周转时间 |
| ------ | -------- | -------- | -------- | -------- | -------- | -------- | ------------ |
| 1      | 8        | 2        | 8        | 0        | 10       | 2        | 1            |
| 2      | 8,4      | 1        | 10.7     | 2.3      | 11.7     | 3.3      | 3.3          |
| 3      | 8.8      | 0.5      | 10.2     | 1.4      | 10.7     | 1.9      | 3.8          |
| 4      | 9        | 0.2      | 10       | 1        | 10.2     | 1.2      | 6            |

```c
//查找当前已经到达的最短进程，调用该进程，判断是否所有进程已经结束，若是则结束，否则，返回最初步骤继续运行。每次选出最短的进程进行调度，调度完毕则淘汰，直到所有进程都调度完毕。

//***优先级排序***
void sort(pcb *p, int N)   
{   	
	/*
	1、对pcb型数组中的元素进行一个简单的排序，找到优先到达的进程
	方便后续工作
	*/	
	//冒泡排序: N-1次循环,每次从p[0]到p[N-1-i]中找到最先到达的进程，放到p[N-1-i]
	//该循环结束就得到按照到达时间排序的结果	
	for(int i=0;i<N-1;i++)  
	{
		//排序规则： 1、arrivetime越大，排序越往后  2、arrivetime相同时，servicetime短的优先 
		for(int j=0;j<N-1-i;j++) 
		{		
			if(p[j].arrivetime>p[j+1].arrivetime || (p[j].arrivetime==p[j+1].arrivetime && p[j].servicetime>p[j+1].servicetime) )   
			{  
				//p[j+1]优先p[j]到达,交换p[j]和p[j+1]
				pcb temp;   
				temp = p[j];   
				p[j] = p[j+1];   
				p[j+1] = temp;   
            } 
		}
	}
	/*
	2、每个进程运行完成之后，找到当前时刻已经到达的最短进程
	P[0]优先级最高，p[0].finishtime=p[0].arrivetime+p[0].servicetime
	m!=0时：p[m].finishtime=p[m-1].finishtime+p[m].servicetime
	   或： p[m].finishtime=p[m].arrivetime+p[m].servicetime（m-1进程执行完时，m进程还未到达） 
	*/
	for(int m=0; m<N-1; m++)      
	{
		if(m == 0)   
			p[m].finishtime = p[m].arrivetime + p[m].servicetime;   
		else
			p[m].finishtime = ((p[m-1].finishtime >= p[m].arrivetime)? p[m-1].finishtime: p[m].arrivetime) + p[m].servicetime;
		
		//(1)找到p[m].finishtime时刻哪些进程已经到达
		int i=0;  //i统计 p[m].finishtime时刻有几个进程已经到达
		//从下一个进程p[m+1]开始寻找
		for(int n = m+1; n <= N-1; n++)   
		{
			if(p[n].arrivetime <= p[m].finishtime)              
				i++;   
			else
				break;
			    /*由于在第1步已经对进程按照到达时间进行了排序
			      故：当p[n].arrivetime > p[m].finishtime时，
				      说明p[n]进程和其后面的其他进程都未到达。
				      不需再进行后续循环继续判断 
			   */
		}  
		
		//(2)找到p[m].finishtime时刻已经到达的最短进程，当前进程为p[m] 
		float min = p[m+1].servicetime;   //next进程服务时间为p[m+1].servicetime （初值） 
		int next = m+1;                   //next进程为m+1 （初值） 
		//p[m+1]至p[m+i]这i个已到达进程中找到最短进程
		for(int k = m+1; k < m+i; k++)       //循环体每次判断为k+1,k+1为m+2 ~m+i,所以，k为m+1 ~ m+i-1   
		{   
			//min的初值是p[m+1].servicetime, k+1为m+2 ~m+i 
			if(p[k+1].servicetime < min)   
			{
				min = p[k+1].servicetime;              
				next = k+1;
			}                              
		}  
		
		//(3)把最短进程放在p[m+1]进程处
		pcb temp;               
		temp=p[m+1];              
		p[m+1]=p[next];              
		p[next]=temp;           
	}    
} 

//***运行***
void run(pcb *p, int N)   
{
	int k; 
	//计算各进程的开始时间和结束时间
	for(k=0; k < N; k++)   
     {            
		if(k==0) //第1个进程   
		{     
			p[k].starttime = p[k].arrivetime; //第1个进程到达之后即可执行   
			p[k].finishtime = p[k].starttime + p[k].servicetime; 
		}    
		else 
		{    
			p[k].starttime = (p[k-1].finishtime >= p[k].arrivetime)? p[k-1].finishtime: p[k].arrivetime;    
			p[k].finishtime = p[k].starttime + p[k].servicetime;
		}    
	} 
	//计算各进程的周转时间和带权周转时间
	for(k=0; k< N; k++)   
	{        
		p[k].zztime = p[k].finishtime - p[k].arrivetime;   
		p[k].dqzztime = p[k].zztime / p[k].servicetime;   
     }   
}  
```

##### 1.3 最短剩余时间优先（SRTN）

- **最短作业优先的抢占式版本，**按剩余运行时间的顺序进行调度。 当一个新的作业到达时，其整个运行时间与当前进程的剩余时间作比较。如果新的进程需要的时间更少，则挂起当前进程，运行新的进程。否则新的进程等待。

#### 2. 交互式系统

>  交互式系统有大量的用户交互操作，在该系统中调度算法的目标是快速地进行响应。

##### 2.1 时间片轮转

- 将所有就绪进程按 FCFS （先来先服务） 的原则排成一个队列，每次调度时，把 CPU 时间分配给队首进程，该进程可以执行一个时间片。当时间片用完时，由计时器发出时钟中断，调度程序便停止该进程的执行，并将它送往就绪队列的末尾，同时继续把 CPU 时间分配给队首的进程。

- **时间片轮转算法的效率和时间片的大小有很大关系。因为进程切换都要保存进程的信息并且载入新进程的信息，如果时间片太小，会导致进程切换得太频繁，在进程切换上就会花过多时间。而如果时间片过长，那么实时性就不能得到保证。**

##### 2.2 优先级调度

- 为每个进程分配一个优先级，按优先级进行调度。(可抢占与不可抢占)

- 为了防止低优先级的进程永远等不到调度，可以随着时间的推移增加等待进程的优先级。

##### 2.3 多级反馈队列

- 一个进程需要执行 100 个时间片，如果采用时间片轮转调度算法，那么需要交换 100 次。多级队列是为这种需要连续执行多个时间片的进程考虑，它设置了多个队列，每个队列时间片大小都不同，例如 1,2,4,8,..。进程在第一个队列没执行完，就会被移到下一个队列。这种方式下，之前的进程只需要交换 7 次。

- 每个队列优先权也不同，最上面的优先权最高。因此只有上一个队列没有进程在排队，才能调度当前队列上的进程。

- 可以将这种调度算法看成是时间片轮转调度算法和优先级调度算法的结合。

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/u0qUMsilGdkhXscKj.lHLgdz6HKkwdTryfnxcnGCHnCGU*rOTfIu9KH8sFlF7POyXD1V9p3Qs1LfNH7UnFUwbg!!/b&bo=9AEOAQAAAAARB8o!&rf=viewer_4)

#### 3. 实时系统

实时系统要求一个请求在一个确定时间内得到响应。分为**硬实时和软实时**，前者必须满足绝对的截止时间，后者可以容忍一定的超时。

在含有硬实时任务的实时系统中，广泛采用抢占机制。当一个优先权更高的任务到达时，允许暂停当前任务，而令高优先权任务立即投入运行，这样便可满足该硬实时任务对截止时间的要求。

##### 3.1**最早截止时间优先算法(EDF)**

- 根据任务的开始截止时间（或者结束截至时间）来确定任务的优先级。截止时间愈早，其优先级愈高。
- 保持一个实时任务就绪队列，该队列按各任务截止时间的早晚排序

## 进程同步

#### 1. 临界区

对临界资源进行访问的**那段代码**称为临界区。为了互斥访问临界资源，进程在进入临界区之前，需要先进行检查。

#### 2. 同步与互斥

- 同步：多个进程因为合作产生的直接制约关系，**使得进程有一定的先后执行关系。**
- 互斥：多个进程在同一时刻只有一个进程能进入临界区。

#### 3. 信号量

信号量（Semaphore）是一个整型变量，可以对其执行 down 和 up 操作，也就是常见的 P 和 V 操作。

- **down** : 如果信号量大于 0 ，执行 -1 操作；如果信号量等于 0，进程睡眠，等待信号量大于 0；（阻塞）
- **up** ：对信号量执行 +1 操作，唤醒睡眠的进程让其完成 down 操作。（唤醒）

**down 和 up 操作需要被设计成原语，不可分割，通常的做法是在执行这些操作的时候屏蔽中断。**

如果信号量的取值只能为 0 或者 1，就成为了 `互斥量（Mutex）` ，0 表示临界区已经加锁，1 表示临界区解锁。

```c
typedef int semaphore;
semaphore mutex = 1;
void P1() {
    down(&mutex);
    // 临界区
    up(&mutex);
}

void P2() {
    down(&mutex);
    // 临界区
    up(&mutex);
}
```

#### 4. 管程

`管程 (英语：Monitors，也称为监视器) `可以看做一个软件模块，**它是将共享的变量和对于这些共享变量的操作封装起来，形成一个具有一定接口的功能模块，进程可以调用管程来实现进程级别的并发控制。**

进程只能**互斥得使用管程，即当一个进程使用管程时，另一个进程必须等待。当一个进程使用完管程后，它必须释放管程并唤醒等待管程的某一个进程。**进程在无法继续执行的时候不能一直占用管程。

管程是为了解决信号量在临界区的 PV 操作上的配对的麻烦，把配对的 PV 操作集中在一起，生成的一种并发编程方法。其中使用了`条件变量`这种同步机制。

c 语言不支持管程

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/u0qUMsilGdkhXscKj.lHLhj64dn.JbI7CTGUXRjoqVmpCPg.ecs5bXhrcfWvR*1yhZMnQG3FcnbC*HnA5VRDhA!!/b&bo=dgT9AgAAAAADB68!&rf=viewer_4)

#### 4. 经典进程同步问题

##### （1）生产者/消费者

> 使用一个缓冲区来保存物品，只有缓冲区没有满，生产者才可以放入物品；只有缓冲区不为空，消费者才可以拿走物品。

```c
#define N 100
typedef int semaphore;
semaphore mutex = 1;
semaphore empty = N;
semaphore full = 0;

//不能先执行 down(mutex) 再执行 down(empty)。如果这么做了，那么可能会出现这种情况：生产者对缓冲区加锁后，执行 down(empty) 操作，发现 empty = 0，此时生产者睡眠。消费者不能进入临界区，因为生产者对缓冲区加锁了，也就无法执行 up(empty) 操作，empty 永远都为 0，那么生产者和消费者就会一直等待下去，造成死锁。
void producer() {
    while(TRUE){
        int item = produce_item(); // 生产一个产品
        // down(&empty) 和 down(&mutex) 不能交换位置，否则造成死锁
        down(&empty); // 记录空缓冲区的数量，这里减少一个产品空间
        down(&mutex); // 互斥锁
        insert_item(item);
        up(&mutex); // 互斥锁
        up(&full); // 记录满缓冲区的数量，这里增加一个产品
    }
}

void consumer() {
    while(TRUE){
        down(&full); // 记录满缓冲区的数量，减少一个产品
        down(&mutex); // 互斥锁
        int item = remove_item();
        up(&mutex); // 互斥锁
        up(&empty); // 记录空缓冲区的数量，这里增加一个产品空间
        consume_item(item);
    }
}
```

> 用管程实现

```pascal
// 管程 paacal语言
monitor ProducerConsumer
    condition full, empty;
    integer count := 0;
    condition c;

    procedure insert(item: integer);
    begin
        if count = N then wait(full);
        insert_item(item);
        count := count + 1;
        if count = 1 then signal(empty);
    end;

    function remove: integer;
    begin
        if count = 0 then wait(empty);
        remove = remove_item;
        count := count - 1;
        if count = N -1 then signal(full);
    end;
end monitor;

// 生产者客户端
procedure producer
begin
    while true do
    begin
        item = produce_item;
        ProducerConsumer.insert(item);
    end
end;

// 消费者客户端
procedure consumer
begin
    while true do
    begin
        item = ProducerConsumer.remove;
        consume_item(item);
    end
end;
```

##### （2）读者-写者问题

> 允许多个进程同时对数据进行读操作，但是不允许读和写以及写和写操作同时发生。读者优先策略

```c
count = 0;
semaphore CountMutex = 1;
semaphore WriteMutex = 1;

void writer(){
    while(true){
        down(WriteMutex);
        write();
        up(WriteMutex);
    }
}

// 读者优先策略
void reader(){
    while(true){
        down(&countmutex);
        	count++;
        	if(count == 1) down(&data_mutex); // 第一个读者需要对数据进行加锁，防止写进程访问
        up(&countmutex);
      
        read();
      
        down(&countmutex);
        	count--;
        	if(count == 0) up(&data_mutex);
        up(&count_mutex);
	}
}
```

##### （3）哲学家进餐问题

> 五个哲学家围着一张圆桌，每个哲学家面前放着食物。哲学家的生活有两种交替活动：吃饭以及思考。当一个哲学家吃饭时，需要先拿起自己左右两边的两根筷子，并且一次只能拿起一根筷子。
>
> 为了防止死锁的发生，可以设置两个条件（临界资源）：
>
> - 必须同时拿起左右两根筷子；
> - 只有在两个邻居都没有进餐的情况下才允许进餐。

```c
//1. 必须由一个数据结构，来描述每个哲学家当前的状态
#define N 5
#define LEFT i // 左邻居
#define RIGHT (i + 1) % N    // 右邻居
#define THINKING 0
#define HUNGRY   1
#define EATING   2
typedef int semaphore;
int state[N];                // 跟踪每个哲学家的状态
//2. 该状态是一个临界资源，对它的访问应该互斥地进行
semaphore mutex = 1;         // 临界区state的互斥
//3. 一个哲学家吃饱后，可能要唤醒邻居，存在着同步关系
semaphore s[N];              // 每个哲学家一个信号量

void philosopher(int i) {
    while(TRUE) {
        think();
        take_two(i);
        eat();
        put_tow(i);
    }
}

void take_two(int i) {
    P(&mutex);  // 进入临界区
    	state[i] = HUNGRY; // 我饿了
    	test(i); // 试图拿两把叉子
    V(&mutex); // 退出临界区
    P(&s[i]); // 没有叉子便阻塞,等待通知可以吃test
}

void put_tow(i) {
    P(&mutex);
    	state[i] = THINKING;
    	test(LEFT); 
    	test(RIGHT);//即使是i身边两个都可以用餐，也只会通知左边，左边后又阻塞
    V(&mutex);
}

void test(i) {         // 尝试拿起两把筷子
    if(state[i] == HUNGRY && state[LEFT] != EATING && state[RIGHT] !=EATING) {
        state[i] = EATING;
        V(&s[i]); // 通知第i个人可以吃饭了
    }
}
```

## 进程通信

> 进程同步：控制多个进程按一定顺序执行
>
> 进程通信：进程间传输信息

- 直接通信：发送进程直接把消息发送给接收进程，并将它挂在接收进程的消息缓冲队列上，接收进程从消息缓冲队列中取得消息。

```
Send(Receiver,message);//发送一个消息message给接收进程Receiver
Receive(Sender,message);//接收Sender进程发送的消息message
```

- 间接通信：间接通信方式是指进程之间的通信需要通过作为**共享数据结构的实体**。该实体用来暂存发送进程发给目标进程的消息。该通信方式广泛应用于计算机网络中，相应的通信系统称为电子邮件系统。

#### 1. 管道PIPE

管道是通过调用 pipe 函数创建的，fd[0] 用于读，fd[1] 用于写。

管道是一种半双工的通信方式，具有固定的读端和写端。只能在**具有亲缘关系的进程间使用**。管道是指用于连接一个读进程和一个写进程以实现它们之间通信的一个共享文件（pipe文件）。

```c
#include <unistd.h>
int pipe(int fd[2]);//创建管道
char outpipe[100],inpipe[100];
while((p1=fork())==-1);        //创建子进程
if(p1==0)
{
       lockf(fd[1],1,0);        //锁定管道写入端
       sprintf(outpipe,"child1 is send message!");         //定义发送缓冲区
       write(fd[1],outpipe,50);  //写入管道
       sleep(1);
       lockf(fd[1],0,0);            //释放管道
       exit(0);                //子进程终止
}
 else
{

       wait(0);           //同步 等待子进程终止
       read(fd[0],inpipe,50);
       printf("%s/n",inpipe);
       exit(0);

 }
```

- 只支持半双工通信（单向交替传输）；指当写(输入)进程把一定数量(如4 KB)的数据写入pipe，便去睡眠等待， 直到读(输出)进程取走数据后，再把他唤醒。当读进程读一空pipe时，也应睡眠等待，直至写进程将数据写入管道后，才将之唤醒。
- 只能在父子进程或者兄弟进程中使用。
- 互斥，即当一个进程正在对pipe执行读/写操作时，其它(另一)进程必须等待。

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/u0qUMsilGdkhXscKj.lHLuXEqaJV3GTA1EYCIDnTe6uOXWGhDFILmAD2sluvl2FZfzhkVP.QQ2Q.SDLDyA.OXA!!/b&bo=AQIZAQAAAAADBzk!&rf=viewer_4)

#### 2. 命名管道FIFO

命名管道也是半双工的通信方式，去除了管道只能在父子进程中使用的限制。常用于客户进程服务器进程通信。

```c
#include <sys/stat.h>
int mkfifo(const char *path, mode_t mode);
int mkfifoat(int fd, const char *path, mode_t mode);
```

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/Ub*4aHxmnNgAqL6c0lJ4K.TQWAlfXPrG48SbIkXCgLrBzdM8ZlepOV6ublY*Rd7O3jo*AkEVDksw0D2W7Vnab7witSe1RbW3TwVTaTgptVg!/b&bo=fwK8AQAAAAADF*I!&rf=viewer_4)

#### 3. 消息队列

消息队列是消息的链表，存放在内核中并由消息队列标识符标识。

独立于读写进程。送者将消息发送到消息队列中（指明type，接收方就可以通过type取）。

接收者从消息队列中获取一条消息。如果消息队列已满，消息将不被写入队列

间接（内核）,相比于 FIFO，消息队列具有以下优点：

- 消息队列可以独立于读写进程存在，从而避免了 FIFO 中同步管道的打开和关闭时可能产生的困难；
- 避免了 FIFO 的同步阻塞问题，不需要进程自己提供同步方法；
- 读进程可以根据消息类型有选择地接收消息，而不像 FIFO 那样只能默认地接收。

#### 4. 信号量

它是一个计数器，用于为多个进程提供对共享数据对象的访问。

#### 5. 共享内存

允许多个进程共享一个给定的存储区。因为数据不需要在进程之间复制，所以这是最快的一种 IPC。

需要使用信号量用来同步对共享存储的访问。

多个进程可以将同一个文件映射到它们的地址空间从而实现共享内存。另外 XSI 共享内存不是使用文件，而是使用使用内存的匿名段。

#### 6. 套接字

与其它通信机制不同的是，它可用于不同机器间的进程通信。

## 死锁

多个线程或进程对同一个资源的争抢或相互依赖

### 1. 必要条件

- 互斥：每个资源要么已经分配给了一个进程，要么就是可用的。
- 占有和等待：已经得到了某个资源的进程可以再请求新的资源。
- 不可抢占：已经分配给一个进程的资源不能强制性地被抢占，它只能被占有它的进程显式地释放。
- 循环等待：有两个或者两个以上的进程组成一条环路，该环路中的每个进程都在等待下一个进程所占有的资源。

### 2. 解决方法

#### （1）鸵鸟策略

把头埋在沙子里，假装根本没发生问题。

因为解决死锁问题的代价很高，因此鸵鸟策略这种不采取任务措施的方案会获得更高的性能。

当发生死锁时不会对用户造成多大影响，或发生死锁的概率很低，可以采用鸵鸟策略。

大多数操作系统，包括 Unix，Linux 和 Windows，处理死锁问题的办法仅仅是忽略它。

#### （2）死锁检测与死锁恢复

不试图阻止死锁，而是当检测到死锁发生时，采取措施进行恢复。

- 每种类型一个资源的死锁检测

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/ruAMsa53pVQWN7FLK88i5tVXtP43JxVZ.yy1AO5ndxjx1pK4cd91WKkDIz3ccKQRXb.Y*xWd3ouXNi*oXA9EW*f0Rh2MfyqtS*.IFXoaSws!/b&bo=VQIxAQAAAAADB0U!&rf=viewer_4)

上图为资源分配图，其中方框表示资源，圆圈表示进程。资源指向进程表示该资源已经分配给该进程，进程指向资源表示进程请求获取该资源。

图 a 可以抽取出环，如图 b，它满足了环路等待条件，因此会发生死锁。

每种类型一个资源的死锁检测算法是通过检测有向图是否存在环来实现，从一个节点出发进行深度优先搜索，对访问过的节点进行标记，如果访问了已经标记的节点，就表示有向图存在环，也就是检测到死锁的发生。

- 每种类型多个资源的死锁检测

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcd1O6mk30u1WGZRaiV1iPs.tkU0Nkm15Ad4golBevDnM7LI6sZ2O5Mxf6QW.2SqJ86OVIHPu5WEIfEM.TMsA51c!/b&bo=wAHcAAAAAAADFy8!&rf=viewer_4)

上图中，有三个进程四个资源，每个数据代表的含义如下：

- E 向量：资源总量
- A 向量：资源剩余量
- C 矩阵：每个进程所拥有的资源数量，每一行都代表一个进程拥有资源的数量
- R 矩阵：每个进程请求的资源数量

每个进程最开始时都不被标记，执行过程有可能被标记。当算法结束时，任何没有被标记的进程都是死锁进程。

1. 寻找一个没有标记的进程 Pi，它所请求的资源小于等于 A。
2. 如果找到了这样一个进程，那么将 C 矩阵的第 i 行向量加到 A 中，标记该进程，并转回 1。
3. 如果没有这样一个进程，算法终止。

#### （3）死锁预防

- 破坏互斥条件：例如假脱机打印机技术允许若干个进程同时输出，唯一真正请求物理打印机的进程是打印机守护进程。

- 破坏占有和等待条件：一种实现方式是规定所有进程在开始执行前请求所需要的全部资源。

- 破坏不可抢占条件

- 破坏环路等待：给资源统一编号，进程只能按编号顺序来请求资源。

#### （4）死锁避免

- 安全状态

是指系统能按某种进程顺序，如<P1，P2，…，Pn>，依次为n个进程分配其所需资源，直至其最大需求，使每个进程都可顺利地完成，称系统处于安全状态。

- 银行家算法

数据结构

```
- 可利用资源向量Available： Available[j]=k,表示系统中现有Rj类资源k个。
- 最大需求矩阵Max：Max[i,j]=K，表示进程i需要Rj类资源的最大数目为K。
- 分配矩阵Allocation。Allocation［i,j]=K，表示进程i当前已分得Rj类资源的数目为K。
- 需求矩阵Need。Need[i,j]=K，表示进程Pi还需要Rj类资源K个，方能完成其任务。Need[i,j]＝Max[i,j]一Allocation[i,j]
```

算法

```
 设Requesti，是进程Pi的请求向量，当Pi发出资源请求后，系统按下述步骤进行检查：

- 如果Requesti[j]≤Need[i,j],便转向步骤2；否则认为出错，因为它所需要的资源数已超过它所声明的最大值。

- 如果Requesti[j]≤Available[j]，便转向步骤（3）；否则，表示尚无足够资源，Pi须阻塞等待。

- 系统试探着把资源分配给进程Pi，并修改下面数据结构中的数值
	Available[j]：＝ Available[j] - Requesti[j]； Allocation[i,j] : ＝ Allocation[i,j] ＋ 				Requesti[j]; Need[i,j]: =Need[i,j] - Requesti[j]； 

- 系统执行安全性算法，检查此次资源分配 后，系统是否处于安全状态。若安全，才正式将资源分配给进程Pi，以完成本次分配。否则，将本次的试探分配作废，恢复原来的资源分配状态，让进程Pi等待。

 安全性算法

- 设置两个向量：
	①工作向量Work:它表示系统可提供给进程继续运行所需的各类资源数目，初始值Work:＝Available.
	②设置数组Finish[n]：它表示系统是否有足够的资源分配给进程，使之运行完成。初始值Finish[i]:=false；当		Finish[i]:=true时，进程pi可获得其所需的全部资源，从而顺利执行完成。
	
- 从进程集合中找到一个能满足下述条件的进程： ①Finish[i]=false； ②Need[i,j]≤work；若找到，执行步骤（3),否则，执行步骤（4）。

- 当进程pi获得资源后，顺利执行直至完成，并释放出分配给它的资源，故应执行：
	Work：= Work+Allocation[i,j]；
	Finish[i] ：=true; go  to  step  2；
	
- 如果所有进程的Finish[i]＝true都满足，则表示系统处于安全状态；否则，系统处于不安全状态。
```

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcWc.n.P4sXB8hV4o05q2yZzow4szpiJ7Wgpsw4Ogw0vHeA.XVUvj*zY99N9uZh.E7JXOlHfjUsegeD3tpU077u4!/b&bo=MAbaAgAAAAADN*w!&rf=viewer_4)

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcWc.n.P4sXB8hV4o05q2yZwYJ0r*YgU59bzmkYM17d6O9lyoYEK*w433kr8KedK1mjMoHSyBOgQAOxGtE03FQdg!/b&bo=KAZwAwAAAAADN08!&rf=viewer_4)


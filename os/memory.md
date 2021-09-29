## 虚拟内存

> 虚拟内存的目的是为了让物理内存扩充成更大的逻辑内存，从而让程序获得更多的可用内存。

`主存`：看作是一个由M个连续的字节大小的单元组成的数组，每个字节都是由一个唯一的物理地址，第一个字节的地址为0接下来的地址为1以此类推。CPU访问内存的最简单的方式是使用物理寻址。

`虚拟地址`：CPU通过生成一个虚拟地址来访问主存。

`内存管理单元MMU`：CPU的专用硬件，利用存放在主存中的查询表来动态翻译虚拟地址。

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/ruAMsa53pVQWN7FLK88i5vBDr35RL0fmo1nGg2cx*YZJ4by4gBTnyPFehhRERISllrlvrPYHZO61VmflED5B2lHSTnVjBCM0OrULO5FXD7w!/b&bo=wgJrAQAAAAADB4g!&rf=viewer_4)

### 1. 处理器的寻址空间

`处理器的寻址空间`：处理器能访问多大的内存空间取决于处理器的程序计数器。对于程序计数器位数为32位的处理器而言，地址发生器所能发出的地址数量位2的32次方约4G，因此处理器能访问的最大内存空间为4G。

> 每个物理存储单元都有唯一的地址与之对应，这显然是最为理想的情况。

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcZHEZZEvGlYnKTvL3FeBktFUpYCAkbvhRinZMVXAkFKmNgqcYGIXb82EDQ2Lc.u*Bb13mb09VNWx.Bb0kFkWQdk!/b&bo=EAL0AAAAAAADF9Q!&rf=viewer_4)

> 实际上计算机所配置内存的实际空间常常会小于处理器的寻址范围，因此处理器的一部分寻址空间将没有对应的物理存储单元，从而导致处理器寻址能力的浪费。

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcZHEZZEvGlYnKTvL3FeBktF8SL.VyI4OdjTCRQ.sG4QPZJ.qbKwYoZlF334EVJLubayTsJCLzjcWyrUaFrBLqzA!/b&bo=SgLjAAAAAAADF5k!&rf=viewer_4)



> 另外还有一些处理器因外部地址线的根数小于处理器程序计数器的位数，而使地址总线的根数不满足处理器的寻址范围，从而处理器的其余存之能力也就被浪费了。

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcZHEZZEvGlYnKTvL3FeBktHrSPZ*QCPKy7MWenC3ottWgM64fl5KqPQ*S6juYrYMjrxu4X8FuyIVi4E2Xx1hAio!/b&bo=PQLeAAAAAAADF9M!&rf=viewer_4)

### 2. 虚拟内存工作原理

> 通过实践和研究证明：一个应用程序总是逐段被运行的，而且在一段时间内会稳定运行在某段程序中。因此，将需要运行的哪段程序从辅存复制到内存中运行，其它暂不运行的程序段让其仍旧保留在辅存中。当需要执行另一段尚未在未在内存的程序段时，可将内存中程序段1的副本复制回辅存，在内存中腾出必要的空间后再将辅存中的程序段2复制到内存空间中来执行即可。

`页表`：在映射工作中，为了记录程序段占用物理内存的情况，操作系统的内存管理模块需要建立一个表格，该表格以虚拟地址为索引，记录了程序段所占用的物理地址，这个虚拟地址/物理地址记录表便是存储管理单元`MMU`将虚拟地址转化为实际物理地址的依据。

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcZHEZZEvGlYnKTvL3FeBktGLvblPdFpj9i14RM*CAtkAgGIXsUFxMqn50v*WGOKxt2noSKT65InJvwSpeuKmXuU!/b&bo=dQI.AQAAAAADF3o!&rf=viewer_4)

## 分页系统

> Linux将虚拟空间分为若干个大小相等的存储分区，Linux将这样的分区称为页。为了换入换出方便，物理内存页按页的大小划分为若干块。由于物理内存中的块空间时用来容纳虚拟页的容器，所以物理内存中的块叫做页框，页与页框是Linux实现虚拟内存技术的基础。

### 1. 映射

内存管理单元（MMU）管理着地址空间和物理内存的转换，其中的页表（Page table）存储着页（程序地址空间）和页框（物理内存空间）的映射表。

Linux中页和页框的大小一般为4KB，根据系统和应用的不同页和页框的大小会有所变化。

一个虚拟地址分成两个部分，一部分存储页面号，一部分存储页内偏移量。

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcSeSwe8.9Fl*aZ3BZMJxQx9iPfhTEETuAR2owfDpDSeszW*WSke4LyjJLHL8da7*BlAtlm8O1VfHE*DFBezY6T4!/b&bo=egJ6AQAAAAADFzE!&rf=viewer_4)

> 下图的页表存放着 16 个页，这 16 个页需要用 4 个比特位来进行索引定位。因此对于虚拟地址（0010 000000000100），前 4 位是用来存储页面号，而后 12 位存储在页中的偏移量。
>
> （0010 000000000100）根据前 4 位得到页号为 2，读取表项内容为（110 1），它的前 3 为为页框号，最后 1 位表示该页在内存中。最后映射得到物理内存地址为（110 000000000100）。

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcSeSwe8.9Fl*aZ3BZMJxQx.nDkP3J1Ww4HZHNmpWweljzfJlu7892wheW8JeCzka02RWl.DUFqGzG8vkN*WDcvg!/b&bo=gwK4AgAAAAADFwk!&rf=viewer_4)

### 2. 虚拟

每个进程都有自己的4G内存空间，每个进程的内存空间都具有类似的结构。新进程建立的时候会建立自己的内存空间，进程的数据、代码等会从磁盘拷贝到自己的进程中间。每个进程已经分配的内存空间都会与对应的磁盘空间映射。

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcSeSwe8.9Fl*aZ3BZMJxQx.1ZvT9qYflFzBJ3qaPVzxlHxkvfVuvd7Gm5cxBCY89N*9pcKyJFelxagrl7.cqYb0!/b&bo=bgJNAgAAAAADFxE!&rf=viewer_4)

每创建一个进程就会为其分配4G内存，并将磁盘上的程序拷贝到进程对应的内存中，计算机中内存是有限的。每个进程的4G内存空间实际上是虚拟内存空间，每次访问内存空间的地址都需要将地址翻译为实际物理内存地址。

所有进程共享同一物理内存，每个进程只会将自己目前所需的虚拟内存空间映射到物理内存上。当进程访问某个虚拟地址时会去查看页表，如果发现对应的数据不在物理内存中则出现缺页异常

- 当不同的进程使用同样的代码时，比如库文件中的代码，物理内存中可以只存储一份这样的代码。
- 每个进程创建加载时，实际上并不立即将虚拟内存对应位置的程序数据和代码拷贝到物理内存中，只是建立好虚拟内存和磁盘文件之间的映射（存储器映射），等到运行到对应的程序时，才会通过缺页异常来拷贝数据。进程运行过程中需要动态分配内存，比如`malloc`也只是分配了虚拟内存即虚拟内存对应的页表项做相应设置，当进程真正访问到此数据时才引发缺页异常。

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcSeSwe8.9Fl*aZ3BZMJxQx8z7kEW5.e7iHDrbDR581scYO4Vz7Pb1y1yFIwxFamWNTJ5cot4rtMMd1ms9JDeqfc!/b&bo=YgKKAQAAAAADF9k!&rf=viewer_4)

### 3. 快表,多级页表

> 快表的工作原理类似于系统中的数据高速缓存(cache)，其中专门保存当前进程最近访问过的一组页表项。



## 页面置换算法

> 在程序运行过程中，如果要访问的页面不在内存中，就发生缺页中断从而将该页调入内存中。此时如果内存已无空闲空间，系统必须从内存中调出一个页面到磁盘对换区中来腾出空间。
>
> 页面置换算法和缓存淘汰策略类似，面置换算法的主要目标是使页面置换频率最低。

### 1. 最佳置换算法

所选择的被换出的页面将是最长时间内不再被访问，通常可以保证获得最低的缺页率。

是一种理论上的算法，因为无法知道一个页面多长时间不再被访问。可以用来对其他可实现算法的性能进行比较。

> 举例：一个系统为某进程分配了三个物理块，并有如下页面引用序列：70120304230321201701
>
> 开始运行时，先将 7, 0, 1 三个页面装入内存。当进程要访问页面 2 时，产生缺页中断，会将页面 7 换出，因为页面 7 再次被访问的时间最长。

### 2. 最近最久未使用LRU

在缺页中断发生时，置换未使用时间最长的页。如果我们使用最近的过去作为不远将来的近似，那么可以置换最长时间没有使用的页。

为了实现 LRU，需要在内存中维护一个所有页面的链表。当一个页面被访问时，将这个页面移到链表表头。因为每次访问都需要更新链表，因此这种方式实现的 LRU 代价很高。

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/ruAMsa53pVQWN7FLK88i5tVXtP43JxVZ.yy1AO5ndxgp*m4U.mD3W2r6WxjBzwHbI7bojPSBKZ*2Yk4rLa*7BGbNRl3ZUlhYzQ42OQI7wHE!/b&bo=GQKmAAAAAAADB58!&rf=viewer_4)

### 3. 最近未使用NRU

当页面被访问（读或写）时设置R位，页面被写入（修改）时设置M位。

当启动一个进程时，它的所有页面的两个位都由操作系统设为0，R位被定期地（比如在每次时钟中断时）清零，以区别最近没有被访问的页面和被访问的页面。当发生缺页中断时，操作系统检查所有的页面并根据它们当前的R位和M位的值，把它们分为4类：

-  第0类：没有被访问，没有被修改。
-  第1类：没有被访问，已被修改（M）。
-  第2类：已被访问，没有被修改（R）。
-  第3类：已被访问，已被修改（RM）。

NRU(Not Recently Used)算法随机地从类编号最小的非空类中挑选一个页面淘汰。在一个时间滴答中（大约20ms）淘汰一个没有被访问的已修改页面要比淘汰一个被频繁使用的“干净”页面好。NRU算法的主要优点是易于理解和能够有效地被实现，虽然它的性能不是最好的，但是已经够用了。

### 4. 先进先出FIFO

选择换出的页面是最先进入的页面。该算法会将那些经常被访问的页面换出，导致缺页率升高。

### 5. 第二次机会算法

FIFO 算法可能会把经常使用的页面置换出去，为了避免这一问题，对该算法做一个简单的修改：

当页面被访问 (读或写) 时设置该页面的 R 位为 1。需要替换的时候，检查最老页面的 R 位。如果 R 位是 0，那么这个页面既老又没有被使用，可以立刻置换掉；如果是 1，就将 R 位清 0，并把该页面放到链表的尾端，修改它的装入时间使它就像刚装入的一样，然后继续从链表的头部开始搜索。

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcd1O6mk30u1WGZRaiV1iPs9tngzTSW.3q923YT2WtP8l1YrtfYXruDvl054CCzIz3cm*yeD*f.VoGj9VFxDYW5I!/b&bo=0AI4AQAAAAADF9k!&rf=viewer_4)

### 6. 时钟CLOCK

第二次机会算法需要在链表中移动页面，降低了效率。时钟算法使用环形链表将页面连接起来，再使用一个指针指向最老的页面。

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcd1O6mk30u1WGZRaiV1iPs*5R30BSGetrpr7408i5Pg0Qz8QnWJ0XQ0Q4WJsd*EZv8D5KYP9ecyLsvJ7crXtI5I!/b&bo=NQJAAQAAAAADF0Q!&rf=viewer_4)

## 分段系统

> 分段的做法是把每个表分成段，一个段构成一个独立的地址空间。每个段的长度可以不同，并且可以动态增长。

根据编程人员的需要将程序分成代码段、数据段等独立信息段。

**分段存储管理方式却能较好地解决数据段增长 。**

在分段存储管理方式中，作业的地址空间被划分为若干个段，每个段定义了一组逻辑信息。例如，有主程序段MAIN、子程序段、数据段 及栈段。

## 段页式

程序的地址空间划分成多个拥有独立地址空间的段，每个段上的地址空间划分成大小相同的页。这样既拥有分段系统的共享和保护，又拥有分页系统的虚拟内存功能。

## 分页与分段的比较

- 对程序员的透明性：分页透明，但是分段需要程序员显式划分每个段。
- 地址空间的维度：分页是一维地址空间，分段是二维的。
- 大小是否可以改变：页的大小不可变，段的大小可以动态改变。
- 出现的原因：分页主要用于实现虚拟内存，从而获得更大的地址空间；分段主要是为了使程序和数据可以被划分为逻辑上独立的地址空间并且有助于共享和保护。
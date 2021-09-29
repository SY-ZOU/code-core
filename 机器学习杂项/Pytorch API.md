## contiguous

> contiguous一般与transpose，permute，view搭配使用：使用transpose或permute进行维度变换后，调用contiguous，然后方可使用view对维度进行变形
>
>  transpose、permute等维度变换操作后，tensor在内存中不再是连续存储的，而view操作要求tensor的内存连续存储，所以需要contiguous来返回一个contiguous copy；
>
> 维度变换后的变量是之前变量的浅拷贝，指向同一区域，即view操作会连带原来的变量一同变形，这是不合法的，所以也会报错；---- 这个解释有部分道理，也即contiguous返回了tensor的深拷贝contiguous copy数据；
>
> 只有很少几个操作是不改变tensor的内容本身，而**只是重新定义下标与元素的对应关系**。换句话说，这种操作**不进行数据拷贝和数据的改变，变的是元数据**，这些操作是：
>
> ```python3
> narrow()，view()，expand()，transpose()；
> ```
>
> 在使用transpose()进行转置操作时，**pytorch并不会创建新的、转置后的tensor**，而是**修改了tensor中的一些属性（也就是元数据），使得此时的offset和stride是与转置tensor相对应的**，而**转置的tensor和原tensor的内存是共享的**！
>
> **当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样；**

## Reshape

> 增加了torch.reshape()，与 numpy.reshape() 的功能类似，大致相当于 tensor.contiguous().view()，这样就省去了对tensor做view()变换前，调用contiguous()的麻烦